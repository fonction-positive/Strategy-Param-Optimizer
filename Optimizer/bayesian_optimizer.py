# -*- coding: utf-8 -*-
"""
贝叶斯优化器模块
基于Optuna实现多目标贝叶斯优化，支持LLM动态调整搜索空间

v2.0 更新:
- 支持正态分布采样（初始探索阶段）
- 支持并行随机探索
- 动态试验次数根据参数量调整
- 两阶段优化：探索阶段 + 利用阶段
"""

import os
import sys
import json
import warnings
import inspect
from typing import Dict, List, Any, Optional, Type, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

import optuna
from optuna.samplers import TPESampler, RandomSampler
from optuna.pruners import MedianPruner, HyperbandPruner

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backtrader as bt

from config import (
    BayesianOptConfig, DEFAULT_BAYESIAN_CONFIG,
    ParallelConfig, DEFAULT_PARALLEL_CONFIG,
    BatchParallelConfig, DEFAULT_BATCH_PARALLEL_CONFIG,
    OPTIMIZATION_OBJECTIVES, OUTPUT_DIR
)
from strategy_analyzer import StrategyAnalyzer, SearchSpaceConfig, convert_to_optuna_space
from backtest_engine import BacktestEngine, BacktestResult
from llm_client import LLMClient, get_llm_client

# 导入增强采样器
try:
    from enhanced_sampler import (
        EnhancedOptimizer, SamplerConfig,
        NormalDistributionSampler, DynamicTrialsCalculator
    )
    ENHANCED_SAMPLER_AVAILABLE = True
except ImportError:
    ENHANCED_SAMPLER_AVAILABLE = False

# 导入并行计算库
try:
    from joblib import Parallel, delayed
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
    import multiprocessing
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# 导入进度条
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


def _strategy_accepts_verbose(strategy_class: Type[bt.Strategy]) -> bool:
    """
    检查策略类的 __init__ 方法是否接受 verbose 参数
    
    Args:
        strategy_class: 策略类
        
    Returns:
        True 如果策略接受 verbose 参数，否则 False
    """
    try:
        sig = inspect.signature(strategy_class.__init__)
        return 'verbose' in sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD 
            for p in sig.parameters.values()
        )
    except Exception:
        return False


@dataclass
class OptimizationResult:
    """单个目标的优化结果"""
    objective: str
    best_params: Dict[str, Any]
    best_value: float
    backtest_result: BacktestResult
    n_trials: int
    optimization_time: float
    best_improvements: List[Dict[str, Any]] = None  # 搜索过程中每次发现更优参数的记录


class BayesianOptimizer:
    """
    贝叶斯优化器
    结合LLM和Optuna进行智能超参数优化
    """
    
    def __init__(
        self,
        config: BayesianOptConfig = None,
        llm_client: LLMClient = None,
        use_llm: bool = True,
        backtest_engine: BacktestEngine = None,
        search_space = None,
        verbose: bool = True,
        parallel_config: ParallelConfig = None,
        batch_parallel_config: BatchParallelConfig = None
    ):
        """
        初始化优化器

        Args:
            config: 贝叶斯优化配置
            llm_client: LLM客户端
            use_llm: 是否使用LLM动态调整
            backtest_engine: 外部传入的回测引擎（可选）
            search_space: 搜索空间配置（可选）
            verbose: 是否打印详细信息
            parallel_config: 并行优化配置
            batch_parallel_config: 批量并行优化配置
        """
        self.config = config or DEFAULT_BAYESIAN_CONFIG
        self.parallel_config = parallel_config or DEFAULT_PARALLEL_CONFIG
        self.batch_parallel_config = batch_parallel_config or DEFAULT_BATCH_PARALLEL_CONFIG
        self.verbose = verbose
        self.search_space = search_space
        
        # 使用外部传入的 backtest_engine 或创建新的
        self.backtest_engine = backtest_engine or BacktestEngine()
        
        # LLM 相关
        if llm_client is not None:
            self.llm_client = llm_client
            self.use_llm = use_llm and self.llm_client.check_connection()
        else:
            try:
                self.llm_client = get_llm_client()
                self.use_llm = use_llm and self.llm_client.check_connection()
            except:
                self.llm_client = None
                self.use_llm = False
        
        if self.llm_client:
            self.strategy_analyzer = StrategyAnalyzer(self.llm_client, use_llm)
        else:
            self.strategy_analyzer = None
        
        # 优化历史记录
        self.optimization_history = {}
        self.all_results = {}
        
        # 设置Optuna日志级别
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def _create_sampler(self, constant_liar: bool = False) -> optuna.samplers.BaseSampler:
        """创建采样器

        Args:
            constant_liar: 是否启用 constant_liar 模式（批量并行时应开启，
                          使 batch 内连续 ask() 不会从同一 posterior 抽样）
        """
        if self.config.sampler == "tpe":
            return TPESampler(seed=self.config.seed, constant_liar=constant_liar)
        elif self.config.sampler == "random":
            return RandomSampler(seed=self.config.seed)
        else:
            return TPESampler(seed=self.config.seed, constant_liar=constant_liar)
    
    def _create_pruner(self) -> optuna.pruners.BasePruner:
        """创建剪枝器"""
        if self.config.pruner == "median":
            return MedianPruner()
        elif self.config.pruner == "hyperband":
            return HyperbandPruner()
        else:
            return MedianPruner()
    
    def _suggest_params(
        self,
        trial: optuna.Trial,
        search_space: Dict[str, SearchSpaceConfig]
    ) -> Dict[str, Any]:
        """
        根据搜索空间配置建议参数
        
        Args:
            trial: Optuna试验对象
            search_space: 搜索空间配置
            
        Returns:
            建议的参数字典
        """
        params = {}
        
        for name, config in search_space.items():
            if config.param_type == "int":
                step = int(config.step) if config.step else 1
                params[name] = trial.suggest_int(
                    name,
                    int(config.min_value),
                    int(config.max_value),
                    step=step
                )
            elif config.param_type == "float":
                if config.distribution == "log_uniform":
                    params[name] = trial.suggest_float(
                        name,
                        config.min_value,
                        config.max_value,
                        log=True
                    )
                else:
                    step = config.step if config.step else None
                    params[name] = trial.suggest_float(
                        name,
                        config.min_value,
                        config.max_value,
                        step=step
                    )
            elif config.param_type == "bool":
                params[name] = trial.suggest_categorical(name, [True, False])
        
        return params

    def _adaptive_batch_size(self, iteration: int, total_iterations: int) -> int:
        """
        自适应批量大小策略

        Args:
            iteration: 当前迭代
            total_iterations: 总迭代次数

        Returns:
            当前迭代应使用的批量大小
        """
        if not self.batch_parallel_config.adaptive_batch:
            return self.batch_parallel_config.batch_size

        progress = iteration / total_iterations if total_iterations > 0 else 0
        max_batch = self.batch_parallel_config.max_batch_size
        min_batch = self.batch_parallel_config.min_batch_size

        if progress < 0.3:      # 前30%: 大批量探索
            return max_batch
        elif progress < 0.7:    # 中40%: 中等批量
            return max(min_batch, max_batch // 2)
        else:                   # 后30%: 小批量精细搜索
            return min_batch

    def _diverse_batch_sampling(self, study, search_space: Dict[str, SearchSpaceConfig], batch_size: int) -> List[Dict[str, Any]]:
        """
        多样化批量采样策略

        Args:
            study: Optuna study对象
            search_space: 搜索空间
            batch_size: 批量大小

        Returns:
            参数列表
        """
        params_batch = []
        diversity_count = int(batch_size * self.batch_parallel_config.diversity_ratio)

        # 主要部分: TPE 采样
        for _ in range(batch_size - diversity_count):
            trial = study.ask()
            params = self._suggest_params(trial, search_space)
            params_batch.append({'trial': trial, 'params': params})

        # 多样性部分: 添加随机扰动
        for _ in range(diversity_count):
            trial = study.ask()
            params = self._suggest_params(trial, search_space)

            # 添加随机扰动
            perturbed_params = self._add_perturbation(params, search_space)
            params_batch.append({'trial': trial, 'params': perturbed_params, 'is_perturbed': True})

        return params_batch

    def _add_perturbation(self, params: Dict[str, Any], search_space: Dict[str, SearchSpaceConfig]) -> Dict[str, Any]:
        """
        为参数添加随机扰动

        Args:
            params: 原始参数
            search_space: 搜索空间

        Returns:
            扰动后的参数
        """
        import numpy as np

        perturbed_params = params.copy()
        strength = self.batch_parallel_config.perturbation_strength

        for name, value in params.items():
            if name in search_space:
                space_config = search_space[name]
                param_range = space_config.max_value - space_config.min_value

                if space_config.param_type == "int":
                    # 整数参数：随机扰动±范围的strength倍
                    perturbation = np.random.randint(-max(1, int(param_range * strength)),
                                                     max(1, int(param_range * strength)) + 1)
                    new_value = max(space_config.min_value,
                                   min(space_config.max_value, value + perturbation))
                    perturbed_params[name] = int(new_value)
                else:
                    # 浮点参数
                    perturbation = np.random.uniform(-param_range * strength, param_range * strength)
                    new_value = max(space_config.min_value,
                                   min(space_config.max_value, value + perturbation))
                    perturbed_params[name] = new_value

        return perturbed_params

    def _build_distributions(self, search_space: Dict[str, SearchSpaceConfig]) -> Dict[str, optuna.distributions.BaseDistribution]:
        """
        根据搜索空间构建 Optuna distributions 字典，用于 create_trial。

        Args:
            search_space: 搜索空间配置

        Returns:
            参数名 -> Optuna Distribution 的映射
        """
        distributions = {}
        for name, config in search_space.items():
            if config.param_type == "int":
                step = int(config.step) if config.step else 1
                distributions[name] = optuna.distributions.IntDistribution(
                    low=int(config.min_value),
                    high=int(config.max_value),
                    step=step,
                )
            elif config.param_type == "float":
                if config.distribution == "log_uniform":
                    distributions[name] = optuna.distributions.FloatDistribution(
                        low=config.min_value,
                        high=config.max_value,
                        log=True,
                    )
                else:
                    step = config.step if config.step else None
                    distributions[name] = optuna.distributions.FloatDistribution(
                        low=config.min_value,
                        high=config.max_value,
                        step=step,
                    )
            elif config.param_type == "bool":
                distributions[name] = optuna.distributions.CategoricalDistribution(
                    choices=[True, False]
                )
        return distributions

    def _batch_parallel_exploitation(
        self,
        study,
        strategy_class: Type[bt.Strategy],
        data: pd.DataFrame,
        search_space: Dict[str, SearchSpaceConfig],
        objective: str,
        exploitation_trials: int,
        history_list: List[Dict],
        best_improvements: List[Dict],
        best_exploration_value: float,
        verbose: bool = True
    ) -> float:
        """
        批量并行利用阶段优化

        Args:
            study: Optuna study对象
            strategy_class: 策略类
            data: 回测数据
            search_space: 搜索空间
            objective: 优化目标
            exploitation_trials: 利用阶段试验次数
            history_list: 历史记录列表
            best_improvements: 最优改进记录
            best_exploration_value: 探索阶段最优值
            verbose: 是否详细输出

        Returns:
            最终最优值
        """
        # 计算并行配置
        if self.batch_parallel_config.hybrid_mode:
            # 混合模式：部分批量并行，部分串行精细搜索
            parallel_trials = int(exploitation_trials * self.batch_parallel_config.parallel_ratio)
            serial_trials = exploitation_trials - parallel_trials
        else:
            # 全批量并行模式
            parallel_trials = exploitation_trials
            serial_trials = 0

        current_best_value = best_exploration_value

        if verbose:
            mode_desc = f"混合模式 (并行{parallel_trials} + 串行{serial_trials})" if self.batch_parallel_config.hybrid_mode else f"全并行模式 ({parallel_trials})"
            print(f"[利用阶段] 批量并行策略: {mode_desc}")

        # === 第一部分：批量并行优化 ===
        if parallel_trials > 0:
            batch_size = self.batch_parallel_config.batch_size
            n_batches = (parallel_trials + batch_size - 1) // batch_size

            if verbose:
                print(f"[利用阶段] 批量并行: {n_batches} 批次 × {batch_size} 并行 = {n_batches * batch_size} 试验")

            for batch_idx in range(n_batches):
                # 自适应批量大小
                current_batch_size = self._adaptive_batch_size(batch_idx, n_batches)
                current_batch_size = min(current_batch_size, parallel_trials - batch_idx * batch_size)

                if current_batch_size <= 0:
                    break

                if verbose:
                    print(f"\n[批次 {batch_idx+1}/{n_batches}] 批量大小: {current_batch_size}")

                # 1. 多样化批量采样
                batch_data = self._diverse_batch_sampling(study, search_space, current_batch_size)

                # 2. 并行回测
                batch_results = self._parallel_batch_backtest(
                    batch_data, strategy_class, data, objective, verbose
                )

                # 3. 批量更新
                for i, result in enumerate(batch_results):
                    trial_data = batch_data[i]
                    trial = trial_data['trial']
                    params = trial_data['params']
                    is_perturbed = trial_data.get('is_perturbed', False)

                    value = result['value']
                    result_dict = result.get('result_dict')
                    error = result.get('error')

                    # 告诉study结果
                    if error:
                        study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    elif is_perturbed:
                        # 扰动 trial: trial 内部记录的是原始采样参数，
                        # 但实际回测用的是扰动后的 params。
                        # 如果直接 study.tell(trial, value)，TPE 会把 value
                        # 和原始参数关联，导致学到错误映射。
                        # 正确做法：丢弃占位 trial，用 add_trial 注入真实参数。
                        study.tell(trial, state=optuna.trial.TrialState.FAIL)
                        distributions = self._build_distributions(search_space)
                        study.add_trial(
                            optuna.trial.create_trial(
                                params=params,
                                distributions=distributions,
                                values=[value],
                                state=optuna.trial.TrialState.COMPLETE,
                            )
                        )
                    else:
                        study.tell(trial, value)

                    # 更新历史记录
                    trial_idx = len(history_list)
                    history_list.append({
                        "trial": trial_idx,
                        "phase": "exploitation_parallel",
                        "batch": batch_idx,
                        "params": params.copy(),
                        "value": value,
                        "sharpe": result_dict['sharpe_ratio'] if result_dict else 0,
                        "annual_return": result_dict['annual_return'] if result_dict else 0,
                        "max_drawdown": result_dict['max_drawdown'] if result_dict else 0,
                        "error": error
                    })

                    # 检查是否是更优结果
                    if value > current_best_value:
                        current_best_value = value

                        # 记录改进
                        if result_dict:
                            new_results = {
                                "sharpe_ratio": round(result_dict['sharpe_ratio'], 4),
                                "annual_return": round(result_dict['annual_return'], 2),
                                "max_drawdown": round(result_dict['max_drawdown'], 2)
                            }
                            if not best_improvements or best_improvements[-1]["backtest_results"] != new_results:
                                best_improvements.append({
                                    "params": params.copy(),
                                    "backtest_results": new_results
                                })

                        # 输出更优结果
                        if verbose:
                            print(f"\n╔{'═'*78}╗")
                            print(f"║ {'🎯 发现更优参数！'.center(70)} ║")
                            print(f"╠{'═'*78}╣")
                            print(f"║ Trial {trial_idx} (利用阶段-批量并行) {'':50} ║")
                            print(f"║ 目标值: {value:<66.4f} ║")
                            if result_dict:
                                print(f"║ 夏普比率: {result_dict['sharpe_ratio']:<62.4f} ║")
                                print(f"║ 年化收益: {result_dict['annual_return']:<61.2f}% ║")
                                print(f"║ 最大回撤: {result_dict['max_drawdown']:<61.2f}% ║")
                            print(f"╚{'═'*78}╝")

        # === 第二部分：串行精细搜索 ===
        if serial_trials > 0:
            if verbose:
                print(f"\n[利用阶段] 串行精细搜索: {serial_trials} 试验")

            # 创建目标函数用于串行优化
            serial_history = []
            objective_fn = self._create_objective_function(
                strategy_class, data, search_space, objective, serial_history,
                verbose=verbose, phase="exploitation_serial",
                best_improvements=best_improvements,
                initial_best_value=current_best_value
            )

            # 串行优化
            try:
                study.optimize(
                    objective_fn,
                    n_trials=serial_trials,
                    show_progress_bar=verbose and TQDM_AVAILABLE,
                    n_jobs=1  # 串行
                )

                # 合并串行历史
                for record in serial_history:
                    record['trial'] = len(history_list)
                    record['phase'] = 'exploitation_serial'
                    history_list.append(record)

                    # 更新最优值
                    if record['value'] > current_best_value:
                        current_best_value = record['value']

            except Exception as e:
                if verbose:
                    print(f"\n⚠️  串行精细搜索异常: {str(e)}")

        return current_best_value

    def _parallel_batch_backtest(
        self,
        batch_data: List[Dict],
        strategy_class: Type[bt.Strategy],
        data: pd.DataFrame,
        objective: str,
        verbose: bool = True
    ) -> List[Dict]:
        """
        并行批量回测

        Args:
            batch_data: 批量数据，包含trial和params
            strategy_class: 策略类
            data: 回测数据
            objective: 优化目标
            verbose: 是否详细输出

        Returns:
            批量回测结果
        """
        from multiprocessing import Pool, get_context
        from parallel_worker import run_single_backtest

        # 准备任务数据
        tasks = []
        for i, item in enumerate(batch_data):
            params = item['params']

            # 获取路径信息
            strategy_path = getattr(self.backtest_engine, 'strategy_path', None)
            data_path = getattr(self.backtest_engine, 'data_path', None)

            if not strategy_path or not data_path:
                # 回退到线程池模式
                return self._thread_batch_backtest(batch_data, strategy_class, data, objective, verbose)

            # 准备配置字典
            broker_config_dict = None
            if hasattr(self.backtest_engine, 'broker_config') and self.backtest_engine.broker_config:
                bc = self.backtest_engine.broker_config
                broker_config_dict = {
                    'asset_type': bc.asset_type,
                    'contract_code': bc.contract_code,
                    'contract_name': bc.contract_name,
                    'mult': bc.mult,
                    'margin': bc.margin,
                    'commission': bc.commission,
                    'comm_type': bc.comm_type,
                    'initial_cash': bc.initial_cash,
                }

            market_maker_config_dict = None
            if hasattr(self.backtest_engine, 'market_maker_config') and self.backtest_engine.market_maker_config:
                mmc = self.backtest_engine.market_maker_config
                market_maker_config_dict = {
                    'alpha': mmc.alpha,
                    'beta': mmc.beta,
                    'gamma': mmc.gamma,
                    'max_drawdown_threshold': mmc.max_drawdown_threshold,
                    'min_trades': mmc.min_trades,
                }

            task_data = {
                'idx': i,
                'params': params,
                'is_default': False,
                'strategy_path': strategy_path,
                'data_path': data_path,
                'objective': objective,
                'data_frequency': getattr(self.backtest_engine.config, 'data_frequency', None),
                'broker_config_dict': broker_config_dict,
                'market_maker_config_dict': market_maker_config_dict,
            }
            tasks.append(task_data)

        # 并行执行
        n_workers = self.parallel_config.n_workers
        if n_workers == -1:
            import multiprocessing
            n_workers = multiprocessing.cpu_count()

        try:
            ctx = get_context('spawn')
            with ctx.Pool(processes=n_workers) as pool:
                if verbose and TQDM_AVAILABLE:
                    results = []
                    with tqdm(total=len(tasks), desc="批量回测", unit="trial",
                              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                        for result in pool.imap_unordered(run_single_backtest, tasks):
                            results.append(result)
                            pbar.update(1)
                    results.sort(key=lambda x: x['idx'])
                else:
                    results = pool.map(run_single_backtest, tasks)
            return results
        except Exception as e:
            if verbose:
                print(f"\n⚠️  进程池批量回测失败: {e}，回退到线程池")
            return self._thread_batch_backtest(batch_data, strategy_class, data, objective, verbose)

    def _thread_batch_backtest(
        self,
        batch_data: List[Dict],
        strategy_class: Type[bt.Strategy],
        data: pd.DataFrame,
        objective: str,
        verbose: bool = True
    ) -> List[Dict]:
        """
        线程池批量回测（回退方案）

        Args:
            batch_data: 批量数据
            strategy_class: 策略类
            data: 回测数据
            objective: 优化目标
            verbose: 是否详细输出

        Returns:
            批量回测结果
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        accepts_verbose = _strategy_accepts_verbose(strategy_class)

        def eval_single(item_with_idx):
            idx, item = item_with_idx
            params = item['params']
            try:
                run_params = params.copy()
                if accepts_verbose:
                    run_params['verbose'] = False

                result = self.backtest_engine.run_backtest(strategy_class, data, run_params)
                if result is None:
                    return {
                        'idx': idx,
                        'value': float('-inf'),
                        'result_dict': None,
                        'error': None
                    }

                value = self.backtest_engine.evaluate_objective(result, objective)
                return {
                    'idx': idx,
                    'value': value,
                    'result_dict': {
                        'sharpe_ratio': result.sharpe_ratio,
                        'annual_return': result.annual_return,
                        'max_drawdown': result.max_drawdown,
                        'total_return': result.total_return,
                        'trades_count': result.trades_count,
                        'win_rate': result.win_rate,
                    },
                    'error': None
                }
            except Exception as e:
                return {
                    'idx': idx,
                    'value': float('-inf'),
                    'result_dict': None,
                    'error': str(e)
                }

        n_workers = self.parallel_config.n_workers
        if n_workers == -1:
            n_workers = multiprocessing.cpu_count()

        tasks = [(i, item) for i, item in enumerate(batch_data)]

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(eval_single, task) for task in tasks]

            if verbose and TQDM_AVAILABLE:
                results = []
                with tqdm(total=len(futures), desc="批量回测", unit="trial",
                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                    for future in as_completed(futures):
                        results.append(future.result())
                        pbar.update(1)
                results.sort(key=lambda x: x['idx'])
            else:
                results = [f.result() for f in futures]

        return results
    
    def _create_objective_function(
        self,
        strategy_class: Type[bt.Strategy],
        data: pd.DataFrame,
        search_space: Dict[str, SearchSpaceConfig],
        objective: str,
        history_list: List[Dict],
        verbose: bool = True,
        phase: str = "exploitation",
        best_improvements: List[Dict] = None,
        initial_best_value: float = float('-inf')
    ) -> Callable:
        """
        创建Optuna目标函数
        
        Args:
            strategy_class: 策略类
            data: 行情数据
            search_space: 搜索空间
            objective: 优化目标
            history_list: 历史记录列表（用于存储）
            verbose: 是否打印详细信息
            phase: 当前阶段（exploration/exploitation）
            best_improvements: 每次发现更优参数时的记录列表
            initial_best_value: 初始最优值（用于继承上一阶段的最优值）
            
        Returns:
            目标函数
        """
        # 使用闭包变量跟踪当前最优值
        best_value_tracker = {'value': initial_best_value, 'params': None}
        
        def objective_fn(trial: optuna.Trial) -> float:
            try:
                # 建议参数
                params = self._suggest_params(trial, search_space)
                
                # 在优化模式下自动禁用策略日志（仅当策略支持时）
                run_params = params.copy()
                if _strategy_accepts_verbose(strategy_class):
                    run_params['verbose'] = False
                
                # 运行回测
                result = self.backtest_engine.run_backtest(
                    strategy_class,
                    data,
                    run_params
                )
                
                if result is None:
                    return float('-inf')
                
                # 获取目标值
                value = self.backtest_engine.evaluate_objective(result, objective)
            
            except Exception as e:
                # 捕获异常，打印错误信息，但不中断优化
                if verbose:
                    print(f"\n⚠️  [Trial {trial.number}] 回测异常: {str(e)}")
                    print(f"   参数: {params if 'params' in locals() else 'N/A'}")
                    import traceback
                    print(f"   详细信息: {traceback.format_exc()[:200]}...")
                return float('-inf')
            
            # 记录历史
            history_list.append({
                "trial": trial.number,
                "params": params.copy(),
                "value": value,
                "sharpe": result.sharpe_ratio,
                "annual_return": result.annual_return,
                "max_drawdown": result.max_drawdown
            })
            
            # 检查是否找到更优参数
            if value > best_value_tracker['value']:
                best_value_tracker['value'] = value
                best_value_tracker['params'] = params.copy()
                
                # 记录更优参数到改进列表（去重：跳过与上一条回测结果相同的记录）
                if best_improvements is not None:
                    new_results = {
                        "sharpe_ratio": round(result.sharpe_ratio, 4),
                        "annual_return": round(result.annual_return, 2),
                        "max_drawdown": round(result.max_drawdown, 2)
                    }
                    # 仅当回测结果有实质性改善时才记录
                    is_dup = bool(best_improvements) and best_improvements[-1]["backtest_results"] == new_results
                    if not is_dup:
                        best_improvements.append({
                            "params": params.copy(),
                            "backtest_results": new_results
                        })
                
                # 实时输出更优参数
                if verbose:
                    phase_cn = "利用阶段" if phase == "exploitation" else "探索阶段"
                    print(f"\n╔{'═'*78}╗")
                    print(f"║ {'🎯 发现更优参数！'.center(70)} ║")
                    print(f"╠{'═'*78}╣")
                    print(f"║ Trial {trial.number} ({phase_cn}) {'':63} ║")
                    print(f"║ 目标值: {value:<66.4f} ║")
                    print(f"║ 夏普比率: {result.sharpe_ratio:<62.4f} ║")
                    print(f"║ 年化收益: {result.annual_return:<61.2f}% ║")
                    print(f"║ 最大回撤: {result.max_drawdown:<61.2f}% ║")
                    if objective == "market_maker_score":
                        print(f"║ 交易次数: {result.trades_count:<62} ║")
                    print(f"╠{'═'*78}╣")
                    print(f"║ {'参数集:'.ljust(76)} ║")
                    for k, v in params.items():
                        if isinstance(v, float):
                            param_str = f"  • {k}: {v:.4f}"
                        else:
                            param_str = f"  • {k}: {v}"
                        print(f"║ {param_str:<76} ║")
                    print(f"╚{'═'*78}╝")
            
            return value
        
        return objective_fn
    
    def optimize_single_objective(
        self,
        strategy_class: Type[bt.Strategy],
        strategy_name: str,
        data: pd.DataFrame,
        objective: str,
        search_space: Dict[str, SearchSpaceConfig] = None,
        n_trials: int = None,
        verbose: bool = True,
        default_params: Dict[str, Any] = None,
        use_enhanced_sampler: bool = True,
        enable_dynamic_trials: bool = True
    ) -> OptimizationResult:
        """
        单目标优化（支持两阶段优化）
        
        Args:
            strategy_class: 策略类
            strategy_name: 策略名称
            data: 行情数据
            objective: 优化目标
            search_space: 搜索空间（如果为None，自动生成）
            n_trials: 试验次数
            verbose: 是否打印进度
            default_params: 策略的默认参数，将作为第一个采样点
            use_enhanced_sampler: 是否使用增强采样器（正态分布 + 并行探索）
            
        Returns:
            优化结果
        """
        n_params = len(search_space) if search_space else 0
        
        # 动态计算试验次数（如果启用增强采样器）
        if enable_dynamic_trials and use_enhanced_sampler and ENHANCED_SAMPLER_AVAILABLE and n_params > 0:
            config = SamplerConfig()
            calculator = DynamicTrialsCalculator(config)
            recommended_trials, exploration_trials, exploitation_trials = \
                calculator.calculate_trials(n_params, search_space, n_trials)
            
            if verbose:
                print(f"\n[动态试验次数] 参数数量: {n_params}")
                print(f"[动态试验次数] 推荐总试验: {recommended_trials} "
                      f"(探索: {exploration_trials}, 利用: {exploitation_trials})")
                if n_trials and n_trials < recommended_trials:
                    print(f"[动态试验次数] ⚠️ 用户指定 {n_trials} 次，已调整为推荐值")
            
            n_trials = recommended_trials
        else:
            n_trials = n_trials or self.config.n_trials
            exploration_trials = int(n_trials * 0.3)
            exploitation_trials = n_trials - exploration_trials
        
        # 生成搜索空间
        if search_space is None:
            search_space = self.strategy_analyzer.generate_search_space(
                strategy_name,
                use_llm_recommendations=self.use_llm
            )
        
        if verbose:
            print(f"\n╔{'═'*58}╗")
            print(f"║ {'开始优化'.center(54)} ║")
            print(f"╠{'═'*58}╣")
            print(f"║ 策略名称: {strategy_name:<44} ║")
            print(f"║ 优化目标: {objective:<44} ║")
            print(f"║ 试验次数: {n_trials:<44} ║")
            if use_enhanced_sampler and ENHANCED_SAMPLER_AVAILABLE:
                print(f"║ 优化策略: {'两阶段优化 (正态分布探索 + 贝叶斯利用)':<44} ║")
            print(f"╚{'═'*58}╝")
        
        # 初始化历史记录
        history_list = []
        best_improvements = []  # 记录每次发现更优参数的快照
        start_time = datetime.now()
        
        # ============ 阶段1: 正态分布随机探索 ============
        best_exploration_params = None
        best_exploration_value = float('-inf')
        
        if use_enhanced_sampler and ENHANCED_SAMPLER_AVAILABLE and exploration_trials > 0:
            if verbose:
                print(f"\n╔{'═'*58}╗")
                print(f"║ {'阶段1: 正态分布随机探索'.center(54)} ║")
                print(f"╠{'═'*58}╣")
                print(f"║ 试验次数: {exploration_trials:<44} ║")
                # 显示是否启用并行
                if self.parallel_config.enable_parallel and JOBLIB_AVAILABLE:
                    n_workers = self.parallel_config.n_workers
                    if n_workers == -1:
                        n_workers = multiprocessing.cpu_count()
                    print(f"║ 并行模式: {'启用 (' + str(n_workers) + ' 进程)':<44} ║")
                else:
                    print(f"║ 并行模式: {'禁用':<44} ║")
                print(f"╚{'═'*58}╝")

            sampler = NormalDistributionSampler(SamplerConfig(), seed=self.config.seed)

            # 生成正态分布采样的参数组（Trial 0 为策略默认参数）
            samples, has_default_trial0 = sampler.generate_initial_samples(
                search_space=search_space,
                n_samples=exploration_trials,
                default_params=default_params,
                include_default=True
            )

            # 记录 Trial 0（默认参数）的结果
            default_params_value = None

            # ============ 并行执行探索阶段回测 ============
            if self.parallel_config.enable_parallel and JOBLIB_AVAILABLE:
                # 使用进程池并行 - 每个子进程独立加载策略模块
                from multiprocessing import Pool, get_context
                from parallel_worker import run_single_backtest

                # 计算进程数
                n_workers = self.parallel_config.n_workers
                if n_workers == -1:
                    n_workers = multiprocessing.cpu_count()

                # 准备任务数据（可序列化的字典）
                # 获取策略路径和数据路径
                strategy_path = getattr(self.backtest_engine, 'strategy_path', None)
                data_path = getattr(self.backtest_engine, 'data_path', None)

                # 如果没有路径信息，回退到线程池模式
                if not strategy_path or not data_path:
                    if verbose:
                        print(f"\n[并行探索] 缺少路径信息，回退到线程池模式...")
                    use_process_pool = False
                else:
                    use_process_pool = True

                if use_process_pool:
                    # 准备经纪商配置字典
                    broker_config_dict = None
                    if hasattr(self.backtest_engine, 'broker_config') and self.backtest_engine.broker_config:
                        bc = self.backtest_engine.broker_config
                        broker_config_dict = {
                            'asset_type': bc.asset_type,
                            'contract_code': bc.contract_code,
                            'contract_name': bc.contract_name,
                            'mult': bc.mult,
                            'margin': bc.margin,
                            'commission': bc.commission,
                            'comm_type': bc.comm_type,
                            'initial_cash': bc.initial_cash,
                        }

                    # 准备做市商配置字典
                    market_maker_config_dict = None
                    if hasattr(self.backtest_engine, 'market_maker_config') and self.backtest_engine.market_maker_config:
                        mmc = self.backtest_engine.market_maker_config
                        market_maker_config_dict = {
                            'alpha': mmc.alpha,
                            'beta': mmc.beta,
                            'gamma': mmc.gamma,
                            'max_drawdown_threshold': mmc.max_drawdown_threshold,
                            'min_trades': mmc.min_trades,
                        }

                    # 构建任务列表
                    tasks = []
                    for i, params in enumerate(samples):
                        task_data = {
                            'idx': i,
                            'params': params,
                            'is_default': (i == 0 and has_default_trial0),
                            'strategy_path': strategy_path,
                            'data_path': data_path,
                            'objective': objective,
                            'data_frequency': getattr(self.backtest_engine.config, 'data_frequency', None),
                            'broker_config_dict': broker_config_dict,
                            'market_maker_config_dict': market_maker_config_dict,
                        }
                        tasks.append(task_data)

                    if verbose:
                        print(f"\n[并行探索] 启动 {len(tasks)} 个回测任务 ({n_workers} 进程)...")

                    # 使用进程池并行执行
                    try:
                        # 使用 spawn 方式创建进程，避免 fork 导致的问题
                        ctx = get_context('spawn')
                        with ctx.Pool(processes=n_workers) as pool:
                            if verbose and TQDM_AVAILABLE:
                                # 使用 imap_unordered 配合 tqdm 显示进度
                                parallel_results = []
                                with tqdm(total=len(tasks), desc="探索阶段", unit="trial",
                                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                                    for result in pool.imap_unordered(run_single_backtest, tasks):
                                        parallel_results.append(result)
                                        pbar.update(1)
                                # 按 idx 排序结果
                                parallel_results.sort(key=lambda x: x['idx'])
                            else:
                                parallel_results = pool.map(run_single_backtest, tasks)
                    except Exception as e:
                        if verbose:
                            print(f"\n⚠️  进程池执行失败: {e}")
                            print(f"    回退到线程池模式...")
                        use_process_pool = False

                # 如果进程池不可用，使用线程池
                if not use_process_pool:
                    accepts_verbose = _strategy_accepts_verbose(strategy_class)

                    def eval_single_backtest(task_tuple):
                        """单次回测评估（线程安全）"""
                        idx, params, is_default = task_tuple
                        try:
                            run_params = params.copy()
                            if accepts_verbose:
                                run_params['verbose'] = False

                            result = self.backtest_engine.run_backtest(strategy_class, data, run_params)
                            if result is None:
                                return {
                                    'idx': idx,
                                    'params': params,
                                    'is_default': is_default,
                                    'value': float('-inf'),
                                    'result_dict': None,
                                    'error': None
                                }
                            value = self.backtest_engine.evaluate_objective(result, objective)
                            return {
                                'idx': idx,
                                'params': params,
                                'is_default': is_default,
                                'value': value,
                                'result_dict': {
                                    'sharpe_ratio': result.sharpe_ratio,
                                    'annual_return': result.annual_return,
                                    'max_drawdown': result.max_drawdown,
                                    'total_return': result.total_return,
                                    'trades_count': result.trades_count,
                                    'win_rate': result.win_rate,
                                },
                                'error': None
                            }
                        except Exception as e:
                            return {
                                'idx': idx,
                                'params': params,
                                'is_default': is_default,
                                'value': float('-inf'),
                                'result_dict': None,
                                'error': str(e)
                            }

                    thread_tasks = [
                        (i, params, (i == 0 and has_default_trial0))
                        for i, params in enumerate(samples)
                    ]

                    if verbose:
                        print(f"\n[并行探索] 启动 {len(thread_tasks)} 个回测任务 ({n_workers} 线程)...")

                    with ThreadPoolExecutor(max_workers=n_workers) as executor:
                        futures = [executor.submit(eval_single_backtest, task) for task in thread_tasks]

                        if verbose and TQDM_AVAILABLE:
                            from concurrent.futures import as_completed
                            parallel_results = []
                            with tqdm(total=len(futures), desc="探索阶段", unit="trial",
                                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                                for future in as_completed(futures):
                                    parallel_results.append(future.result())
                                    pbar.update(1)
                            parallel_results.sort(key=lambda x: x['idx'])
                        else:
                            parallel_results = [f.result() for f in futures]

                # 处理并行结果（统一格式：result_dict 而非 result）
                for res in parallel_results:
                    idx = res['idx']
                    params = res['params']
                    is_default_trial = res['is_default']
                    value = res['value']
                    result_dict = res.get('result_dict')
                    error = res.get('error')

                    # 记录错误
                    if error:
                        if verbose:
                            print(f"\n⚠️  [探索阶段 Trial {idx}] 回测异常: {error[:200]}")
                        history_list.append({
                            "trial": idx,
                            "phase": "exploration",
                            "is_default": is_default_trial,
                            "params": params.copy(),
                            "value": float('-inf'),
                            "sharpe": 0,
                            "annual_return": 0,
                            "max_drawdown": 0,
                            "error": error
                        })
                        continue

                    # 记录 Trial 0 的结果
                    if is_default_trial:
                        default_params_value = value
                        if verbose:
                            print(f"\n╔{'═'*78}╗")
                            print(f"║ {'Trial 0: 策略默认参数回测'.center(74)} ║")
                            print(f"╠{'═'*78}╣")
                            print(f"║ 目标值 ({objective}): {value:<57.4f} ║")
                            if result_dict:
                                print(f"║ 夏普比率: {result_dict['sharpe_ratio']:<62.4f} ║")
                                print(f"║ 年化收益: {result_dict['annual_return']:<61.2f}% ║")
                                print(f"║ 最大回撤: {result_dict['max_drawdown']:<61.2f}% ║")
                            print(f"╚{'═'*78}╝")

                    # 记录历史
                    history_list.append({
                        "trial": idx,
                        "phase": "exploration",
                        "is_default": is_default_trial,
                        "params": params.copy(),
                        "value": value,
                        "sharpe": result_dict['sharpe_ratio'] if result_dict else 0,
                        "annual_return": result_dict['annual_return'] if result_dict else 0,
                        "max_drawdown": result_dict['max_drawdown'] if result_dict else 0
                    })

                    # 更新最优参数
                    if value > best_exploration_value:
                        best_exploration_value = value
                        best_exploration_params = params.copy()

                        # 记录更优参数到改进列表（去重）
                        new_results = {
                            "sharpe_ratio": round(result_dict['sharpe_ratio'], 4) if result_dict else 0,
                            "annual_return": round(result_dict['annual_return'], 2) if result_dict else 0,
                            "max_drawdown": round(result_dict['max_drawdown'], 2) if result_dict else 0
                        }
                        if not best_improvements or best_improvements[-1]["backtest_results"] != new_results:
                            best_improvements.append({
                                "params": params.copy(),
                                "backtest_results": new_results
                            })

                        # 发现更优参数时输出
                        if verbose:
                            print(f"\n╔{'═'*78}╗")
                            print(f"║ {'🎯 发现更优参数！'.center(70)} ║")
                            print(f"╠{'═'*78}╣")
                            print(f"║ Trial {idx} (探索阶段-并行) {'':57} ║")
                            print(f"║ 目标值: {value:<66.4f} ║")
                            if result_dict:
                                print(f"║ 夏普比率: {result_dict['sharpe_ratio']:<62.4f} ║")
                                print(f"║ 年化收益: {result_dict['annual_return']:<61.2f}% ║")
                                print(f"║ 最大回撤: {result_dict['max_drawdown']:<61.2f}% ║")
                                if objective == "market_maker_score" and 'trades_count' in result_dict:
                                    print(f"║ 交易次数: {result_dict['trades_count']:<62} ║")
                            print(f"╠{'═'*78}╣")
                            print(f"║ {'参数集:'.ljust(76)} ║")
                            for k, v in params.items():
                                if isinstance(v, float):
                                    param_str = f"  • {k}: {v:.4f}"
                                else:
                                    param_str = f"  • {k}: {v}"
                                print(f"║ {param_str:<76} ║")
                            print(f"╚{'═'*78}╝")

                if verbose:
                    print(f"\n[并行探索] 完成 {len(parallel_results)} 个回测任务")

            else:
                # ============ 串行执行探索阶段回测（原有逻辑）============
                # 评估每组参数
                for i, params in enumerate(samples):
                    try:
                        # 标记 Trial 0（策略默认参数）
                        is_default_trial = (i == 0 and has_default_trial0)

                        if verbose and is_default_trial:
                            print(f"\n[Trial 0] 策略默认参数回测:")
                            for k, v in params.items():
                                if isinstance(v, float):
                                    print(f"   • {k}: {v:.4f}")
                                else:
                                    print(f"   • {k}: {v}")

                        # 在优化模式下自动禁用策略日志（仅当策略支持时）
                        run_params = params.copy()
                        if _strategy_accepts_verbose(strategy_class):
                            run_params['verbose'] = False

                        result = self.backtest_engine.run_backtest(
                            strategy_class, data, run_params
                        )
                        if result is None:
                            value = float('-inf')
                        else:
                            value = self.backtest_engine.evaluate_objective(result, objective)

                        # 记录 Trial 0 的结果
                        if is_default_trial:
                            default_params_value = value
                            if verbose:
                                print(f"\n╔{'═'*78}╗")
                                print(f"║ {'Trial 0: 策略默认参数回测'.center(74)} ║")
                                print(f"╠{'═'*78}╣")
                                print(f"║ 目标值 ({objective}): {value:<57.4f} ║")
                                if result:
                                    print(f"║ 夏普比率: {result.sharpe_ratio:<62.4f} ║")
                                    print(f"║ 年化收益: {result.annual_return:<61.2f}% ║")
                                    print(f"║ 最大回撤: {result.max_drawdown:<61.2f}% ║")
                                print(f"╚{'═'*78}╝")

                        history_list.append({
                            "trial": i,
                            "phase": "exploration",
                            "is_default": is_default_trial,
                            "params": params.copy(),
                            "value": value,
                            "sharpe": result.sharpe_ratio if result else 0,
                            "annual_return": result.annual_return if result else 0,
                            "max_drawdown": result.max_drawdown if result else 0
                        })

                        if value > best_exploration_value:
                            best_exploration_value = value
                            best_exploration_params = params.copy()

                            # 记录更优参数到改进列表（去重）
                            new_results = {
                                "sharpe_ratio": round(result.sharpe_ratio, 4) if result else 0,
                                "annual_return": round(result.annual_return, 2) if result else 0,
                                "max_drawdown": round(result.max_drawdown, 2) if result else 0
                            }
                            if not best_improvements or best_improvements[-1]["backtest_results"] != new_results:
                                best_improvements.append({
                                    "params": params.copy(),
                                    "backtest_results": new_results
                                })

                            # 发现更优参数时立即输出
                            if verbose:
                                print(f"\n╔{'═'*78}╗")
                                print(f"║ {'🎯 发现更优参数！'.center(70)} ║")
                                print(f"╠{'═'*78}╣")
                                print(f"║ Trial {i} (探索阶段) {'':63} ║")
                                print(f"║ 目标值: {value:<66.4f} ║")
                                if result:
                                    print(f"║ 夏普比率: {result.sharpe_ratio:<62.4f} ║")
                                    print(f"║ 年化收益: {result.annual_return:<61.2f}% ║")
                                    print(f"║ 最大回撤: {result.max_drawdown:<61.2f}% ║")
                                    if objective == "market_maker_score":
                                        print(f"║ 交易次数: {result.trades_count:<62} ║")
                                print(f"╠{'═'*78}╣")
                                print(f"║ {'参数集:'.ljust(76)} ║")
                                for k, v in params.items():
                                    if isinstance(v, float):
                                        param_str = f"  • {k}: {v:.4f}"
                                    else:
                                        param_str = f"  • {k}: {v}"
                                    print(f"║ {param_str:<76} ║")
                                print(f"╚{'═'*78}╝")

                        if verbose and (i + 1) % 10 == 0:
                            progress_pct = (i + 1) / exploration_trials * 100
                            print(f"[探索阶段] 进度: {i+1}/{exploration_trials} ({progress_pct:.1f}%) | "
                                  f"当前最优: {best_exploration_value:.4f}")

                    except Exception as e:
                        # 探索阶段异常处理：打印详细错误但继续执行
                        if verbose:
                            print(f"\n⚠️  [探索阶段 Trial {i}] 回测异常: {str(e)}")
                            print(f"   参数: {params}")
                            import traceback
                            traceback.print_exc()
                        # 记录失败的试验
                        history_list.append({
                            "trial": i,
                            "phase": "exploration",
                            "is_default": is_default_trial if 'is_default_trial' in locals() else False,
                            "params": params.copy(),
                            "value": float('-inf'),
                            "sharpe": 0,
                            "annual_return": 0,
                            "max_drawdown": 0,
                            "error": str(e)
                        })
                        continue
            if verbose:
                print(f"\n╔{'═'*78}╗")
                print(f"║ {'探索阶段完成'.center(74)} ║")
                print(f"╠{'═'*78}╣")
                print(f"║ 最佳目标值: {best_exploration_value:<61.4f} ║")
                if default_params_value is not None:
                    improvement = ((best_exploration_value - default_params_value) / abs(default_params_value) * 100) if default_params_value != 0 else 0
                    print(f"║ 相比默认参数: {improvement:>+60.2f}% ║")
                print(f"╚{'═'*78}╝")
        
        # ============ 阶段2: 贝叶斯智能采样（利用阶段）============
        if verbose:
            print(f"\n╔{'═'*58}╗")
            print(f"║ {'阶段2: 贝叶斯智能采样'.center(54)} ║")
            print(f"╠{'═'*58}╣")
            print(f"║ 试验次数: {exploitation_trials:<44} ║")
            print(f"╚{'═'*58}╝")
        
        # 创建Study
        direction = "maximize"  # 回撤已在evaluate_objective中取负
        
        # 批量并行模式下开启 constant_liar，使 batch 内连续 ask() 采样更分散
        use_constant_liar = self.batch_parallel_config.enable_batch_parallel and JOBLIB_AVAILABLE
        study = optuna.create_study(
            direction=direction,
            sampler=self._create_sampler(constant_liar=use_constant_liar),
            pruner=self._create_pruner()
        )
        
        # 将探索阶段的最佳参数作为初始点
        if best_exploration_params:
            enqueue_params = {k: v for k, v in best_exploration_params.items() if k in search_space}
            if enqueue_params:
                study.enqueue_trial(enqueue_params)
                if verbose:
                    print(f"[利用阶段] 已将探索阶段最佳参数加入采样队列")
        elif default_params:
            # 如果没有探索阶段，使用默认参数
            enqueue_params = {k: v for k, v in default_params.items() if k in search_space}
            if enqueue_params:
                study.enqueue_trial(enqueue_params)
                if verbose:
                    print(f"[利用阶段] 已将默认参数加入采样队列")
        
        # 创建目标函数
        exploitation_history = []
        objective_fn = self._create_objective_function(
            strategy_class, data, search_space, objective, exploitation_history,
            verbose=self.verbose, phase="exploitation",
            best_improvements=best_improvements,
            initial_best_value=best_exploration_value
        )
        
        # 运行贝叶斯优化（支持批量并行）
        try:
            if self.batch_parallel_config.enable_batch_parallel and JOBLIB_AVAILABLE:
                # 批量并行优化
                if verbose:
                    print(f"[利用阶段] 使用批量并行优化")

                final_best_value = self._batch_parallel_exploitation(
                    study=study,
                    strategy_class=strategy_class,
                    data=data,
                    search_space=search_space,
                    objective=objective,
                    exploitation_trials=exploitation_trials,
                    history_list=exploitation_history,
                    best_improvements=best_improvements,
                    best_exploration_value=best_exploration_value,
                    verbose=verbose
                )
            else:
                # 传统串行优化
                if verbose:
                    print(f"[利用阶段] 使用传统串行优化")

                study.optimize(
                    objective_fn,
                    n_trials=exploitation_trials,
                    show_progress_bar=verbose,
                    n_jobs=self.config.n_jobs
                )
                final_best_value = study.best_value if study.best_value != float('-inf') else best_exploration_value

        except Exception as e:
            # 捕获优化过程中的严重异常
            if verbose:
                print(f"\n❌ [利用阶段] 优化过程异常: {str(e)}")
                import traceback
                traceback.print_exc()
                print(f"\n⚠️  优化将使用已完成的 {len(exploitation_history)} 次试验结果继续...")
        
        # 合并历史记录（批量并行模式已经处理过了）
        if not self.batch_parallel_config.enable_batch_parallel or not JOBLIB_AVAILABLE:
            for i, record in enumerate(exploitation_history):
                record['trial'] = len(history_list) + i
                record['phase'] = 'exploitation'
                history_list.append(record)
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        # 获取最佳结果（优先使用批量并行的结果）
        try:
            if self.batch_parallel_config.enable_batch_parallel and JOBLIB_AVAILABLE:
                best_value = final_best_value
                # 从study或历史中找到对应的最佳参数
                best_params = None
                max_value = float('-inf')
                for record in history_list:
                    if 'exploitation' in record.get('phase', '') and record.get('value', float('-inf')) > max_value:
                        max_value = record['value']
                        best_params = record['params']

                if best_params is None:
                    best_params = study.best_params if hasattr(study, 'best_params') else {}
                    best_value = study.best_value if hasattr(study, 'best_value') else final_best_value
            else:
                best_params = study.best_params
                best_value = study.best_value
        except Exception as e:
            # 如果无法从study获取最佳结果，使用探索阶段的最佳结果
            if verbose:
                print(f"\n⚠️  无法从利用阶段获取最佳结果: {str(e)}")
                print(f"   使用探索阶段的最佳结果...")
            best_params = best_exploration_params if best_exploration_params else {}
            best_value = best_exploration_value
        
        if best_exploration_value > best_value:
            best_params = best_exploration_params
            best_value = best_exploration_value
            if verbose:
                print(f"\n[结果] 探索阶段找到的参数更优!")
        
        # 重新运行最佳参数获取完整回测结果
        try:
            final_params = best_params.copy()
            if _strategy_accepts_verbose(strategy_class):
                final_params['verbose'] = False
            best_result = self.backtest_engine.run_backtest(
                strategy_class, data, final_params
            )
        except Exception as e:
            if verbose:
                print(f"\n⚠️  重新运行最佳参数时异常: {str(e)}")
                print(f"   将使用历史记录中的结果...")
            # 尝试从历史记录中获取最佳结果
            best_result = None
            for record in history_list:
                if record.get('params') == best_params and record.get('value') == best_value:
                    best_result = record.get('result')
                    break
        
        if verbose:
            print(f"\n╔{'═'*78}╗")
            print(f"║ {'✅ 优化完成！'.center(70)} ║")
            print(f"╠{'═'*78}╣")
            print(f"║ 最佳目标值 ({objective}): {best_value:<53.4f} ║")
            if best_result:
                print(f"║ 夏普比率: {best_result.sharpe_ratio:<62.4f} ║")
                print(f"║ 年化收益: {best_result.annual_return:<61.2f}% ║")
                print(f"║ 最大回撤: {best_result.max_drawdown:<61.2f}% ║")
                print(f"║ 总交易次数: {best_result.trades_count:<60} ║")
            print(f"║ 总耗时: {optimization_time:<65.2f}s ║")
            print(f"╠{'═'*78}╣")
            print(f"║ {'最佳参数集:'.ljust(76)} ║")
            for k, v in best_params.items():
                if isinstance(v, float):
                    param_str = f"  • {k}: {v:.4f}"
                else:
                    param_str = f"  • {k}: {v}"
                print(f"║ {param_str:<76} ║")
            print(f"╚{'═'*78}╝")
        
        # 保存历史
        key = f"{strategy_name}_{objective}"
        self.optimization_history[key] = history_list
        
        return OptimizationResult(
            objective=objective,
            best_params=best_params,
            best_value=best_value if objective != "max_drawdown" else -best_value,
            backtest_result=best_result,
            n_trials=n_trials,
            optimization_time=optimization_time,
            best_improvements=best_improvements
        )
    
    def optimize_with_llm_feedback(
        self,
        strategy_class: Type[bt.Strategy],
        strategy_name: str,
        data: pd.DataFrame,
        objective: str,
        n_rounds: int = None,
        trials_per_round: int = None,
        verbose: bool = True
    ) -> OptimizationResult:
        """
        带LLM反馈的多轮优化
        
        Args:
            strategy_class: 策略类
            strategy_name: 策略名称
            data: 行情数据
            objective: 优化目标
            n_rounds: 优化轮数
            trials_per_round: 每轮试验次数
            verbose: 是否打印进度
            
        Returns:
            最终优化结果
        """
        n_rounds = n_rounds or self.config.n_rounds
        trials_per_round = trials_per_round or (self.config.n_trials // n_rounds)
        
        # 初始搜索空间
        current_space = self.strategy_analyzer.generate_search_space(
            strategy_name,
            use_llm_recommendations=self.use_llm
        )
        
        if verbose:
            print(f"\n{'#'*60}")
            print(f"开始LLM引导的多轮优化")
            print(f"策略: {strategy_name}")
            print(f"目标: {objective}")
            print(f"轮数: {n_rounds}, 每轮试验: {trials_per_round}")
            print(f"{'#'*60}")
            
            self.strategy_analyzer.print_search_space(current_space)
        
        all_history = []
        best_result = None
        best_value = float('-inf')
        best_params = None
        
        for round_idx in range(n_rounds):
            if verbose:
                print(f"\n{'='*40}")
                print(f"第 {round_idx + 1}/{n_rounds} 轮优化")
                print(f"{'='*40}")
            
            # 运行这一轮优化
            result = self.optimize_single_objective(
                strategy_class,
                strategy_name,
                data,
                objective,
                search_space=current_space,
                n_trials=trials_per_round,
                verbose=verbose
            )
            
            # 获取这一轮的历史
            key = f"{strategy_name}_{objective}"
            round_history = self.optimization_history.get(key, [])
            all_history.extend(round_history)
            
            # 更新最佳结果
            current_value = result.best_value if objective != "max_drawdown" else -result.best_value
            compare_value = current_value if objective != "max_drawdown" else -current_value
            compare_best = best_value if objective != "max_drawdown" else -best_value
            
            if objective == "max_drawdown":
                # 回撤越小越好
                if best_result is None or result.best_value < best_value:
                    best_result = result
                    best_value = result.best_value
                    best_params = result.best_params.copy()
            else:
                if compare_value > compare_best:
                    best_result = result
                    best_value = current_value
                    best_params = result.best_params.copy()
            
            # 使用LLM调整搜索空间（除了最后一轮）
            if round_idx < n_rounds - 1 and self.use_llm:
                if verbose:
                    print("\n[LLM] 分析优化历史...")
                
                current_space = self.strategy_analyzer.adjust_search_space(
                    current_space,
                    all_history,
                    objective
                )
                
                if verbose:
                    self.strategy_analyzer.print_search_space(current_space)
        
        # 返回最终结果
        if best_result:
            best_result.best_params = best_params
            best_result.n_trials = len(all_history)
        
        return best_result
    
    def optimize_all_objectives(
        self,
        strategy_class: Type[bt.Strategy],
        strategy_name: str,
        data: pd.DataFrame,
        use_llm_feedback: bool = True,
        verbose: bool = True
    ) -> Dict[str, OptimizationResult]:
        """
        针对所有目标进行优化
        
        Args:
            strategy_class: 策略类
            strategy_name: 策略名称
            data: 行情数据
            use_llm_feedback: 是否使用LLM反馈
            verbose: 是否打印进度
            
        Returns:
            各目标的优化结果
        """
        results = {}
        objectives = ["sharpe_ratio", "annual_return", "max_drawdown"]
        
        for objective in objectives:
            if verbose:
                print(f"\n{'*'*60}")
                print(f"优化目标: {OPTIMIZATION_OBJECTIVES[objective].description}")
                print(f"{'*'*60}")
            
            if use_llm_feedback and self.use_llm:
                result = self.optimize_with_llm_feedback(
                    strategy_class,
                    strategy_name,
                    data,
                    objective,
                    verbose=verbose
                )
            else:
                result = self.optimize_single_objective(
                    strategy_class,
                    strategy_name,
                    data,
                    objective,
                    verbose=verbose
                )
            
            results[objective] = result
        
        # 保存结果
        self.all_results[strategy_name] = results
        self._save_results(strategy_name, results)
        
        return results
    
    def _save_results(self, strategy_name: str, results: Dict[str, OptimizationResult]):
        """保存优化结果到文件"""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{strategy_name}_optimization_{timestamp}.json"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        save_data = {
            "strategy": strategy_name,
            "timestamp": timestamp,
            "results": {}
        }
        
        for objective, result in results.items():
            save_data["results"][objective] = {
                "best_params": result.best_params,
                "best_value": result.best_value,
                "n_trials": result.n_trials,
                "optimization_time": result.optimization_time,
                "backtest": {
                    "total_return": result.backtest_result.total_return,
                    "annual_return": result.backtest_result.annual_return,
                    "max_drawdown": result.backtest_result.max_drawdown,
                    "sharpe_ratio": result.backtest_result.sharpe_ratio,
                    "trades_count": result.backtest_result.trades_count,
                    "win_rate": result.backtest_result.win_rate
                } if result.backtest_result else None
            }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存到: {filepath}")
    
    def get_optimization_summary(
        self,
        results: Dict[str, OptimizationResult]
    ) -> pd.DataFrame:
        """
        生成优化结果摘要表格
        
        Args:
            results: 优化结果字典
            
        Returns:
            DataFrame格式的摘要
        """
        summary_data = []
        
        for objective, result in results.items():
            row = {
                "优化目标": objective,
                "最佳值": result.best_value,
                "试验次数": result.n_trials,
                "优化时间(秒)": result.optimization_time
            }
            
            # 添加参数
            for param, value in result.best_params.items():
                row[f"参数_{param}"] = value
            
            # 添加回测结果
            if result.backtest_result:
                row["总收益率(%)"] = result.backtest_result.total_return
                row["年化收益率(%)"] = result.backtest_result.annual_return
                row["最大回撤(%)"] = result.backtest_result.max_drawdown
                row["夏普比率"] = result.backtest_result.sharpe_ratio
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)


if __name__ == "__main__":
    # 测试代码
    from src.Aberration import AberrationStrategy
    
    optimizer = BayesianOptimizer(use_llm=False)
    engine = BacktestEngine()
    
    # 加载数据
    data = engine.load_data("BTC")
    
    if data is not None:
        print(f"数据加载完成: {len(data)} 条记录")
        
        # 简单测试
        result = optimizer.optimize_single_objective(
            AberrationStrategy,
            "AberrationStrategy",
            data,
            "sharpe_ratio",
            n_trials=20,
            verbose=True
        )
        
        print(f"\n优化完成!")
        print(f"最佳夏普比率: {result.best_value:.4f}")
        print(f"最佳参数: {result.best_params}")
