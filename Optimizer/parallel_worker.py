# -*- coding: utf-8 -*-
"""
并行回测工作进程模块

用于在子进程中独立执行回测任务，避免 pickle 序列化问题。
每个子进程会独立加载策略模块和数据。
"""

import os
import sys
import importlib.util
import inspect
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtrader as bt
from backtest_engine import BacktestEngine
from futures_config import BrokerConfig


def _load_strategy_from_path(strategy_path: str):
    """
    从文件路径动态加载策略类

    Args:
        strategy_path: 策略文件的绝对路径

    Returns:
        (strategy_class, strategy_module, custom_data_class, custom_commission_class)
    """
    from pathlib import Path

    if not os.path.exists(strategy_path):
        raise FileNotFoundError(f"策略文件不存在: {strategy_path}")

    # 动态导入模块
    module_name = f"strategy_worker_{Path(strategy_path).stem}_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(module_name, strategy_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # 查找策略类
    strategy_classes = []
    for name, obj in inspect.getmembers(module):
        if (inspect.isclass(obj) and hasattr(obj, 'params')
            and obj.__module__ == module_name
            and issubclass(obj, bt.Strategy)):
            strategy_classes.append(obj)

    if not strategy_classes:
        raise ValueError(f"未在策略文件中找到有效的策略类: {strategy_path}")

    strategy_class = strategy_classes[0]

    # 查找自定义数据类
    custom_data_class = None
    for name, obj in inspect.getmembers(module):
        if (inspect.isclass(obj) and
            obj.__module__ == module_name and
            issubclass(obj, bt.feeds.PandasData) and
            obj is not bt.feeds.PandasData):
            custom_data_class = obj
            break

    # 查找自定义手续费类
    custom_commission_class = None
    for name, obj in inspect.getmembers(module):
        if (inspect.isclass(obj) and
            obj.__module__ == module_name and
            issubclass(obj, bt.CommInfoBase) and
            obj is not bt.CommInfoBase):
            custom_commission_class = obj
            break

    return strategy_class, module, custom_data_class, custom_commission_class


def _strategy_accepts_verbose(strategy_class) -> bool:
    """检查策略类是否接受 verbose 参数"""
    try:
        sig = inspect.signature(strategy_class.__init__)
        return 'verbose' in sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
    except Exception:
        return False


def run_single_backtest(task_data: dict) -> dict:
    """
    在子进程中执行单次回测

    Args:
        task_data: 包含回测所需信息的字典
            - idx: 任务索引
            - params: 策略参数
            - is_default: 是否是默认参数
            - strategy_path: 策略文件路径
            - data_path: 数据文件路径（或路径列表）
            - objective: 优化目标
            - data_frequency: 数据频率
            - broker_config_dict: 经纪商配置字典（可选）
            - market_maker_config_dict: 做市商配置字典（可选）

    Returns:
        回测结果字典
    """
    import io
    import contextlib

    idx = task_data['idx']
    params = task_data['params']
    is_default = task_data['is_default']
    strategy_path = task_data['strategy_path']
    data_path = task_data['data_path']
    objective = task_data['objective']
    data_frequency = task_data.get('data_frequency')
    broker_config_dict = task_data.get('broker_config_dict')
    market_maker_config_dict = task_data.get('market_maker_config_dict')

    # 抑制子进程中的所有 print 输出
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            # 1. 加载策略
            strategy_class, strategy_module, custom_data_class, custom_commission_class = \
                _load_strategy_from_path(strategy_path)

            # 2. 加载数据
            if isinstance(data_path, (list, tuple)):
                data = [pd.read_csv(p) for p in data_path]
                for df in data:
                    if 'datetime' not in df.columns:
                        for col in ['date', 'time_key', 'time']:
                            if col in df.columns:
                                df.rename(columns={col: 'datetime'}, inplace=True)
                                break
                    df['datetime'] = pd.to_datetime(df['datetime'])
            else:
                data = pd.read_csv(data_path)
                if 'datetime' not in data.columns:
                    for col in ['date', 'time_key', 'time']:
                        if col in data.columns:
                            data.rename(columns={col: 'datetime'}, inplace=True)
                            break
                data['datetime'] = pd.to_datetime(data['datetime'])

            # 3. 重建经纪商配置
            broker_config = None
            if broker_config_dict:
                broker_config = BrokerConfig(
                    asset_type=broker_config_dict.get('asset_type', 'stock'),
                    mult=broker_config_dict.get('mult', 1),
                    margin=broker_config_dict.get('margin', 1.0),
                    comm_type=broker_config_dict.get('comm_type', 'PERC'),
                    commission=broker_config_dict.get('commission', 0.001),
                    initial_cash=broker_config_dict.get('initial_cash', 100000.0),
                    contract_code=broker_config_dict.get('contract_code', ''),
                    contract_name=broker_config_dict.get('contract_name', '')
                )

            # 4. 重建做市商配置
            market_maker_config = None
            if market_maker_config_dict:
                from config import MarketMakerConfig
                market_maker_config = MarketMakerConfig(**market_maker_config_dict)

            # 5. 创建回测引擎
            initial_cash = broker_config.initial_cash if broker_config else 100000.0
            commission = broker_config.commission if (broker_config and not broker_config.is_futures) else 0.001

            engine = BacktestEngine(
                data=data,
                strategy_class=strategy_class,
                initial_cash=initial_cash,
                commission=commission,
                data_frequency=data_frequency,
                custom_data_class=custom_data_class,
                custom_commission_class=custom_commission_class,
                strategy_module=strategy_module,
                broker_config=broker_config,
                market_maker_config=market_maker_config
            )

            # 6. 准备参数并验证
            run_params = params.copy()
            if _strategy_accepts_verbose(strategy_class):
                run_params['verbose'] = False

            # 添加参数安全检查
            if 'q_pressure' in run_params:
                run_params['q_pressure'] = max(0.05, min(0.95, run_params['q_pressure']))
            if 'q_support' in run_params:
                run_params['q_support'] = max(0.05, min(0.95, run_params['q_support']))
            if 'lookback' in run_params:
                run_params['lookback'] = max(20, min(200, int(run_params['lookback'])))
            if 'window' in run_params:
                run_params['window'] = max(2, min(20, int(run_params['window'])))

            # 7. 执行回测
            result = engine.run_backtest(strategy_class, data, run_params)

            if result is None:
                return {
                    'idx': idx,
                    'params': params,
                    'is_default': is_default,
                    'value': float('-inf'),
                    'result_dict': None,
                    'error': None
                }

            # 8. 计算目标值
            value = engine.evaluate_objective(result, objective)

            # 9. 将结果转换为可序列化的字典
            result_dict = {
                'sharpe_ratio': result.sharpe_ratio,
                'annual_return': result.annual_return,
                'max_drawdown': result.max_drawdown,
                'total_return': result.total_return,
                'trades_count': result.trades_count,
                'win_rate': result.win_rate,
                'sortino_ratio': getattr(result, 'sortino_ratio', 0),
                'calmar_ratio': getattr(result, 'calmar_ratio', 0),
            }

            return {
                'idx': idx,
                'params': params,
                'is_default': is_default,
                'value': value,
                'result_dict': result_dict,
                'error': None
            }

        except Exception as e:
            import traceback
            return {
                'idx': idx,
                'params': params,
                'is_default': is_default,
                'value': float('-inf'),
                'result_dict': None,
                'error': f"{str(e)}\n{traceback.format_exc()[:500]}"
            }


def run_batch_backtest(tasks: list) -> list:
    """
    批量执行回测（用于单进程内顺序执行多个任务）

    Args:
        tasks: 任务数据列表

    Returns:
        结果列表
    """
    return [run_single_backtest(task) for task in tasks]


# 用于 multiprocessing.Pool 的包装函数
def _worker_wrapper(task_data: dict) -> dict:
    """进程池工作函数包装"""
    return run_single_backtest(task_data)
