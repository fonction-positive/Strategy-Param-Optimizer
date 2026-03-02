# -*- coding: utf-8 -*-
"""
SPP (System Parameter Permutation) 鲁棒性分析 CLI 入口 — v3.0

以最优参数为中心进行蒙特卡洛采样，评估参数鲁棒性。
"""

import sys
import os
import json
import argparse
import importlib.util
import inspect
from pathlib import Path
from datetime import datetime

import pandas as pd
import backtrader as bt

from backtest_engine import BacktestEngine
from config import StrategyParam
from spp_analyzer import SPPConfig, SPPAnalyzer

try:
    import futures_config as fc
except ImportError:
    fc = None


def load_strategy(strategy_path: str):
    """动态加载策略类，同时提取自定义数据类、手续费类和 trade_log 检测"""
    if not os.path.exists(strategy_path):
        raise FileNotFoundError(f"策略文件不存在: {strategy_path}")
    module_name = f"strategy_module_{Path(strategy_path).stem}"
    spec = importlib.util.spec_from_file_location(module_name, strategy_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    strategy_classes = []
    for name, obj in inspect.getmembers(module):
        if (inspect.isclass(obj) and hasattr(obj, 'params')
                and obj.__module__ == module_name
                and issubclass(obj, bt.Strategy)):
            strategy_classes.append(obj)
    if not strategy_classes:
        raise ValueError(f"未在策略文件中找到有效的策略类: {strategy_path}")

    strategy_class = strategy_classes[0]

    # 查找自定义数据类（继承自 bt.feeds.PandasData）
    custom_data_class = None
    for name, obj in inspect.getmembers(module):
        if (inspect.isclass(obj) and obj.__module__ == module_name
                and issubclass(obj, bt.feeds.PandasData)
                and obj is not bt.feeds.PandasData):
            custom_data_class = obj
            break

    # 查找自定义手续费类（继承自 bt.CommInfoBase）
    custom_commission_class = None
    for name, obj in inspect.getmembers(module):
        if (inspect.isclass(obj) and obj.__module__ == module_name
                and issubclass(obj, bt.CommInfoBase)
                and obj is not bt.CommInfoBase):
            custom_commission_class = obj
            break

    # 检测策略是否使用 trade_log
    use_trade_log_metrics = False
    try:
        source = inspect.getsource(strategy_class)
        use_trade_log_metrics = 'trade_log' in source
    except Exception:
        pass

    return strategy_class, module, custom_data_class, custom_commission_class, use_trade_log_metrics


def load_data(data_path: str) -> pd.DataFrame:
    """加载并标准化 CSV 数据"""
    df = pd.read_csv(data_path)
    if 'datetime' not in df.columns:
        for col in ('date', 'time_key', 'time'):
            if col in df.columns:
                df.rename(columns={col: 'datetime'}, inplace=True)
                break
    if 'datetime' not in df.columns:
        raise ValueError("数据文件必须包含 datetime/date/time_key 列")
    if pd.api.types.is_numeric_dtype(df['datetime']):
        unit = 'ms' if df['datetime'].iloc[0] > 1e12 else 's'
        df['datetime'] = pd.to_datetime(df['datetime'], unit=unit)
    else:
        df['datetime'] = pd.to_datetime(df['datetime'])
    return df


def load_params_file(path: str) -> dict:
    """解析 key=value 格式的参数文件，返回 {name: value} 字典"""
    params = {}
    with open(path, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                raise ValueError(f"第{lineno}行格式错误（需要 key = value）: {line}")
            key, val = line.split('=', 1)
            key, val = key.strip(), val.strip()
            # 自动推断类型：int / float
            try:
                params[key] = int(val)
            except ValueError:
                try:
                    params[key] = float(val)
                except ValueError:
                    raise ValueError(f"第{lineno}行值无法解析为数字: {key} = {val}")
    return params


def build_search_space(strategy_class, best_params: dict) -> dict:
    """从策略类提取参数并构建搜索空间"""
    from param_space_optimizer import ParamSpaceOptimizer
    EXCLUDED = {'printlog', 'verbose', 'mult', 'margin', 'commission', 'percent',
                'stocklike', 'commtype', 'percabs'}
    raw_params = []
    if hasattr(strategy_class, 'params'):
        for pname in dir(strategy_class.params):
            if pname.startswith('_') or pname.lower() in EXCLUDED:
                continue
            default_val = getattr(strategy_class.params, pname)
            if not isinstance(default_val, (int, float)):
                continue
            raw_params.append(StrategyParam(
                name=pname, param_type=type(default_val).__name__,
                default_value=default_val, description=f"{pname} parameter"))
    pso = ParamSpaceOptimizer(verbose=False)
    enriched = pso.generate_space(raw_params, strategy_type=strategy_class.__name__)
    return {p.name: p for p in enriched}


def main():
    parser = argparse.ArgumentParser(
        description="SPP 蒙特卡洛鲁棒性分析 (v3.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用参数文件（推荐）
  python run_spp_analysis.py \\
    --params-file best_params.txt \\
    -d project_trend/data/AG.csv \\
    -s project_trend/src/Aberration.py

  # 使用优化结果 JSON
  python run_spp_analysis.py \\
    -r optimization_results/AG/optimization_AG_Aberration_20260202.json \\
    -d project_trend/data/AG.csv \\
    -s project_trend/src/Aberration.py

  # 多数据源（策略需要多个CSV）
  python run_spp_analysis.py \\
    --params-file best_params.txt \\
    -d data/BTC.csv data/ETH.csv \\
    -s strategy.py

参数文件格式 (key = value):
  period = 35
  std_dev_upper = 2.1
  std_dev_lower = 1.8
        """
    )

    parser.add_argument("-r", "--result", default=None, help="优化结果 JSON 路径（与 --params-file 二选一）")
    parser.add_argument("--params-file", default=None,
                        help="参数文件路径（key=value 格式，与 -r 二选一）")
    parser.add_argument("-d", "--data", required=True, nargs='+',
                        help="CSV 数据文件路径（支持多个，空格分隔）")
    parser.add_argument("-s", "--strategy", required=True, help="策略 .py 文件路径")

    parser.add_argument("-n", "--samples", type=int, default=300,
                        help="蒙特卡洛采样次数 (默认: 300)")
    parser.add_argument("-p", "--perturbation", type=float, default=0.20,
                        help="扰动比例 (默认: 0.20)")
    parser.add_argument("-o", "--objective", default=None,
                        help="分析指标 (默认: 从 JSON 读取)")

    parser.add_argument("--use-llm", action="store_true",
                        help="启用 LLM 识别敏感参数")
    parser.add_argument("--llm-model", default="xuanyuan",
                        help="LLM 模型名 (默认: xuanyuan)")
    parser.add_argument("--sensitive-params", default=None,
                        help="手动指定敏感参数 (逗号分隔)")

    parser.add_argument("--output", default="./spp_results",
                        help="输出目录 (默认: ./spp_results)")
    parser.add_argument("--data-frequency", default=None,
                        help="数据频率 (daily, 1m, 5m, ...)")
    parser.add_argument("--asset-type", default="stock",
                        choices=["stock", "futures"])
    parser.add_argument("--contract-code", default=None, help="期货合约代码")
    parser.add_argument("-q", "--quiet", action="store_true", help="静默模式")

    args = parser.parse_args()

    # 校验：-r 和 --params-file 必须提供其一
    if not args.result and not args.params_file:
        parser.error("必须提供 -r/--result 或 --params-file 其中之一")
    if args.result and args.params_file:
        parser.error("-r/--result 和 --params-file 不能同时使用")

    # 1. 读取参数来源
    best_metrics = {}  # params-file 模式下稍后自动回测获取
    source_label = ''

    if args.params_file:
        # --- params-file 模式 ---
        if not Path(args.params_file).exists():
            print(f"错误: 参数文件不存在: {args.params_file}")
            return 1
        try:
            best_params = load_params_file(args.params_file)
        except ValueError as e:
            print(f"错误: 解析参数文件失败: {e}")
            return 1
        if not best_params:
            print("错误: 参数文件为空")
            return 1
        source_label = args.params_file
        objective = args.objective or 'sharpe_ratio'
    else:
        # --- JSON 模式（原有逻辑） ---
        if not Path(args.result).exists():
            print(f"错误: JSON 文件不存在: {args.result}")
            return 1
        with open(args.result, 'r', encoding='utf-8') as f:
            opt_result = json.load(f)
        best_params = opt_result.get('best_parameters', {})
        best_metrics = opt_result.get('performance_metrics', {})
        opt_info = opt_result.get('optimization_info',
                                  opt_result.get('optimization_metadata', {}))
        if not best_params:
            print("错误: JSON 中未找到 best_parameters")
            return 1
        source_label = args.result
        objective = args.objective or opt_info.get('optimization_objective', 'sharpe_ratio')

    asset_name = Path(args.data[0]).stem
    strategy_name = Path(args.strategy).stem
    verbose = not args.quiet
    multi_data = len(args.data) > 1

    if verbose:
        print(f"\n{'='*60}")
        print(f"SPP 蒙特卡洛鲁棒性分析 (v3.0)")
        print(f"{'='*60}")
        print(f"参数来源: {source_label}")
        if multi_data:
            print(f"数据文件: {len(args.data)} 个")
            for dp in args.data:
                print(f"  - {dp}")
        else:
            print(f"数据文件: {args.data[0]}")
        print(f"策略文件: {args.strategy}")
        print(f"分析指标: {objective}")
        print(f"采样次数: {args.samples}, 扰动: ±{args.perturbation:.0%}")
        print(f"最优参数: {best_params}")
        print(f"{'='*60}\n")

    # 2. 加载策略和数据
    try:
        strategy_class, strategy_module, custom_data_class, custom_commission_class, use_trade_log_metrics = load_strategy(
            str(Path(args.strategy).absolute()))
    except Exception as e:
        print(f"错误: 加载策略失败: {e}")
        return 1

    try:
        if multi_data:
            data = [load_data(dp) for dp in args.data]
            data_names = [Path(dp).stem for dp in args.data]
        else:
            data = load_data(args.data[0])
            data_names = None
    except Exception as e:
        print(f"错误: 加载数据失败: {e}")
        return 1

    if verbose:
        if multi_data:
            for dp, df in zip(args.data, data):
                print(f"[数据] {Path(dp).stem}: {len(df)} 条记录")
        else:
            print(f"[数据] {len(data)} 条记录")
        print(f"[策略] {strategy_class.__name__}")

    # 3. 构建经纪商配置
    broker_config = None
    if fc and args.asset_type == 'futures' and args.contract_code:
        try:
            broker_config = fc.build_broker_config(
                asset_type='futures', contract_code=args.contract_code)
        except Exception as e:
            print(f"[警告] 期货配置失败: {e}，使用默认股票配置")

    # 4. 构建搜索空间
    search_space = build_search_space(strategy_class, best_params)
    if verbose:
        print(f"[搜索空间] {len(search_space)} 个参数:")
        for name, sp in search_space.items():
            print(f"  {name}: [{sp.min_value}, {sp.max_value}] "
                  f"step={sp.step} type={sp.param_type}")

    # 5. 创建回测引擎
    effective_freq = None if (args.data_frequency is None
                              or args.data_frequency == 'auto') else args.data_frequency
    engine = BacktestEngine(
        data=data, strategy_class=strategy_class,
        data_frequency=effective_freq,
        strategy_module=strategy_module, broker_config=broker_config,
        custom_data_class=custom_data_class,
        custom_commission_class=custom_commission_class,
        use_trade_log_metrics=use_trade_log_metrics)

    # 5.1 params-file 模式：自动回测获取 best_metrics
    if not best_metrics:
        if verbose:
            print(f"[回测] 使用最优参数运行基准回测...")
        baseline = engine.run_backtest(strategy_class, data, best_params, calculate_yearly=False)
        if baseline is None:
            print("错误: 基准回测失败，无法获取 best_metrics")
            return 1
        best_metrics = {
            'sharpe_ratio': baseline.sharpe_ratio,
            'annual_return': baseline.annual_return,
            'max_drawdown': baseline.max_drawdown,
            'sortino_ratio': baseline.sortino_ratio,
            'calmar_ratio': baseline.calmar_ratio,
            'total_return': baseline.total_return,
            'trades_count': baseline.trades_count,
            'win_rate': baseline.win_rate,
        }
        if verbose:
            print(f"[回测] 基准 {objective}: {best_metrics.get(objective, 'N/A')}")

    # 6. 解析敏感参数
    sensitive_list = None
    if args.sensitive_params:
        sensitive_list = [s.strip() for s in args.sensitive_params.split(',') if s.strip()]

    # 7. 创建 LLM 客户端（如需要）
    llm_client = None
    if args.use_llm:
        try:
            from universal_llm_client import UniversalLLMClient, PRESET_CONFIGS
            preset_key = f"ollama-{args.llm_model}"
            if preset_key in PRESET_CONFIGS:
                llm_client = UniversalLLMClient(PRESET_CONFIGS[preset_key])
            else:
                from universal_llm_client import UniversalLLMConfig
                llm_client = UniversalLLMClient(UniversalLLMConfig(
                    api_type="ollama", base_url="http://localhost:11434",
                    model_name=args.llm_model))
            if verbose:
                print(f"[LLM] 已启用: {args.llm_model}")
        except Exception as e:
            print(f"[警告] LLM 初始化失败: {e}，将使用全部参数")

    # 8. 创建 SPP 分析器
    spp_config = SPPConfig(
        n_samples=args.samples,
        perturbation_ratio=args.perturbation,
        objective=objective,
        use_llm=args.use_llm,
        sensitive_params=sensitive_list,
    )

    analyzer = SPPAnalyzer(
        backtest_engine=engine, strategy_class=strategy_class,
        data=data, search_space=search_space,
        config=spp_config, verbose=verbose, llm_client=llm_client)

    # 9. 运行分析
    output_dir = os.path.join(args.output, asset_name)
    result = analyzer.run_full_analysis(
        best_params=best_params, best_metrics=best_metrics,
        output_dir=output_dir, asset_name=asset_name,
        strategy_name=strategy_name, source_json=source_label)

    return 0


if __name__ == "__main__":
    sys.exit(main())
