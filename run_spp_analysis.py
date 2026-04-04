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

optimizer_path = str(Path(__file__).parent / "optimizer")
if optimizer_path not in sys.path:
    sys.path.insert(0, optimizer_path)

from backtest_engine import BacktestEngine
from config import StrategyParam
from spp_analyzer import SPPConfig, SPPAnalyzer

try:
    import futures_config as fc
except ImportError:
    fc = None


def load_strategy(strategy_path: str):
    """动态加载策略类"""
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
    return strategy_classes[0], module


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
            try:
                params[key] = int(val)
            except ValueError:
                try:
                    params[key] = float(val)
                except ValueError:
                    raise ValueError(f"第{lineno}行值无法解析为数字: {key} = {val}")
    return params


def validate_and_merge_center_params(base_params: dict, override_params: dict,
                                     search_space: dict):
    """校验并合并中心参数覆盖"""
    unknown = sorted(set(override_params) - set(search_space))
    if unknown:
        valid = ', '.join(sorted(search_space))
        raise ValueError(f"未知参数: {', '.join(unknown)}。可用参数: {valid}")

    normalized = {}
    warnings = []
    for name, value in override_params.items():
        sp = search_space[name]
        if sp.param_type == 'int':
            if isinstance(value, float) and not float(value).is_integer():
                raise ValueError(f"参数 {name} 需要整数，当前值为 {value}")
            normalized_value = int(value)
        else:
            normalized_value = float(value)

        if normalized_value < sp.min_value or normalized_value > sp.max_value:
            warnings.append(
                f"参数 {name}={normalized_value} 超出搜索空间 [{sp.min_value}, {sp.max_value}]"
            )
        normalized[name] = normalized_value

    merged = dict(base_params)
    merged.update(normalized)
    return merged, normalized, warnings


def build_metrics_from_backtest(result) -> dict:
    """从回测结果提取 SPP 所需指标"""
    return {
        'sharpe_ratio': result.sharpe_ratio,
        'annual_return': result.annual_return,
        'max_drawdown': result.max_drawdown,
        'sortino_ratio': result.sortino_ratio,
        'calmar_ratio': result.calmar_ratio,
        'total_return': result.total_return,
        'trades_count': result.trades_count,
        'win_rate': result.win_rate,
    }


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
  python run_spp_analysis.py \\
    -r optimization_results/AG/optimization_AG_Aberration_20260202.json \\
    -d project_trend/data/AG.csv \\
    -s project_trend/src/Aberration.py

  # 手动覆盖 SPP 中心参数
  python run_spp_analysis.py \
    -r result.json -d data.csv -s strategy.py \
    --center-params-file center_params.txt

  # 手动指定敏感参数
  python run_spp_analysis.py \\
    -r result.json -d data.csv -s strategy.py \\
    --sensitive-params period,std_dev_upper

  # 使用 LLM 识别敏感参数
  python run_spp_analysis.py \\
    -r result.json -d data.csv -s strategy.py \\
    --use-llm --llm-model xuanyuan

中心参数文件格式 (key = value):
  period = 35
  std_dev_upper = 2.1
        """
    )

    parser.add_argument("-r", "--result", required=True, help="优化结果 JSON 路径")
    parser.add_argument("-d", "--data", required=True, help="CSV 数据文件路径")
    parser.add_argument("-s", "--strategy", required=True, help="策略 .py 文件路径")

    parser.add_argument("-n", "--samples", type=int, default=300,
                        help="蒙特卡洛采样次数 (默认: 300)")
    parser.add_argument("-p", "--perturbation", type=float, default=0.20,
                        help="扰动比例 (默认: 0.20)")
    parser.add_argument("-o", "--objective", default=None,
                        help="分析指标 (默认: 从 JSON 读取)")

    parser.add_argument("--center-params-file", default=None,
                        help="中心参数覆盖文件（key=value 格式，在 JSON best_parameters 基础上覆盖）")
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

    # 1. 读取优化结果 JSON
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

    objective = args.objective or opt_info.get('optimization_objective', 'sharpe_ratio')
    asset_name = opt_info.get('asset_name', Path(args.data).stem)
    strategy_name = opt_info.get('strategy_name', Path(args.strategy).stem)
    verbose = not args.quiet
    parameter_source = 'result_json'
    parameter_source_path = args.result
    center_override_file = args.center_params_file
    center_overrides = {}
    baseline_recomputed = False

    if verbose:
        print(f"\n{'='*60}")
        print(f"SPP 蒙特卡洛鲁棒性分析 (v3.0)")
        print(f"{'='*60}")
        print(f"参数来源: {parameter_source_path}")
        if center_override_file:
            print(f"中心参数覆盖: {center_override_file}")
        print(f"数据文件: {args.data}")
        print(f"策略文件: {args.strategy}")
        print(f"分析指标: {objective}")
        print(f"采样次数: {args.samples}, 扰动: ±{args.perturbation:.0%}")
        print(f"基础参数: {best_params}")
        print(f"{'='*60}\n")

    # 2. 加载策略和数据
    try:
        strategy_class, strategy_module = load_strategy(
            str(Path(args.strategy).absolute()))
    except Exception as e:
        print(f"错误: 加载策略失败: {e}")
        return 1

    try:
        data = load_data(args.data)
    except Exception as e:
        print(f"错误: 加载数据失败: {e}")
        return 1

    if verbose:
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
        strategy_module=strategy_module, broker_config=broker_config)

    if args.center_params_file:
        if not Path(args.center_params_file).exists():
            print(f"错误: 中心参数文件不存在: {args.center_params_file}")
            return 1
        try:
            raw_center_overrides = load_params_file(args.center_params_file)
            best_params, center_overrides, center_warnings = validate_and_merge_center_params(
                best_params, raw_center_overrides, search_space)
        except ValueError as e:
            print(f"错误: 解析中心参数文件失败: {e}")
            return 1
        if verbose:
            print(f"[中心参数] 覆盖文件: {args.center_params_file}")
            print(f"[中心参数] 生效覆盖: {center_overrides}")
            for warning in center_warnings:
                print(f"[警告] {warning}")

        baseline = engine.run_backtest(
            strategy_class, data, best_params, calculate_yearly=False)
        if baseline is None:
            print("错误: 覆盖中心参数后的基准回测失败")
            return 1
        best_metrics = build_metrics_from_backtest(baseline)
        baseline_recomputed = True
        if verbose:
            print(f"[回测] 已基于覆盖后的中心参数重算基准 {objective}: {best_metrics.get(objective, 'N/A')}")

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
        strategy_name=strategy_name, source_json=args.result,
        provenance={
            'parameter_source': parameter_source,
            'parameter_source_path': parameter_source_path,
            'center_override_file': center_override_file,
            'center_params_overridden': sorted(center_overrides),
            'baseline_recomputed': baseline_recomputed,
        })

    return 0


if __name__ == "__main__":
    sys.exit(main())
