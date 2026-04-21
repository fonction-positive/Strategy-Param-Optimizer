#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并行优化收敛性测试

目标：验证 bug 修复后，批量并行优化是否能稳定收敛
方法：
  - 串行模式运行 1 次作为 baseline
  - 并行模式运行 3 次，检查结果稳定性和收敛性
数据：multivwap/data_1m_QQQ_test.csv + data_1m_TQQQ_test.csv (~25k rows, 3个月)
预计时间：~15-25 分钟

用法：
  .venv/bin/python test_convergence.py              # 串行×1 + 并行×3
  .venv/bin/python test_convergence.py --parallel-only   # 只跑并行×3
"""

import sys
import os
import time
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# === 路径设置 ===
PROJECT_ROOT = Path(__file__).parent
OPTIMIZER_DIR = PROJECT_ROOT / "Optimizer"
sys.path.insert(0, str(OPTIMIZER_DIR))

from config import BayesianOptConfig, ParallelConfig, BatchParallelConfig
from universal_optimizer import UniversalOptimizer

# ============================================================
#                        测试配置
# ============================================================
STRATEGY_PATH = str(PROJECT_ROOT / "multivwap" / "multivwap2.py")
DATA_PATHS = [
    str(PROJECT_ROOT / "multivwap" / "data_1m_QQQ_test.csv"),
    str(PROJECT_ROOT / "multivwap" / "data_1m_TQQQ_test.csv"),
]
DATA_NAMES = ["QQQ", "TQQQ"]
SPACE_CONFIG_PATH = str(PROJECT_ROOT / "multivwap" / "multivwap_space_config.json")
PARAMS_FILE = str(PROJECT_ROOT / "multivwap" / "params-file.txt")

OBJECTIVE = "sharpe_ratio"
TRIALS = 96                # 每次优化总试验数
N_PARALLEL_RUNS = 3        # 并行模式重复次数
BATCH_SIZE = 4             # 批量大小
N_WORKERS = 4              # 并行工作进程数
DATA_FREQ = "1m"


def load_target_params(path: str) -> list:
    params = []
    with open(path, 'r') as f:
        for line in f:
            p = line.strip()
            if p and not p.startswith('#'):
                params.append(p)
    return params


def load_space_config(path: str) -> dict:
    """加载搜索空间配置，返回 param_space 字典"""
    with open(path, 'r') as f:
        config = json.load(f)
    return config.get("param_space", config)


# 预加载（只加载一次）
TARGET_PARAMS = load_target_params(PARAMS_FILE)
CUSTOM_SPACE = load_space_config(SPACE_CONFIG_PATH)


def run_single_optimization(mode: str, run_id: int) -> dict:
    """
    运行一次完整优化，返回结果摘要。
    """
    is_parallel = (mode == "parallel")

    parallel_config = ParallelConfig(
        enable_parallel=is_parallel,
        n_workers=N_WORKERS if is_parallel else 1,
    )
    batch_parallel_config = BatchParallelConfig(
        enable_batch_parallel=is_parallel,
        batch_size=BATCH_SIZE if is_parallel else 1,
        adaptive_batch=False,
        hybrid_mode=True,
        parallel_ratio=0.7,
    )

    optimizer = UniversalOptimizer(
        data_path=DATA_PATHS,
        strategy_path=STRATEGY_PATH,
        objective=OBJECTIVE,
        data_frequency=DATA_FREQ,
        data_names=DATA_NAMES,
        verbose=False,
        target_params=TARGET_PARAMS,
        custom_space=CUSTOM_SPACE,
        parallel_config=parallel_config,
        batch_parallel_config=batch_parallel_config,
    )

    t0 = time.time()
    result = optimizer.optimize(
        n_trials=TRIALS,
        auto_expand_boundary=False,
        use_enhanced_sampler=True,
        enable_dynamic_trials=False,
    )
    elapsed = time.time() - t0

    # --- 从返回的 result 字典中提取信息 ---
    metrics = result.get("performance_metrics", {})
    best_params = result.get("best_parameters", {})
    improvements = result.get("best_improvements", [])

    # 从 improvements 提取收敛轨迹（每次发现更优参数时的 Sharpe）
    improvement_sharpes = [
        imp["backtest_results"]["sharpe_ratio"]
        for imp in improvements
        if "backtest_results" in imp
    ]

    return {
        "mode": mode,
        "run_id": run_id,
        "sharpe": metrics.get("sharpe_ratio", 0),
        "annual_return": metrics.get("annual_return", 0),
        "max_drawdown": metrics.get("max_drawdown", 0),
        "total_return": metrics.get("total_return", 0),
        "trades_count": metrics.get("trades_count", 0),
        "best_params": best_params,
        "elapsed": elapsed,
        "n_improvements": len(improvements),
        "improvement_sharpes": improvement_sharpes,
    }


def main():
    parser = argparse.ArgumentParser(description="并行优化收敛性测试")
    parser.add_argument("--parallel-only", action="store_true",
                        help="只跑并行模式，跳过串行 baseline")
    args = parser.parse_args()

    parallel_only = args.parallel_only
    total_start = time.time()

    mode_desc = f"并行×{N_PARALLEL_RUNS}" if parallel_only else f"串行×1 + 并行×{N_PARALLEL_RUNS}"
    print(f"\n{'=' * 80}")
    print(f"  并行优化收敛性测试")
    print(f"  策略: multivwap2 | 数据: QQQ+TQQQ 1m test (3个月)")
    print(f"  Trials: {TRIALS} | {mode_desc}")
    print(f"  Workers: {N_WORKERS} | Batch: {BATCH_SIZE}")
    print(f"  开始时间: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'=' * 80}\n")

    all_results = []
    total_runs = N_PARALLEL_RUNS if parallel_only else 1 + N_PARALLEL_RUNS
    run_counter = 0

    # ======== 1. 串行 baseline（可跳过）========
    serial_result = None
    if not parallel_only:
        run_counter += 1
        print(f"[{run_counter}/{total_runs}] 串行模式 baseline ...")
        serial_result = run_single_optimization("serial", 0)
        all_results.append(serial_result)
        print(f"  -> Sharpe={serial_result['sharpe']:.4f}  "
              f"耗时={serial_result['elapsed']:.1f}s\n")

    # ======== 2. 并行模式 × N 次 ========
    for i in range(N_PARALLEL_RUNS):
        run_counter += 1
        print(f"[{run_counter}/{total_runs}] 并行模式 Run {i+1}/{N_PARALLEL_RUNS} ...")
        par_result = run_single_optimization("parallel", i + 1)
        all_results.append(par_result)
        print(f"  -> Sharpe={par_result['sharpe']:.4f}  "
              f"耗时={par_result['elapsed']:.1f}s\n")

    total_elapsed = time.time() - total_start

    # ============================================================
    #                     结果分析
    # ============================================================
    parallels = [r for r in all_results if r["mode"] == "parallel"]
    par_sharpes = [r["sharpe"] for r in parallels]
    par_times = [r["elapsed"] for r in parallels]

    print(f"\n{'=' * 80}")
    print(f"  结果对照表")
    print(f"{'=' * 80}")

    header = f"{'Run':<12} {'Sharpe':>10} {'年化%':>8} {'回撤%':>8} {'总收益%':>8} {'交易':>6} {'耗时s':>8}"
    print(f"\n{header}")
    print(f"{'-' * 70}")

    for r in all_results:
        tag = "串行" if r['mode'] == 'serial' else f"并行#{r['run_id']}"
        print(f"{tag:<12} {r['sharpe']:>10.4f} {r['annual_return']:>8.2f} "
              f"{r['max_drawdown']:>8.2f} {r['total_return']:>8.2f} "
              f"{r['trades_count']:>6} {r['elapsed']:>8.1f}")

    # --- 收敛性指标 ---
    print(f"\n{'=' * 80}")
    print(f"  收敛性分析")
    print(f"{'=' * 80}")

    par_mean = float(np.mean(par_sharpes))
    par_std = float(np.std(par_sharpes))
    par_cv = (par_std / abs(par_mean) * 100) if par_mean != 0 else float('inf')
    par_min = float(np.min(par_sharpes))
    par_max = float(np.max(par_sharpes))

    if serial_result:
        print(f"\n  串行 baseline Sharpe:      {serial_result['sharpe']:.4f}")

    print(f"\n  并行 {N_PARALLEL_RUNS} 次:")
    print(f"    平均 Sharpe:             {par_mean:.4f}")
    print(f"    标准差:                  {par_std:.4f}")
    print(f"    变异系数 (CV):           {par_cv:.1f}%")
    print(f"    范围:                    [{par_min:.4f}, {par_max:.4f}]")

    if serial_result and serial_result['sharpe'] != 0:
        gap = (par_mean - serial_result['sharpe']) / abs(serial_result['sharpe']) * 100
        print(f"    与串行差距:              {gap:+.1f}%")

    # --- 收敛轨迹（每次发现更优参数的 Sharpe） ---
    print(f"\n{'=' * 80}")
    print(f"  改进轨迹 (每次发现更优参数时的 Sharpe)")
    print(f"{'=' * 80}\n")

    def format_trajectory(label, sharpes):
        if not sharpes:
            return f"  {label:<12} (无改进记录)"
        items = [f"{s:.4f}" for s in sharpes]
        return f"  {label:<12} {' -> '.join(items)}"

    if serial_result:
        print(format_trajectory("串行", serial_result["improvement_sharpes"]))
    for r in parallels:
        print(format_trajectory(f"并行#{r['run_id']}", r["improvement_sharpes"]))

    # --- 最优参数对比 ---
    print(f"\n{'=' * 80}")
    print(f"  最优参数对比")
    print(f"{'=' * 80}\n")

    all_param_names = sorted(set(
        k for r in all_results for k in r["best_params"].keys()
    ))

    header = f"  {'参数':<25}" + "".join(
        f"{'串行' if r['mode']=='serial' else '并行#'+str(r['run_id']):>12}"
        for r in all_results
    )
    print(header)
    print(f"  {'-' * (25 + 12 * len(all_results))}")

    for param in all_param_names:
        row = f"  {param:<25}"
        for r in all_results:
            v = r["best_params"].get(param, "N/A")
            if isinstance(v, float):
                row += f"{v:>12.4f}"
            else:
                row += f"{str(v):>12}"
        print(row)

    # --- 稳定性判定 ---
    print(f"\n{'=' * 80}")
    print(f"  结论")
    print(f"{'=' * 80}")

    is_stable = par_cv < 30
    runs_with_improvement = sum(1 for r in parallels if r["n_improvements"] >= 2)

    print(f"\n  1. 并行结果稳定性:  {'PASS' if is_stable else 'FAIL'} "
          f"(CV={par_cv:.1f}%, 阈值<30%)")

    quality_ratio = None
    is_quality_ok = None
    if serial_result and serial_result['sharpe'] != 0:
        quality_ratio = par_mean / serial_result['sharpe']
        is_quality_ok = quality_ratio > 0.7
        print(f"  2. 搜索质量:        {'PASS' if is_quality_ok else 'FAIL'} "
              f"(并行/串行={quality_ratio:.2f}, 阈值>0.70)")
    else:
        print(f"  2. 搜索质量:        N/A (无串行 baseline)")

    print(f"  3. 单次运行收敛:    {runs_with_improvement}/{N_PARALLEL_RUNS} "
          f"个并行运行有 >=2 次改进")

    if is_stable and (is_quality_ok is None or is_quality_ok):
        verdict = "PASS — 并行优化能够稳定收敛"
    elif is_stable:
        verdict = "PARTIAL — 结果稳定但搜索质量仍有差距"
    else:
        verdict = "FAIL — 并行结果仍有较大波动"

    print(f"\n  >>> 总体判定: {verdict}")

    if serial_result:
        speed_ratio = serial_result['elapsed'] / np.mean(par_times) if np.mean(par_times) > 0 else 0
        print(f"\n  速度: 串行 {serial_result['elapsed']:.1f}s vs 并行平均 {np.mean(par_times):.1f}s "
              f"(加速比 {speed_ratio:.2f}x)")
    else:
        speed_ratio = None
        print(f"\n  并行平均耗时: {np.mean(par_times):.1f}s")

    print(f"  总测试耗时: {total_elapsed:.1f}s ({total_elapsed/60:.1f} 分钟)")
    print(f"{'=' * 80}\n")

    # --- 保存 JSON ---
    output = {
        "test_time": datetime.now().isoformat(),
        "config": {
            "trials": TRIALS,
            "n_parallel_runs": N_PARALLEL_RUNS,
            "batch_size": BATCH_SIZE,
            "n_workers": N_WORKERS,
            "parallel_only": parallel_only,
        },
        "serial": {
            "sharpe": serial_result["sharpe"],
            "annual_return": serial_result["annual_return"],
            "max_drawdown": serial_result["max_drawdown"],
            "total_return": serial_result["total_return"],
            "best_params": serial_result["best_params"],
            "elapsed": serial_result["elapsed"],
        } if serial_result else None,
        "parallel_runs": [
            {
                "run_id": r["run_id"],
                "sharpe": r["sharpe"],
                "annual_return": r["annual_return"],
                "max_drawdown": r["max_drawdown"],
                "total_return": r["total_return"],
                "best_params": r["best_params"],
                "elapsed": r["elapsed"],
            }
            for r in parallels
        ],
        "analysis": {
            "par_mean_sharpe": par_mean,
            "par_std_sharpe": par_std,
            "par_cv_pct": par_cv,
            "quality_ratio": quality_ratio,
            "speed_ratio": float(speed_ratio) if speed_ratio else None,
            "verdict": verdict,
            "total_elapsed_s": total_elapsed,
        },
    }

    output_path = PROJECT_ROOT / "convergence_test_result.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"  结果已保存到: {output_path}\n")


if __name__ == "__main__":
    main()
