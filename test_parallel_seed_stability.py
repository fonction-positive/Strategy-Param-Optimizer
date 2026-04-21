# -*- coding: utf-8 -*-
"""
不同 seed 稳定性测试（最小样本，12 核 Mac 友好）
==============================================

固定数据/策略/搜索空间/trial 数，只变化随机 seed，跑 N 次并行优化，
统计 best_value 的分布：mean / std / min / max / median / range。

用来评估：
  - 并行 TPE + constant_liar 在相同预算下的稳定性
  - 不同 seed 对最佳参数的敏感性（参数漂移）

用法:
  python3 test_parallel_seed_stability.py
  python3 test_parallel_seed_stability.py --seeds 1 2 3 4 5 --trials 60
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

# 项目路径
PROJECT_ROOT = Path(__file__).resolve().parent
OPTIMIZER_DIR = PROJECT_ROOT / "Optimizer"
sys.path.insert(0, str(OPTIMIZER_DIR))

from strategy_analyzer import SearchSpaceConfig  # noqa: E402
from enhanced_sampler import NormalDistributionSampler, SamplerConfig  # noqa: E402
from parallel_engine import (  # noqa: E402
    BatchParallelOptimizer,
    WorkerInitArgs,
    create_parallel_study,
)

# ----- 测试配置 --------------------------------------------------------------
DATA_PATH = PROJECT_ROOT / "multivwap" / "data_1m_QQQ_test.csv"
TQQQ_PATH = PROJECT_ROOT / "multivwap" / "data_1m_TQQQ_test.csv"  # 添加TQQQ路径
STRATEGY_PATH = PROJECT_ROOT / "multivwap" / "multivwap2_test.py"
SPACE_CONFIG_PATH = PROJECT_ROOT / "multivwap" / "multivwap_space_config.json"
OUTPUT_DIR = PROJECT_ROOT / "test_results" / "seed_stability"

DEFAULT_PARAMS = {
    "volatility_window": 10,
    "leverage": 2.5,
    "target_volatility": 0.03,
    "trade_frequency": 1,
    "position_calculation_mode": 0,
    "lookback": 1,
    "VM": 0.8,
    "mode": 1,
}


def build_search_space(space_cfg_path: Path) -> dict:
    with open(space_cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)["param_space"]
    space = {}
    for name, c in cfg.items():
        distribution = c.get("distribution", "uniform")
        param_type = "int" if distribution == "int_uniform" else "float"
        space[name] = SearchSpaceConfig(
            param_name=name,
            param_type=param_type,
            distribution=distribution,
            min_value=float(c["min"]),
            max_value=float(c["max"]),
            step=c.get("step"),
        )
    return space


def run_once(
    search_space: dict,
    seed: int,
    n_trials: int,
    explore_ratio: float,
    n_workers: int,
    explore_batch: int,
    exploit_batch: int,
    trial_timeout: int,
    objective: str,
) -> Dict:
    """跑一次完整优化，返回汇总字典。"""
    n_explore = int(n_trials * explore_ratio)
    n_exploit = n_trials - n_explore

    sampler = NormalDistributionSampler(SamplerConfig(), seed=seed)
    exploration_samples, _ = sampler.generate_initial_samples(
        search_space=search_space,
        n_samples=n_explore,
        default_params=DEFAULT_PARAMS,
        include_default=True,
    )

    worker_init = WorkerInitArgs(
        data_paths=[str(DATA_PATH), str(TQQQ_PATH)],  # 添加TQQQ数据路径
        data_names=["QQQ", "TQQQ"],  # 数据名称列表
        strategy_path=str(STRATEGY_PATH),
        objective=objective,
        data_frequency="1m",
        initial_cash=100000.0,
        commission=0.001,
        is_multi_data=True,
    )

    study = create_parallel_study(
        direction="maximize",
        seed=seed,
        n_startup_trials=min(10, max(3, n_exploit // 5)),
        constant_liar=True,
    )

    engine = BatchParallelOptimizer(
        study=study,
        search_space=search_space,
        worker_init_args=worker_init,
        n_exploit_trials=n_exploit,
        exploration_samples=exploration_samples,
        n_workers=n_workers,
        explore_batch_size=explore_batch,
        exploit_batch_size=exploit_batch,
        timeout_per_trial=trial_timeout,
        verbose=False,  # 多次运行时关掉每批日志
    )

    t0 = time.time()
    best_params, best_value, best_metrics = engine.run()
    elapsed = time.time() - t0

    failed = sum(1 for r in engine.history if r.score == float("-inf"))
    return {
        "seed": seed,
        "best_value": best_value,
        "best_params": best_params,
        "best_metrics": best_metrics or {},
        "elapsed_s": elapsed,
        "n_trials": len(engine.history),
        "failed_trials": failed,
    }


def summarize(values: List[float]) -> Dict[str, float]:
    clean = [v for v in values if v != float("-inf")]
    if not clean:
        return {"mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan"),
                "median": float("nan"), "range": float("nan"), "cv": float("nan")}
    mean = statistics.fmean(clean)
    std = statistics.pstdev(clean) if len(clean) > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "min": min(clean),
        "max": max(clean),
        "median": statistics.median(clean),
        "range": max(clean) - min(clean),
        "cv": (std / abs(mean)) if mean != 0 else float("inf"),  # 变异系数
    }


def main():
    parser = argparse.ArgumentParser(description="并行优化器 seed 稳定性测试")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[1, 7, 42, 123, 2024],
                        help="要测试的 seed 列表")
    parser.add_argument("--trials", type=int, default=60,
                        help="每次运行的总 trial 数（默认 60）")
    parser.add_argument("--explore-ratio", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=10,
                        help="并行进程数（12 核 Mac 建议 10）")
    parser.add_argument("--explore-batch", type=int, default=10)
    parser.add_argument("--exploit-batch", type=int, default=8)
    parser.add_argument("--trial-timeout", type=int, default=180)
    parser.add_argument("--objective", default="sharpe_ratio")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    for p in (DATA_PATH, STRATEGY_PATH, SPACE_CONFIG_PATH):
        if not p.exists():
            print(f"[FATAL] 缺少文件: {p}")
            return 1

    output_dir = Path(args.output) if args.output else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    search_space = build_search_space(SPACE_CONFIG_PATH)

    print("=" * 70)
    print("并行优化器 · 不同 seed 稳定性测试")
    print("=" * 70)
    print(f"数据: {DATA_PATH.name}  策略: {STRATEGY_PATH.name}")
    print(f"每次 trials: {args.trials}   objective: {args.objective}")
    print(f"seeds: {args.seeds}   workers: {args.workers}")
    print(f"exploit_batch={args.exploit_batch}  explore_batch={args.explore_batch}")
    print("=" * 70)

    runs = []
    wall_start = time.time()
    for idx, seed in enumerate(args.seeds, 1):
        print(f"\n[{idx}/{len(args.seeds)}] seed={seed} 启动...")
        result = run_once(
            search_space=search_space,
            seed=seed,
            n_trials=args.trials,
            explore_ratio=args.explore_ratio,
            n_workers=args.workers,
            explore_batch=args.explore_batch,
            exploit_batch=args.exploit_batch,
            trial_timeout=args.trial_timeout,
            objective=args.objective,
        )
        m = result["best_metrics"]
        print(f"  → best={result['best_value']:.4f}  "
              f"sharpe={m.get('sharpe_ratio', 0):.3f}  "
              f"ann={m.get('annual_return', 0):.2f}%  "
              f"dd={m.get('max_drawdown', 0):.2f}%  "
              f"trades={m.get('trades_count', 0)}  "
              f"({result['elapsed_s']:.1f}s, failed={result['failed_trials']})")
        runs.append(result)

    total_elapsed = time.time() - wall_start

    # ---------- 统计 ----------
    print("\n" + "=" * 70)
    print("稳定性汇总")
    print("=" * 70)

    best_values = [r["best_value"] for r in runs]
    stats = summarize(best_values)
    print(f"best_value 统计 (n={len(runs)}):")
    print(f"  mean   : {stats['mean']:.4f}")
    print(f"  std    : {stats['std']:.4f}")
    print(f"  min    : {stats['min']:.4f}")
    print(f"  max    : {stats['max']:.4f}")
    print(f"  median : {stats['median']:.4f}")
    print(f"  range  : {stats['range']:.4f}")
    print(f"  CV     : {stats['cv'] * 100:.2f}%   (变异系数，越小越稳定)")

    # 每个关键指标分别统计
    for metric_key in ("sharpe_ratio", "annual_return", "max_drawdown", "trades_count"):
        vals = [r["best_metrics"].get(metric_key, 0) for r in runs]
        s = summarize(vals)
        print(f"{metric_key:>14} : "
              f"mean={s['mean']:.3f}  std={s['std']:.3f}  "
              f"range=[{s['min']:.3f}, {s['max']:.3f}]")

    # ---------- 参数漂移 ----------
    print("\n" + "-" * 70)
    print("不同 seed 选到的最佳参数（观察漂移幅度）")
    print("-" * 70)
    param_names = sorted(search_space.keys())
    header = f"{'seed':>6} " + " ".join(f"{n[:10]:>11}" for n in param_names)
    print(header)
    for r in runs:
        row = f"{r['seed']:>6} " + " ".join(
            f"{r['best_params'].get(n, 0):>11.3f}" if isinstance(r['best_params'].get(n), float)
            else f"{r['best_params'].get(n, ''):>11}" for n in param_names
        )
        print(row)

    print("\n每个参数的 std / range（相对搜索空间宽度）:")
    for n in param_names:
        vals = [r["best_params"].get(n) for r in runs if n in r["best_params"]]
        if not vals:
            continue
        try:
            vf = [float(v) for v in vals]
        except (TypeError, ValueError):
            continue
        if len(vf) < 2:
            continue
        sp = search_space[n]
        width = float(sp.max_value) - float(sp.min_value)
        std = statistics.pstdev(vf)
        rng = max(vf) - min(vf)
        rel_std = (std / width * 100) if width > 0 else 0.0
        rel_rng = (rng / width * 100) if width > 0 else 0.0
        print(f"  {n:28s} std={std:8.3f} ({rel_std:5.1f}% of space)  "
              f"range={rng:8.3f} ({rel_rng:5.1f}% of space)")

    # ---------- 落盘 ----------
    csv_path = output_dir / f"seed_stability_n{args.trials}_{len(args.seeds)}seeds.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["seed", "best_value", "sharpe_ratio", "annual_return",
             "max_drawdown", "trades_count", "failed_trials",
             "elapsed_s"] + param_names
        )
        for r in runs:
            m = r["best_metrics"]
            w.writerow(
                [r["seed"], r["best_value"],
                 m.get("sharpe_ratio", ""), m.get("annual_return", ""),
                 m.get("max_drawdown", ""), m.get("trades_count", ""),
                 r["failed_trials"], f"{r['elapsed_s']:.2f}"]
                + [r["best_params"].get(n, "") for n in param_names]
            )
    print(f"\n[写入] 明细 CSV: {csv_path}")

    json_path = output_dir / f"seed_stability_n{args.trials}_{len(args.seeds)}seeds.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": vars(args),
                "stats_best_value": stats,
                "runs": [
                    {
                        "seed": r["seed"],
                        "best_value": r["best_value"],
                        "best_params": r["best_params"],
                        "best_metrics": r["best_metrics"],
                        "elapsed_s": r["elapsed_s"],
                        "failed_trials": r["failed_trials"],
                    }
                    for r in runs
                ],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[写入] 明细 JSON: {json_path}")

    print(f"\n总墙钟耗时: {total_elapsed:.1f}s "
          f"(平均每个 seed {total_elapsed / max(1, len(args.seeds)):.1f}s)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
