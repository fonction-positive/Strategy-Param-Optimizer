# -*- coding: utf-8 -*-
"""
收敛性测试（最小样本，12 核 Mac 友好）
======================================

在 multivwap/data_1m_QQQ_test.csv × multivwap2_test.py 上跑一次
`BatchParallelOptimizer`，记录每个 trial 的 score，输出：

  1. 逐 trial 的 running-best 曲线（stdout + CSV）
  2. 分阶段收敛速度（探索 vs 利用）
  3. 可选：matplotlib 收敛曲线 PNG（如已安装 matplotlib）

用法:
  python3 test_parallel_convergence.py
  python3 test_parallel_convergence.py --trials 100 --workers 10
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

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

# ----- 测试配置（针对 12 核 Mac 调小） ----------------------------------------
DATA_PATH = PROJECT_ROOT / "multivwap" / "data_1m_QQQ_test.csv"
TQQQ_PATH = PROJECT_ROOT / "multivwap" / "data_1m_TQQQ_test.csv"  # 添加TQQQ路径
STRATEGY_PATH = PROJECT_ROOT / "multivwap" / "multivwap2_test.py"
SPACE_CONFIG_PATH = PROJECT_ROOT / "multivwap" / "multivwap_space_config.json"
OUTPUT_DIR = PROJECT_ROOT / "test_results" / "convergence"


def build_search_space(space_cfg_path: Path) -> dict:
    """从 multivwap_space_config.json 构造 {name: SearchSpaceConfig} 字典。"""
    with open(space_cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)["param_space"]

    space = {}
    for name, c in cfg.items():
        distribution = c.get("distribution", "uniform")
        if distribution == "int_uniform":
            param_type = "int"
        else:
            param_type = "float"
        space[name] = SearchSpaceConfig(
            param_name=name,
            param_type=param_type,
            distribution=distribution,
            min_value=float(c["min"]),
            max_value=float(c["max"]),
            step=c.get("step"),
        )
    return space


# 策略默认参数（从 multivwap2_test.py 中抄下来，用作探索阶段 Trial 0）
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


def main():
    parser = argparse.ArgumentParser(description="并行优化器收敛性测试")
    parser.add_argument("--trials", type=int, default=80,
                        help="总 trial 数（默认 80）")
    parser.add_argument("--workers", type=int, default=10,
                        help="并行进程数（12 核 Mac 建议 10）")
    parser.add_argument("--exploit-batch", type=int, default=8,
                        help="利用阶段 batch size（默认 8）")
    parser.add_argument("--explore-batch", type=int, default=10,
                        help="探索阶段 batch size（默认 10）")
    parser.add_argument("--explore-ratio", type=float, default=0.3,
                        help="探索阶段占比（默认 0.3）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trial-timeout", type=int, default=180)
    parser.add_argument("--objective", default="sharpe_ratio")
    parser.add_argument("--output", default=None,
                        help="输出目录（默认 test_results/convergence）")
    args = parser.parse_args()

    # 验证必需文件
    for p in (DATA_PATH, TQQQ_PATH, STRATEGY_PATH, SPACE_CONFIG_PATH):
        if not p.exists():
            print(f"[FATAL] 缺少文件: {p}")
            return 1

    output_dir = Path(args.output) if args.output else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # 拆分 explore / exploit
    n_explore = int(args.trials * args.explore_ratio)
    n_exploit = args.trials - n_explore

    print("=" * 70)
    print("并行优化器 · 收敛性测试")
    print("=" * 70)
    print(f"数据: {DATA_PATH}")
    print(f"策略: {STRATEGY_PATH}")
    print(f"总 trials: {args.trials}  (探索={n_explore}, 利用={n_exploit})")
    print(f"并行进程: {args.workers}  "
          f"explore_batch={args.explore_batch}  exploit_batch={args.exploit_batch}")
    print(f"seed: {args.seed}  objective: {args.objective}")
    print("=" * 70)

    search_space = build_search_space(SPACE_CONFIG_PATH)

    # 预生成探索样本（Trial 0 = 策略默认参数）
    sampler = NormalDistributionSampler(SamplerConfig(), seed=args.seed)
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
        objective=args.objective,
        data_frequency="1m",
        initial_cash=100000.0,
        commission=0.001,
        is_multi_data=True,
    )

    study = create_parallel_study(
        direction="maximize",
        seed=args.seed,
        n_startup_trials=min(10, max(3, n_exploit // 5)),
        constant_liar=True,
    )

    engine = BatchParallelOptimizer(
        study=study,
        search_space=search_space,
        worker_init_args=worker_init,
        n_exploit_trials=n_exploit,
        exploration_samples=exploration_samples,
        n_workers=args.workers,
        explore_batch_size=args.explore_batch,
        exploit_batch_size=args.exploit_batch,
        timeout_per_trial=args.trial_timeout,
        verbose=True,
    )

    t0 = time.time()
    best_params, best_value, best_metrics = engine.run()
    elapsed = time.time() - t0

    # ---------- 收敛分析 ----------
    print("\n" + "=" * 70)
    print("收敛曲线 (trial → running-best score)")
    print("=" * 70)

    running_best = float("-inf")
    curve = []
    explore_best = float("-inf")
    exploit_best = float("-inf")
    failed = 0
    for rec in engine.history:
        if rec.score == float("-inf"):
            failed += 1
        if rec.score > running_best:
            running_best = rec.score
        curve.append((rec.trial_number, rec.phase, rec.score, running_best))
        if rec.phase == "exploration" and rec.score > explore_best:
            explore_best = rec.score
        if rec.phase == "exploitation" and rec.score > exploit_best:
            exploit_best = rec.score

    # 打印每 N 个 trial 的摘要
    step = max(1, len(curve) // 20)
    print(f"{'trial':>5} {'phase':>12} {'score':>12} {'running_best':>14}")
    for i, (tn, phase, score, rb) in enumerate(curve):
        if i % step == 0 or i == len(curve) - 1:
            score_s = f"{score:.4f}" if score != float("-inf") else "    FAIL"
            print(f"{tn:>5} {phase:>12} {score_s:>12} {rb:>14.4f}")

    # 写 CSV
    csv_path = output_dir / f"convergence_seed{args.seed}_n{args.trials}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["trial", "phase", "score", "running_best"])
        for row in curve:
            tn, phase, score, rb = row
            w.writerow([tn, phase, "" if score == float("-inf") else score, rb])
    print(f"\n[写入] 收敛曲线 CSV: {csv_path}")

    # ---------- 汇总 ----------
    print("\n" + "=" * 70)
    print("收敛结果汇总")
    print("=" * 70)
    print(f"总耗时: {elapsed:.2f}s  ({elapsed / max(1, args.trials):.3f}s/trial 平均)")
    print(f"失败 trials: {failed}/{len(engine.history)}")
    print(f"探索阶段最佳: {explore_best:.4f}")
    print(f"利用阶段最佳: {exploit_best:.4f}")
    print(f"全局最佳: {best_value:.4f}")
    print(f"利用阶段相对探索的提升: "
          f"{((exploit_best - explore_best) / abs(explore_best) * 100):+.2f}%"
          if explore_best not in (float("-inf"), 0) else "N/A")

    if best_metrics:
        print(f"\n最佳回测指标:")
        print(f"  sharpe_ratio : {best_metrics.get('sharpe_ratio', 0):.4f}")
        print(f"  annual_return: {best_metrics.get('annual_return', 0):.2f}%")
        print(f"  max_drawdown : {best_metrics.get('max_drawdown', 0):.2f}%")
        print(f"  trades_count : {best_metrics.get('trades_count', 0)}")

    print(f"\n最佳参数:")
    for k, v in best_params.items():
        print(f"  {k:30s}: {v}")

    # ---------- 可选：画 PNG ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = [c[0] for c in curve]
        scores = [c[2] if c[2] != float("-inf") else None for c in curve]
        rbs = [c[3] for c in curve]
        explore_mask = [c[1] == "exploration" for c in curve]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(xs, rbs, label="running best", color="C0", linewidth=2)
        ax.scatter(
            [x for x, m in zip(xs, explore_mask) if m],
            [s for s, m in zip(scores, explore_mask) if m and s is not None],
            s=14, color="C2", alpha=0.6, label="explore trial",
        )
        ax.scatter(
            [x for x, m in zip(xs, explore_mask) if not m],
            [s for s, m in zip(scores, explore_mask) if (not m) and s is not None],
            s=14, color="C3", alpha=0.6, label="exploit trial",
        )
        ax.axvline(n_explore - 0.5, color="gray", linestyle="--", alpha=0.5,
                   label="explore→exploit")
        ax.set_xlabel("trial")
        ax.set_ylabel(f"score ({args.objective})")
        ax.set_title(f"Convergence  seed={args.seed}  "
                     f"trials={args.trials}  best={best_value:.4f}")
        ax.legend()
        ax.grid(alpha=0.3)
        png_path = output_dir / f"convergence_seed{args.seed}_n{args.trials}.png"
        fig.tight_layout()
        fig.savefig(png_path, dpi=120)
        print(f"[写入] 收敛曲线 PNG: {png_path}")
    except ImportError:
        print("[提示] 未安装 matplotlib，跳过 PNG 输出 (pip install matplotlib 启用)")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
