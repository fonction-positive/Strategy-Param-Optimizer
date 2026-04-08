# -*- coding: utf-8 -*-
"""
性能监控脚本
包装现有优化器，收集并报告性能指标

使用方法:
    python monitor_optimizer.py -d data/AG.csv -s strategy/Aberration.py
    python monitor_optimizer.py -d data/AG.csv -s strategy/Aberration.py --n-trials 50
"""

import os
import sys
import time
import argparse
import threading
import psutil
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Optimizer'))


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    start_time: float = 0
    end_time: float = 0
    total_time: float = 0
    n_trials: int = 0
    exploration_trials: int = 0
    exploitation_trials: int = 0

    # CPU 指标
    cpu_samples: list = field(default_factory=list)
    avg_cpu_usage: float = 0
    max_cpu_usage: float = 0

    # 内存指标
    memory_samples: list = field(default_factory=list)
    peak_memory_mb: float = 0
    avg_memory_mb: float = 0
    initial_memory_mb: float = 0

    # 并行效率
    n_workers: int = 1
    theoretical_speedup: float = 1.0
    actual_speedup: float = 1.0
    parallel_efficiency: float = 1.0

    # 串行基准时间（用于计算加速比）
    estimated_serial_time: float = 0
    avg_backtest_time: float = 0  # 单次回测的平均时间（从优化器获取）

    # 优化结果
    best_value: float = 0
    best_params: Dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self, sample_interval: float = 0.5):
        """
        初始化监控器

        Args:
            sample_interval: 采样间隔（秒）
        """
        self.sample_interval = sample_interval
        self.metrics = PerformanceMetrics()
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self.process = psutil.Process()

        # 记录CPU核心数
        self.cpu_count = psutil.cpu_count()

        # 预热系统CPU监控
        psutil.cpu_percent(percpu=False)

    def _get_total_memory_mb(self) -> float:
        """获取进程树的总内存使用（包括所有子进程）"""
        try:
            total_memory = self.process.memory_info().rss

            try:
                children = self.process.children(recursive=True)
                for child in children:
                    try:
                        total_memory += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

            return total_memory / (1024 * 1024)
        except Exception:
            return 0.0

    def _get_child_count(self) -> int:
        """获取子进程数量"""
        try:
            return len(self.process.children(recursive=True))
        except:
            return 0

    def start(self):
        """开始监控"""
        self.metrics = PerformanceMetrics()
        self.metrics.start_time = time.time()
        self.metrics.initial_memory_mb = self._get_total_memory_mb()

        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

        print(f"\n[监控] 性能监控已启动 (采样间隔: {self.sample_interval}s, CPU核心数: {self.cpu_count})")

    def stop(self):
        """停止监控"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)

        self.metrics.end_time = time.time()
        self.metrics.total_time = self.metrics.end_time - self.metrics.start_time

        # 计算统计值
        if self.metrics.cpu_samples:
            self.metrics.avg_cpu_usage = sum(self.metrics.cpu_samples) / len(self.metrics.cpu_samples)
            self.metrics.max_cpu_usage = max(self.metrics.cpu_samples)

        if self.metrics.memory_samples:
            self.metrics.avg_memory_mb = sum(self.metrics.memory_samples) / len(self.metrics.memory_samples)
            self.metrics.peak_memory_mb = max(self.metrics.memory_samples)

    def _monitor_loop(self):
        """监控循环"""
        sample_count = 0
        while self._monitoring:
            try:
                # 使用系统级CPU监控（更准确）
                # 返回所有核心的平均使用率 (0-100%)
                sys_cpu_avg = psutil.cpu_percent(percpu=False)
                # 转换为累计百分比（与活动监视器一致）
                # 例如：6核全满 = 6 * 100% = 600%
                cpu_percent = sys_cpu_avg * self.cpu_count / 100 * 100  # 实际就是 sys_cpu_avg * cpu_count

                self.metrics.cpu_samples.append(cpu_percent)

                # 内存使用（包括子进程）
                memory_mb = self._get_total_memory_mb()
                self.metrics.memory_samples.append(memory_mb)

                # 获取子进程数
                child_count = self._get_child_count()

                # 每20次采样打印一次
                sample_count += 1
                if sample_count % 20 == 0:
                    print(f"[监控 #{sample_count}] CPU: {cpu_percent:.0f}% ({sys_cpu_avg:.1f}% × {self.cpu_count}核), 内存: {memory_mb:.0f}MB, 子进程: {child_count}")

            except Exception:
                pass

            time.sleep(self.sample_interval)

    def set_trial_info(self, n_trials: int, exploration: int, exploitation: int, n_workers: int):
        """设置试验信息"""
        self.metrics.n_trials = n_trials
        self.metrics.exploration_trials = exploration
        self.metrics.exploitation_trials = exploitation
        self.metrics.n_workers = n_workers

    def set_result(self, best_value: float, best_params: Dict[str, Any]):
        """设置优化结果"""
        self.metrics.best_value = best_value
        self.metrics.best_params = best_params

    def calculate_efficiency(self, serial_time_estimate: Optional[float] = None, avg_backtest_time: Optional[float] = None):
        """
        计算并行效率

        Args:
            serial_time_estimate: 串行执行时间估计（可选）
            avg_backtest_time: 单次回测平均时间（可选，从优化器获取）
        """
        if avg_backtest_time:
            self.metrics.avg_backtest_time = avg_backtest_time

        if self.metrics.n_workers > 1:
            self.metrics.theoretical_speedup = self.metrics.n_workers

            if serial_time_estimate:
                # 使用外部提供的串行时间估计
                self.metrics.estimated_serial_time = serial_time_estimate
                self.metrics.actual_speedup = serial_time_estimate / self.metrics.total_time
            elif avg_backtest_time and avg_backtest_time > 0:
                # 使用单次回测时间估算串行总时间
                self.metrics.estimated_serial_time = avg_backtest_time * self.metrics.n_trials
                self.metrics.actual_speedup = self.metrics.estimated_serial_time / self.metrics.total_time
            else:
                # 无法准确估算，使用 CPU 利用率作为参考
                # 如果 CPU 平均利用率接近 n_workers * 100%，说明并行效率高
                if self.metrics.avg_cpu_usage > 0:
                    # 假设单核 100% 对应 1x 加速
                    # 如果 12 核平均 600%，说明约 6 个核在工作，加速比约 6x
                    effective_cores = self.metrics.avg_cpu_usage / 100.0
                    self.metrics.actual_speedup = max(1.0, effective_cores)
                else:
                    self.metrics.actual_speedup = 1.0

                # 粗略估算串行时间
                self.metrics.estimated_serial_time = self.metrics.total_time * self.metrics.actual_speedup

            self.metrics.parallel_efficiency = (
                self.metrics.actual_speedup / self.metrics.theoretical_speedup
            )

    def print_report(self):
        """打印性能报告"""
        m = self.metrics

        # 格式化时间
        if m.total_time >= 3600:
            time_str = f"{m.total_time/3600:.1f} 小时"
        elif m.total_time >= 60:
            time_str = f"{m.total_time/60:.1f} 分钟"
        else:
            time_str = f"{m.total_time:.1f} 秒"

        # 并行后的平均时间 vs 估算的串行单次时间
        parallel_avg = m.total_time / max(m.n_trials, 1)

        # 估算串行时间
        if m.estimated_serial_time > 0:
            if m.estimated_serial_time >= 3600:
                serial_time_str = f"{m.estimated_serial_time/3600:.1f} 小时"
            elif m.estimated_serial_time >= 60:
                serial_time_str = f"{m.estimated_serial_time/60:.1f} 分钟"
            else:
                serial_time_str = f"{m.estimated_serial_time:.1f} 秒"
        else:
            serial_time_str = "未知"

        # 单次回测时间
        if m.avg_backtest_time > 0:
            backtest_time_str = f"{m.avg_backtest_time:.2f} 秒"
        else:
            backtest_time_str = "未知"

        print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                       性能监控报告                                ║
╠══════════════════════════════════════════════════════════════════╣
║  时间统计                                                         ║
║  ────────────────────────────────────────────────────────────    ║
║  总耗时 (并行后):     {time_str:<40} ║
║  估算串行耗时:        {serial_time_str:<40} ║
║  总试验次数:          {m.n_trials:<40} ║
║  单次回测时间:        {backtest_time_str:<40} ║
║  探索阶段:            {m.exploration_trials} trials{'':<33} ║
║  利用阶段:            {m.exploitation_trials} trials{'':<33} ║
╠══════════════════════════════════════════════════════════════════╣
║  资源使用                                                         ║
║  ────────────────────────────────────────────────────────────    ║
║  CPU 平均使用率:      {m.avg_cpu_usage:.1f}%{'':<37} ║
║  CPU 峰值使用率:      {m.max_cpu_usage:.1f}%{'':<37} ║
║  内存初始值:          {m.initial_memory_mb:.1f} MB{'':<34} ║
║  内存平均值:          {m.avg_memory_mb:.1f} MB{'':<34} ║
║  内存峰值:            {m.peak_memory_mb:.1f} MB{'':<34} ║
╠══════════════════════════════════════════════════════════════════╣
║  并行效率                                                         ║
║  ────────────────────────────────────────────────────────────    ║
║  并行进程数:          {m.n_workers:<40} ║
║  理论加速比:          {m.theoretical_speedup:.1f}x{'':<37} ║
║  实际加速比:          {m.actual_speedup:.1f}x{'':<38} ║
║  并行效率:            {m.parallel_efficiency:.1%}{'':<37} ║
╠══════════════════════════════════════════════════════════════════╣
║  优化结果                                                         ║
║  ────────────────────────────────────────────────────────────    ║
║  最优目标值:          {m.best_value:.4f}{'':<34} ║
╚══════════════════════════════════════════════════════════════════╝
""")

        # 打印最优参数
        if m.best_params:
            print("  最优参数:")
            for k, v in m.best_params.items():
                if isinstance(v, float):
                    print(f"    • {k}: {v:.4f}")
                else:
                    print(f"    • {k}: {v}")
            print()

    def save_report(self, filepath: str):
        """保存报告到文件"""
        import json

        m = self.metrics
        report = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "time_metrics": {
                "total_time_seconds": m.total_time,
                "estimated_serial_time_seconds": m.estimated_serial_time,
                "avg_backtest_time_seconds": m.avg_backtest_time,
                "n_trials": m.n_trials,
                "exploration_trials": m.exploration_trials,
                "exploitation_trials": m.exploitation_trials,
            },
            "resource_metrics": {
                "avg_cpu_usage_percent": m.avg_cpu_usage,
                "max_cpu_usage_percent": m.max_cpu_usage,
                "initial_memory_mb": m.initial_memory_mb,
                "avg_memory_mb": m.avg_memory_mb,
                "peak_memory_mb": m.peak_memory_mb,
            },
            "parallel_metrics": {
                "n_workers": m.n_workers,
                "theoretical_speedup": m.theoretical_speedup,
                "actual_speedup": m.actual_speedup,
                "parallel_efficiency": m.parallel_efficiency,
            },
            "optimization_result": {
                "best_value": m.best_value,
                "best_params": m.best_params,
            }
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"[监控] 性能报告已保存至: {filepath}")


def run_with_monitoring(
    data_path: str,
    strategy_path: str,
    objective: str = "sharpe_ratio",
    n_trials: int = 50,
    asset_type: str = "stock",
    contract_code: str = None,
    output_dir: str = "./optimization_results",
    save_report: bool = True,
    # 并行优化参数
    enable_parallel: bool = True,
    n_workers: int = -1,
    # 批量并行优化参数
    enable_batch_parallel: bool = True,
    batch_size: int = 8,
    adaptive_batch: bool = True,
    hybrid_mode: bool = True,
    parallel_ratio: float = 0.7,
    # 增强采样器参数
    use_enhanced_sampler: bool = True,
    enable_dynamic_trials: bool = True,
    auto_expand_boundary: bool = True,
    max_boundary_rounds: int = 2
):
    """
    带性能监控的优化运行

    Args:
        data_path: 数据文件路径
        strategy_path: 策略文件路径
        objective: 优化目标
        n_trials: 试验次数
        asset_type: 资产类型
        contract_code: 合约代码（期货）
        output_dir: 输出目录
        save_report: 是否保存报告
        enable_parallel: 是否启用并行优化
        n_workers: 并行工作进程数
        enable_batch_parallel: 是否启用批量并行优化
        batch_size: 批量大小
        adaptive_batch: 是否启用自适应批量大小
        hybrid_mode: 是否启用混合模式
        parallel_ratio: 批量并行阶段占比
        use_enhanced_sampler: 是否使用增强采样器
        enable_dynamic_trials: 是否启用动态试验次数
        auto_expand_boundary: 是否启用边界二次搜索
        max_boundary_rounds: 边界二次搜索最大轮数
    """
    import multiprocessing
    from pathlib import Path

    # 导入优化器
    from Optimizer.universal_optimizer import UniversalOptimizer
    from Optimizer.config import ParallelConfig, BatchParallelConfig
    from Optimizer.futures_config import build_broker_config

    # 创建监控器
    monitor = PerformanceMonitor(sample_interval=0.5)

    # 计算实际工作进程数
    actual_workers = n_workers if n_workers != -1 else multiprocessing.cpu_count()

    print(f"\n{'='*60}")
    print(f"  性能监控优化器")
    print(f"{'='*60}")
    print(f"  数据: {data_path}")
    print(f"  策略: {strategy_path}")
    print(f"  目标: {objective}")
    print(f"  试验次数: {n_trials}")
    print(f"{'='*60}")
    print(f"  并行配置:")
    print(f"    • 并行优化: {'启用' if enable_parallel else '禁用'}")
    if enable_parallel:
        print(f"    • 工作进程: {actual_workers}")
    print(f"    • 批量并行: {'启用' if enable_batch_parallel else '禁用'}")
    if enable_batch_parallel:
        print(f"    • 批量大小: {batch_size}")
        print(f"    • 自适应批量: {'启用' if adaptive_batch else '禁用'}")
        print(f"    • 混合模式: {'启用' if hybrid_mode else '禁用'}")
        if hybrid_mode:
            print(f"    • 并行比例: {parallel_ratio:.0%}")
    print(f"{'='*60}")
    print(f"  采样配置:")
    print(f"    • 增强采样器: {'启用' if use_enhanced_sampler else '禁用'}")
    print(f"    • 动态试验: {'启用' if enable_dynamic_trials else '禁用'}")
    print(f"    • 边界搜索: {'启用' if auto_expand_boundary else '禁用'}")
    if auto_expand_boundary:
        print(f"    • 最大轮数: {max_boundary_rounds}")
    print(f"{'='*60}\n")

    # 配置
    parallel_config = ParallelConfig(
        enable_parallel=enable_parallel,
        n_workers=n_workers
    )
    batch_config = BatchParallelConfig(
        enable_batch_parallel=enable_batch_parallel,
        batch_size=batch_size,
        adaptive_batch=adaptive_batch,
        hybrid_mode=hybrid_mode,
        parallel_ratio=parallel_ratio
    )

    # 期货配置
    broker_config = None
    if asset_type == "futures" and contract_code:
        broker_config = build_broker_config(contract_code=contract_code)

    # 创建优化器
    optimizer = UniversalOptimizer(
        data_path=data_path,
        strategy_path=strategy_path,
        objective=objective,
        use_llm=False,
        output_dir=output_dir,
        verbose=True,
        broker_config=broker_config,
        parallel_config=parallel_config,
        batch_parallel_config=batch_config
    )

    # 获取并行配置
    actual_n_workers = parallel_config.n_workers
    if actual_n_workers == -1:
        actual_n_workers = multiprocessing.cpu_count()

    # ============ 测量单次回测基准时间 ============
    print(f"\n[性能监控] 测量单次回测基准时间...")
    avg_backtest_time = None
    try:
        # 获取策略默认参数
        default_params = {}
        for param in optimizer.strategy_info['params']:
            default_params[param.name] = param.default_value

        # 运行3次回测取平均值
        benchmark_times = []
        for i in range(3):
            bench_start = time.time()
            _ = optimizer.backtest_engine.run_backtest(
                optimizer.strategy_class,
                optimizer.data,
                default_params
            )
            bench_end = time.time()
            benchmark_times.append(bench_end - bench_start)

        avg_backtest_time = sum(benchmark_times) / len(benchmark_times)
        print(f"[性能监控] 单次回测时间: {avg_backtest_time:.3f} 秒 (3次平均)")
        print(f"[性能监控] 估算串行总时间: {avg_backtest_time * n_trials:.1f} 秒 = {avg_backtest_time * n_trials / 60:.1f} 分钟\n")
    except Exception as e:
        print(f"[性能监控] 无法测量基准时间: {e}\n")

    # 开始监控
    monitor.start()

    try:
        # 执行优化
        result = optimizer.optimize(
            n_trials=n_trials,
            use_enhanced_sampler=use_enhanced_sampler,
            enable_dynamic_trials=enable_dynamic_trials,
            auto_expand_boundary=auto_expand_boundary,
            max_expansion_rounds=max_boundary_rounds
        )

        # 设置结果
        if result:
            monitor.set_result(
                best_value=result.get("performance_metrics", {}).get("sharpe_ratio", 0),
                best_params=result.get("best_parameters", {})
            )

            # 获取试验信息
            opt_info = result.get("optimization_info", {})
            exploration = opt_info.get("exploration_trials", int(n_trials * 0.3))
            exploitation = opt_info.get("exploitation_trials", n_trials - exploration)
            actual_trials = opt_info.get("total_trials", n_trials)

            monitor.set_trial_info(
                n_trials=actual_trials,
                exploration=exploration,
                exploitation=exploitation,
                n_workers=actual_n_workers
            )

    except Exception as e:
        print(f"\n❌ 优化过程出错: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 停止监控
        monitor.stop()

        # 计算并行效率（传入单次回测时间）
        monitor.calculate_efficiency(avg_backtest_time=avg_backtest_time)

        # 打印报告
        monitor.print_report()

        # 保存报告
        if save_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            asset_name = Path(data_path).stem.replace('_processed', '')
            report_path = os.path.join(output_dir, f"performance_report_{asset_name}_{timestamp}.json")
            monitor.save_report(report_path)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="性能监控优化器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python monitor_optimizer.py -d data/AG.csv -s strategy/Aberration.py

  # 指定试验次数
  python monitor_optimizer.py -d data/AG.csv -s strategy/Aberration.py --n-trials 100

  # 期货优化
  python monitor_optimizer.py -d data/AG.csv -s strategy/Aberration.py --asset-type futures --contract-code AG

  # 禁用混合模式（全并行）
  python monitor_optimizer.py -d data/AG.csv -s strategy/Aberration.py --no-hybrid-mode

  # 禁用并行优化（串行模式）
  python monitor_optimizer.py -d data/AG.csv -s strategy/Aberration.py --no-parallel

  # 自定义并行配置
  python monitor_optimizer.py -d data/AG.csv -s strategy/Aberration.py --n-workers 4 --batch-size 4
        """
    )

    # 基本参数
    parser.add_argument("-d", "--data", required=True, help="数据文件路径")
    parser.add_argument("-s", "--strategy", required=True, help="策略文件路径")
    parser.add_argument("-o", "--objective", default="sharpe_ratio",
                        choices=["sharpe_ratio", "annual_return", "max_drawdown", "sortino_ratio", "calmar_ratio", "market_maker_score"],
                        help="优化目标 (默认: sharpe_ratio)")
    parser.add_argument("--n-trials", type=int, default=50, help="试验次数 (默认: 50)")
    parser.add_argument("--asset-type", default="stock", choices=["stock", "futures"],
                        help="资产类型 (默认: stock)")
    parser.add_argument("--contract-code", help="期货合约代码 (如 AG, AU, IF)")
    parser.add_argument("--output-dir", default="./optimization_results", help="输出目录")
    parser.add_argument("--no-save", action="store_true", help="不保存性能报告")

    # 并行优化参数
    parser.add_argument("--no-parallel", action="store_true",
                        help="禁用并行优化（默认启用）")
    parser.add_argument("--n-workers", type=int, default=-1,
                        help="并行工作进程数（默认: -1 表示自动检测CPU核心数）")

    # 批量并行优化参数
    parser.add_argument("--no-batch-parallel", action="store_true",
                        help="禁用批量并行优化（回到传统串行模式）")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="批量大小（默认: 8）")
    parser.add_argument("--no-adaptive-batch", action="store_true",
                        help="禁用自适应批量大小")
    parser.add_argument("--no-hybrid-mode", action="store_true",
                        help="禁用混合模式（不使用串行精细搜索）")
    parser.add_argument("--parallel-ratio", type=float, default=0.7,
                        help="批量并行阶段占比（默认: 0.7，剩余用串行精细搜索）")

    # 增强采样器参数
    parser.add_argument("--no-enhanced-sampler", action="store_true",
                        help="禁用增强采样器（正态分布采样），使用传统均匀采样")
    parser.add_argument("--no-dynamic-trials", action="store_true",
                        help="禁用动态试验次数，使用用户指定的固定值")
    parser.add_argument("--no-boundary-search", action="store_true",
                        help="禁用边界二次搜索")
    parser.add_argument("--max-boundary-rounds", type=int, default=2,
                        help="边界二次搜索最大轮数（默认: 2）")

    args = parser.parse_args()

    run_with_monitoring(
        data_path=args.data,
        strategy_path=args.strategy,
        objective=args.objective,
        n_trials=args.n_trials,
        asset_type=args.asset_type,
        contract_code=args.contract_code,
        output_dir=args.output_dir,
        save_report=not args.no_save,
        # 并行优化参数
        enable_parallel=not args.no_parallel,
        n_workers=args.n_workers,
        # 批量并行优化参数
        enable_batch_parallel=not args.no_batch_parallel,
        batch_size=args.batch_size,
        adaptive_batch=not args.no_adaptive_batch,
        hybrid_mode=not args.no_hybrid_mode,
        parallel_ratio=args.parallel_ratio,
        # 增强采样器参数
        use_enhanced_sampler=not args.no_enhanced_sampler,
        enable_dynamic_trials=not args.no_dynamic_trials,
        auto_expand_boundary=not args.no_boundary_search,
        max_boundary_rounds=args.max_boundary_rounds
    )


if __name__ == "__main__":
    main()
