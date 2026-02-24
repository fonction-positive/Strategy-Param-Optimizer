# -*- coding: utf-8 -*-
"""
SPP (System Parameter Permutation) 鲁棒性分析模块 — v3.0

以贝叶斯优化找到的最优参数为中心，在邻域内进行蒙特卡洛采样，
评估参数鲁棒性。可选 LLM 辅助识别敏感参数，只扰动敏感参数以减少计算量。
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.font_manager as _fm

# 按优先级查找可用中文字体文件
_CN_FONT_CANDIDATES = [
    '/System/Library/Fonts/STHeiti Medium.ttc',
    '/System/Library/Fonts/STHeiti Light.ttc',
    '/Library/Fonts/Arial Unicode.ttf',
    '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
]
_cn_font_path = next((p for p in _CN_FONT_CANDIDATES if os.path.exists(p)), None)
if _cn_font_path:
    _fm.fontManager.addfont(_cn_font_path)
    _cn_font_name = _fm.FontProperties(fname=_cn_font_path).get_name()
    plt.rcParams['font.sans-serif'] = [_cn_font_name, 'DejaVu Sans']
else:
    plt.rcParams['font.sans-serif'] = ['PingFang HK', 'STHeiti', 'Heiti TC', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest_engine import BacktestEngine, BacktestResult
from config import StrategyParam


@dataclass
class SPPConfig:
    """SPP 分析配置"""
    n_samples: int = 300
    perturbation_ratio: float = 0.20
    objective: str = 'sharpe_ratio'
    use_llm: bool = False
    llm_config: Optional[Any] = None
    sensitive_params: Optional[List[str]] = None


class SPPAnalyzer:
    """SPP 鲁棒性分析器 (v3.0) — 最优参数邻域蒙特卡洛采样"""

    def __init__(self, backtest_engine, strategy_class, data, search_space,
                 config=None, verbose=True, llm_client=None):
        self.engine = backtest_engine
        self.strategy_class = strategy_class
        self.data = data
        self.search_space = search_space
        self.config = config or SPPConfig()
        self.verbose = verbose
        self.llm_client = llm_client

    # ---- 敏感参数识别 ----

    def _identify_sensitive_params(self, best_params, best_metrics):
        """优先级: 手动指定 > LLM > 全部参数。返回 (list, method, reasoning)"""
        if self.config.sensitive_params:
            valid = [p for p in self.config.sensitive_params if p in self.search_space]
            if valid:
                if self.verbose:
                    print(f"  [敏感参数] 手动指定: {valid}")
                return valid, 'manual', ''
        if self.config.use_llm and self.llm_client is not None:
            try:
                params, reasoning = self._llm_extract_sensitive_params(best_params, best_metrics)
                if params:
                    if self.verbose:
                        print(f"  [敏感参数] LLM 识别: {params}")
                        if reasoning:
                            print(f"  [LLM 理由] {reasoning[:120]}...")
                    return params, 'llm', reasoning
            except Exception as e:
                if self.verbose:
                    print(f"  [警告] LLM 失败: {e}，fallback 到全部参数")
        all_p = list(self.search_space.keys())
        if self.verbose:
            print(f"  [敏感参数] 使用全部参数: {all_p}")
        return all_p, 'all', ''

    def _llm_extract_sensitive_params(self, best_params, best_metrics):
        """用 LLM 识别敏感参数。返回 (param_list, reasoning)"""
        sys_prompt = (
            '你是一个量化策略参数分析专家。给定策略的参数列表、最优值和搜索范围，'
            '判断哪些参数是"敏感参数"——即微小变化会显著影响策略表现的参数。'
            '通常与信号生成直接相关的参数（如周期、阈值）是敏感的，'
            '而与风控、仓位管理相关的参数相对不敏感。'
            '请返回 JSON: {"sensitive_params": ["p1","p2"], "reasoning": "..."}'
        )
        param_info = []
        for name, sp in self.search_space.items():
            param_info.append({
                'name': name, 'type': sp.param_type,
                'best_value': best_params.get(name, sp.default_value),
                'default_value': sp.default_value,
                'range': [sp.min_value, sp.max_value], 'step': sp.step,
            })
        mc = {k: round(v, 4) if isinstance(v, float) else v for k, v in best_metrics.items()}
        user_prompt = (
            f"策略名称: {self.strategy_class.__name__}\n"
            f"最优参数性能: {json.dumps(mc, ensure_ascii=False)}\n"
            f"参数列表:\n{json.dumps(param_info, indent=2, ensure_ascii=False)}\n"
            f"请分析哪些是敏感参数，返回 JSON。"
        )
        resp = self.llm_client.client.call_llm([
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ])
        if not resp:
            return [], ''
        js, je = resp.find('{'), resp.rfind('}') + 1
        if js == -1 or je <= js:
            return [], ''
        r = json.loads(resp[js:je])
        return ([p for p in r.get('sensitive_params', []) if p in self.search_space],
                r.get('reasoning', ''))

    # ---- 蒙特卡洛采样 ----

    def _generate_mc_samples(self, best_params, sensitive_params):
        """对敏感参数均匀扰动采样，非敏感参数固定为最优值"""
        rng = np.random.default_rng()
        ratio = self.config.perturbation_ratio
        samples = []
        for _ in range(self.config.n_samples):
            params = {}
            for name, sp in self.search_space.items():
                best_val = best_params.get(name, sp.default_value)
                if name not in sensitive_params:
                    params[name] = best_val
                    continue
                if sp.param_type == 'int':
                    int_ratio = ratio * 0.75  # int 用稍小扰动
                    lo = max(sp.min_value, best_val * (1 - int_ratio))
                    hi = min(sp.max_value, best_val * (1 + int_ratio))
                    if best_val == 0:
                        lo, hi = sp.min_value, sp.max_value
                    step = int(sp.step) if sp.step and sp.step >= 1 else 1
                    lo_i, hi_i = int(np.ceil(lo)), int(np.floor(hi))
                    possible = list(range(lo_i, hi_i + 1, step))
                    if not possible:
                        possible = [int(best_val)]
                    params[name] = int(rng.choice(possible))
                else:
                    lo = max(sp.min_value, best_val * (1 - ratio))
                    hi = min(sp.max_value, best_val * (1 + ratio))
                    if best_val == 0:
                        lo, hi = sp.min_value, sp.max_value
                    if lo >= hi:
                        lo, hi = sp.min_value, sp.max_value
                    if lo >= hi:
                        params[name] = float(best_val)
                        continue
                    val = rng.uniform(lo, hi)
                    if sp.step:
                        val = lo + round((val - lo) / sp.step) * sp.step
                        val = np.clip(val, lo, hi)
                    params[name] = float(round(val, 6))
            samples.append(params)
        return samples

    # ---- 批量回测 ----

    def _evaluate_batch(self, param_list, desc=""):
        """批量回测，收集完整结果到 DataFrame"""
        records = []
        total = len(param_list)
        start = time.time()
        for i, params in enumerate(param_list):
            result = self.engine.run_backtest(
                self.strategy_class, self.data, params, calculate_yearly=False
            )
            if result is not None:
                obj_val = self.engine.evaluate_objective(result, self.config.objective)
                records.append({
                    'params': params,
                    self.config.objective: obj_val,
                    'annual_return': result.annual_return,
                    'max_drawdown': result.max_drawdown,
                    'sharpe_ratio': result.sharpe_ratio,
                    'sortino_ratio': result.sortino_ratio,
                    'trades_count': result.trades_count,
                })
            if self.verbose and (i + 1) % max(1, total // 10) == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                remaining = (total - i - 1) / rate if rate > 0 else 0
                print(f"  [{desc}] {i+1}/{total} ({100*(i+1)/total:.0f}%) 剩余 {remaining:.0f}s")
        return pd.DataFrame(records)

    # ---- 蒙特卡洛分析 (核心) ----

    def run_monte_carlo_analysis(self, best_params, best_metrics):
        """
        以最优参数为中心的蒙特卡洛鲁棒性分析。
        返回 (DataFrame, sensitive_params, method, reasoning)
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[SPP] 蒙特卡洛鲁棒性分析 ({self.config.n_samples} 组)")
            print(f"{'='*60}")

        sensitive_params, method, reasoning = self._identify_sensitive_params(best_params, best_metrics)
        samples = self._generate_mc_samples(best_params, sensitive_params)

        if self.verbose:
            print(f"  扰动比例: ±{self.config.perturbation_ratio:.0%}")
            print(f"  敏感参数 ({len(sensitive_params)}): {sensitive_params}")
            fixed = [n for n in self.search_space if n not in sensitive_params]
            if fixed:
                print(f"  固定参数 ({len(fixed)}): {fixed}")

        df = self._evaluate_batch(samples, desc="蒙特卡洛")

        if self.verbose and len(df) > 0:
            obj = self.config.objective
            print(f"  有效样本: {len(df)}/{self.config.n_samples}")
            print(f"  {obj} 中位数: {df[obj].median():.4f}")
            print(f"  {obj} 均值:   {df[obj].mean():.4f}")
            print(f"  {obj} 标准差: {df[obj].std():.4f}")

        return df, sensitive_params, method, reasoning

    # ---- 参数相关性 ----

    def _compute_param_correlations(self, df, sensitive_params):
        """计算各敏感参数与目标指标的 Pearson 相关系数"""
        obj = self.config.objective
        if len(df) < 5 or obj not in df.columns:
            return {}
        corrs = {}
        for pname in sensitive_params:
            vals = df['params'].apply(lambda d: d.get(pname))
            if vals.nunique() < 2:
                continue
            corrs[pname] = round(float(vals.astype(float).corr(df[obj])), 4)
        return corrs

    # ---- 综合判定 ----

    def _compute_verdict(self, mc_df, best_metrics, sensitive_params):
        """计算综合判定"""
        obj = self.config.objective
        best_obj = best_metrics.get(obj, 0)
        verdict = {'parameter_robust': '未知', 'stability_score': 0,
                   'sensitive_param_count': len(sensitive_params), 'summary': ''}

        if len(mc_df) > 0:
            mc_median = mc_df[obj].median()
            decay = (best_obj - mc_median) / abs(best_obj) if best_obj != 0 else 0
            stability = mc_median / best_obj if best_obj != 0 else 0

            if abs(decay) < 0.15:
                verdict['parameter_robust'] = '强 (衰减<15%)'
            elif abs(decay) < 0.30:
                verdict['parameter_robust'] = '中 (衰减15-30%)'
            else:
                verdict['parameter_robust'] = '弱 (衰减>30%)'

            verdict['stability_score'] = round(float(stability), 4)

            parts = []
            if abs(decay) >= 0.30:
                parts.append('参数敏感性高，建议扩大扰动范围验证')
            if abs(decay) < 0.15:
                parts.append('参数鲁棒性强，局部扰动对策略表现影响较小')
            elif abs(decay) < 0.30:
                parts.append('参数鲁棒性中等，建议关注敏感参数的取值')
            verdict['summary'] = '；'.join(parts) if parts else ''

        return verdict

    # ---- 可视化报告 (2x2) ----

    def generate_report(self, mc_df, best_params, best_metrics,
                        sensitive_params, param_corrs, output_path):
        """生成 2x2 PNG 报告"""
        obj = self.config.objective
        best_obj = best_metrics.get(obj, 0)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'SPP 蒙特卡洛鲁棒性分析 — {self.strategy_class.__name__}',
                     fontsize=14, fontweight='bold')

        # (0,0) MC 分布直方图 + KDE
        ax = axes[0, 0]
        if len(mc_df) > 0:
            vals = mc_df[obj].dropna()
            ax.hist(vals, bins=40, density=True, alpha=0.6,
                    color='seagreen', edgecolor='white')
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(vals)
                x_range = np.linspace(vals.min(), vals.max(), 200)
                ax.plot(x_range, kde(x_range), 'g-', lw=2, label='KDE')
            except ImportError:
                pass
            median_val = vals.median()
            mean_val = vals.mean()
            ax.axvline(best_obj, color='red', ls='-', lw=2,
                       label=f'最优={best_obj:.3f}')
            ax.axvline(median_val, color='orange', ls='--', lw=2,
                       label=f'中位数={median_val:.3f}')
            decay = (best_obj - median_val) / abs(best_obj) if best_obj != 0 else 0
            ax.text(0.05, 0.95, f'衰减率={decay:.1%}',
                    transform=ax.transAxes, fontsize=11, va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            ax.legend(fontsize=9)
        ax.set_title(f'蒙特卡洛 {obj} 分布 (n={len(mc_df)}, '
                     f'扰动={self.config.perturbation_ratio:.0%})', fontsize=12)
        ax.set_xlabel(obj)
        ax.set_ylabel('密度')

        # (0,1) 参数敏感度柱状图
        ax = axes[0, 1]
        if param_corrs:
            names = list(param_corrs.keys())
            values = [param_corrs[n] for n in names]
            colors = ['#e74c3c' if abs(v) > 0.3 else '#3498db' for v in values]
            bars = ax.barh(names, values, color=colors, edgecolor='white')
            ax.axvline(0, color='gray', ls='-', lw=0.8)
            ax.axvline(0.3, color='red', ls=':', lw=1, alpha=0.5)
            ax.axvline(-0.3, color='red', ls=':', lw=1, alpha=0.5)
            for bar, v in zip(bars, values):
                ax.text(v + 0.02 if v >= 0 else v - 0.02, bar.get_y() + bar.get_height()/2,
                        f'{v:.2f}', va='center', ha='left' if v >= 0 else 'right', fontsize=9)
        ax.set_title(f'参数敏感度 (与{obj}的相关系数)', fontsize=12)
        ax.set_xlabel('Pearson 相关系数')

        # (1,0) 风险-收益散点图
        ax = axes[1, 0]
        if len(mc_df) > 0 and 'max_drawdown' in mc_df.columns:
            ax.scatter(mc_df['max_drawdown'], mc_df['annual_return'],
                       alpha=0.4, s=15, c='seagreen')
            best_dd = best_metrics.get('max_drawdown', 0)
            best_ar = best_metrics.get('annual_return', 0)
            ax.scatter([best_dd], [best_ar], c='red', s=120, zorder=5,
                       marker='*', label='最优参数')
            ax.legend(fontsize=9)
        ax.set_title('风险-收益散点图 (MC采样)', fontsize=12)
        ax.set_xlabel('最大回撤 (%)')
        ax.set_ylabel('年化收益 (%)')

        # (1,1) 文字总结面板
        ax = axes[1, 1]
        ax.axis('off')
        summary_lines = self._build_text_summary(
            mc_df, best_params, best_metrics, sensitive_params, param_corrs)
        ax.text(0.05, 0.95, '\n'.join(summary_lines),
                transform=ax.transAxes, fontsize=10, va='top', ha='left',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        if self.verbose:
            print(f"\n[SPP] 报告已保存: {output_path}")
        return output_path

    def _build_text_summary(self, mc_df, best_params, best_metrics,
                            sensitive_params, param_corrs):
        """构建文字总结面板内容"""
        obj = self.config.objective
        best_obj = best_metrics.get(obj, 0)
        lines = ['═══ SPP 鲁棒性总结 ═══', '']

        # 最优参数
        lines.append('▸ 最优参数:')
        for k, v in best_params.items():
            marker = ' *' if k in sensitive_params else ''
            lines.append(f'    {k} = {v}{marker}')
        lines.append(f'▸ 最优 {obj}: {best_obj:.4f}')
        lines.append('')

        # MC 统计
        if len(mc_df) > 0:
            vals = mc_df[obj]
            mc_median = vals.median()
            decay = (best_obj - mc_median) / abs(best_obj) if best_obj != 0 else 0
            robust_score = max(0, 1 - abs(decay)) * 100
            lines += [
                f'▸ MC中位数:      {mc_median:.4f}',
                f'▸ MC均值:        {vals.mean():.4f}',
                f'▸ MC标准差:      {vals.std():.4f}',
                f'▸ 衰减率:        {decay:.1%}',
                f'▸ 鲁棒性评分:    {robust_score:.0f}/100',
                '',
            ]

        # 敏感参数
        lines.append(f'▸ 敏感参数 ({len(sensitive_params)}个):')
        for p in sensitive_params:
            corr = param_corrs.get(p, 0)
            lines.append(f'    {p}  (corr={corr:+.2f})')
        lines.append('')

        # 综合判定
        verdict = self._compute_verdict(mc_df, best_metrics, sensitive_params)
        lines += [
            '═══ 综合判定 ═══',
            f'参数鲁棒:   {verdict["parameter_robust"]}',
            f'稳定性得分: {verdict["stability_score"]:.2f}',
            f'{verdict["summary"]}',
        ]
        return lines

    # ---- 完整分析流程 ----

    def run_full_analysis(self, best_params, best_metrics, output_dir,
                          asset_name='ASSET', strategy_name='Strategy',
                          source_json=''):
        """运行完整 SPP 分析并输出 JSON + PNG"""
        start_time = time.time()
        os.makedirs(output_dir, exist_ok=True)

        if self.verbose:
            print(f"\n{'#'*60}")
            print(f"# SPP 蒙特卡洛鲁棒性分析 (v3.0)")
            print(f"# 标的: {asset_name}  策略: {strategy_name}")
            print(f"# 目标: {self.config.objective}")
            print(f"# 采样: {self.config.n_samples} 次, 扰动: ±{self.config.perturbation_ratio:.0%}")
            print(f"{'#'*60}")

        # 蒙特卡洛分析
        mc_df, sensitive_params, sp_method, sp_reasoning = \
            self.run_monte_carlo_analysis(best_params, best_metrics)

        elapsed = time.time() - start_time

        # 参数相关性
        param_corrs = self._compute_param_correlations(mc_df, sensitive_params)

        # 生成 PNG 报告
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        png_path = os.path.join(output_dir,
                                f'spp_report_{asset_name}_{timestamp}.png')
        self.generate_report(mc_df, best_params, best_metrics,
                             sensitive_params, param_corrs, png_path)

        # 构建 JSON 结果
        obj = self.config.objective
        best_obj = best_metrics.get(obj, 0)

        result = {
            'spp_info': {
                'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'elapsed_seconds': round(elapsed, 1),
                'source_json': source_json,
                'asset': asset_name,
                'strategy': strategy_name,
                'config': {
                    'n_samples': self.config.n_samples,
                    'perturbation_ratio': self.config.perturbation_ratio,
                    'objective': self.config.objective,
                    'use_llm': self.config.use_llm,
                },
            },
            'best_parameters': best_params,
            'best_metrics': best_metrics,
        }

        # 敏感参数信息
        result['sensitive_params'] = {
            'method': sp_method,
            'params': sensitive_params,
            'llm_reasoning': sp_reasoning,
        }

        # MC 统计
        if len(mc_df) > 0:
            vals = mc_df[obj]
            mc_median = float(vals.median())
            decay = (best_obj - mc_median) / abs(best_obj) if best_obj != 0 else 0
            result['monte_carlo_stability'] = {
                'sample_count': len(mc_df),
                'perturbation_ratio': self.config.perturbation_ratio,
                'median': round(mc_median, 4),
                'mean': round(float(vals.mean()), 4),
                'std': round(float(vals.std()), 4),
                'p5': round(float(vals.quantile(0.05)), 4),
                'p25': round(float(vals.quantile(0.25)), 4),
                'p75': round(float(vals.quantile(0.75)), 4),
                'p95': round(float(vals.quantile(0.95)), 4),
                'decay_rate': round(float(decay), 4),
                'robustness_score': round(max(0, 1 - abs(decay)) * 100, 1),
                'param_correlations': param_corrs,
            }
        else:
            result['monte_carlo_stability'] = {}

        # 综合判定
        result['verdict'] = self._compute_verdict(
            mc_df, best_metrics, sensitive_params)

        # 保存 JSON
        json_path = os.path.join(output_dir,
                                 f'spp_result_{asset_name}_{timestamp}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[SPP] 分析完成! 耗时 {elapsed:.1f}s")
            print(f"[SPP] JSON: {json_path}")
            print(f"[SPP] PNG:  {png_path}")
            print(f"{'='*60}")
            v = result['verdict']
            print(f"\n  参数鲁棒:   {v['parameter_robust']}")
            print(f"  稳定性得分: {v['stability_score']}")
            if v['summary']:
                print(f"  总结: {v['summary']}")

        return result
