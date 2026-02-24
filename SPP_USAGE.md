# SPP 蒙特卡洛鲁棒性分析使用指南 (v3.0)

## 概述

SPP (System Parameter Permutation) 鲁棒性分析工具，以贝叶斯优化找到的最优参数为中心，在参数邻域内进行蒙特卡洛采样，评估参数鲁棒性。

**核心思想**: 对最优参数施加随机扰动（默认 ±20%），观察策略表现的衰减程度。衰减越小，参数越鲁棒。

**v3.0 改进**:
- 单一维度分析（蒙特卡洛邻域采样），替代原有三维度分析，大幅减少计算量
- 支持 LLM 辅助识别敏感参数，只扰动敏感参数
- 参数敏感度分析（相关系数）
- 2×2 可视化报告

## 敏感参数识别

分析器支持三级优先级识别敏感参数：

1. **手动指定** (`--sensitive-params`) — 最高优先级，直接指定要扰动的参数
2. **LLM 分析** (`--use-llm`) — 让 LLM 根据策略类型和参数含义判断哪些参数敏感
3. **全部参数** (默认) — 扰动所有参数

非敏感参数在采样时固定为最优值，只有敏感参数被扰动，从而减少噪声和计算量。

## 基本用法

```bash
# 标准分析（全部参数扰动）
python run_spp_analysis.py \
  -r optimization_results/AG/optimization_AG_Aberration_20260202.json \
  -d project_trend/data/AG.csv \
  -s project_trend/src/Aberration.py

# 手动指定敏感参数
python run_spp_analysis.py \
  -r result.json -d data.csv -s strategy.py \
  --sensitive-params period,std_dev_upper

# 使用 LLM 识别敏感参数
python run_spp_analysis.py \
  -r result.json -d data.csv -s strategy.py \
  --use-llm --llm-model xuanyuan

# 快速测试（少量采样）
python run_spp_analysis.py \
  -r result.json -d data.csv -s strategy.py \
  --samples 50
```

## 可选参数

| 参数 | 短选项 | 默认值 | 说明 |
|------|--------|--------|------|
| `--samples` | `-n` | 300 | 蒙特卡洛采样次数 |
| `--perturbation` | `-p` | 0.20 | 扰动比例（±20%） |
| `--objective` | `-o` | 从JSON读取 | 分析指标 |
| `--sensitive-params` | | 无 | 手动指定敏感参数（逗号分隔） |
| `--use-llm` | | 否 | 启用 LLM 识别敏感参数 |
| `--llm-model` | | xuanyuan | LLM 模型名 |
| `--output` | | ./spp_results | 输出目录 |
| `--data-frequency` | | 自动检测 | 数据频率 |
| `--asset-type` | | stock | 资产类型 (stock/futures) |
| `--contract-code` | | 无 | 期货合约代码 |
| `-q, --quiet` | | 否 | 静默模式 |

## 输出文件

分析完成后在输出目录生成两个文件：

### 1. PNG 可视化报告 (2×2 布局)

**文件名**: `spp_report_{资产名}_{时间戳}.png`

- **(0,0)** MC 分布直方图 + KDE + 最优值/中位数标线 + 衰减率
- **(0,1)** 参数敏感度柱状图（各参数与目标指标的 Pearson 相关系数）
- **(1,0)** 风险-收益散点图（最大回撤 vs 年化收益）
- **(1,1)** 文字总结面板（关键指标 + 鲁棒性判定）

### 2. JSON 结果文件

**文件名**: `spp_result_{资产名}_{时间戳}.json`

```json
{
  "spp_info": {
    "analysis_time": "2026-02-23 10:30:00",
    "elapsed_seconds": 45.2,
    "source_json": "...",
    "asset": "AG",
    "strategy": "AberrationStrategy",
    "config": {
      "n_samples": 300,
      "perturbation_ratio": 0.20,
      "objective": "sharpe_ratio",
      "use_llm": false
    }
  },
  "best_parameters": { "period": 35, "std_dev_upper": 2.0 },
  "best_metrics": { "sharpe_ratio": 1.05, "annual_return": 15.2 },
  "sensitive_params": {
    "method": "llm|manual|all",
    "params": ["period", "std_dev_upper"],
    "llm_reasoning": "..."
  },
  "monte_carlo_stability": {
    "sample_count": 300,
    "perturbation_ratio": 0.20,
    "median": 0.95,
    "mean": 0.93,
    "std": 0.08,
    "p5": 0.80, "p25": 0.88, "p75": 0.98, "p95": 1.02,
    "decay_rate": 0.11,
    "robustness_score": 89.0,
    "param_correlations": { "period": -0.32, "std_dev_upper": 0.15 }
  },
  "verdict": {
    "parameter_robust": "强 (衰减<15%)",
    "stability_score": 0.89,
    "sensitive_param_count": 2,
    "summary": "参数鲁棒性强，局部扰动对策略表现影响较小"
  }
}
```

## 判定标准

### 参数鲁棒性（衰减率 = (最优值 - MC中位数) / |最优值|）

- **强**: 衰减率 < 15%
- **中**: 衰减率 15-30%
- **弱**: 衰减率 > 30%

### 稳定性得分（stability_score = MC中位数 / 最优值）

- 越接近 1.0 越好，表示邻域参数表现接近最优
- > 0.85 为良好
- < 0.70 需要关注

### 参数敏感度（Pearson 相关系数）

- |corr| > 0.3: 该参数对策略表现有显著影响
- |corr| < 0.1: 该参数对策略表现影响很小

## 典型工作流

```bash
# 1. 运行优化器
python run_optimizer.py \
  -d project_trend/data/AG.csv \
  -s project_trend/src/Aberration.py \
  --trials 100

# 2. 运行 SPP 分析
python run_spp_analysis.py \
  -r optimization_results/AG/optimization_AG_Aberration_20260212.json \
  -d project_trend/data/AG.csv \
  -s project_trend/src/Aberration.py

# 3. 查看结果
# - PNG 报告: spp_results/AG/spp_report_AG_*.png
# - JSON 结果: spp_results/AG/spp_result_AG_*.json
```

## 常见问题

**Q: 采样次数设多少合适？**

A: 默认 300 次在大多数场景下足够。参数少（<3个）可降到 200，参数多（>5个）建议 400-500。

**Q: 扰动比例怎么选？**

A: 默认 ±20% 适合大多数策略。如果参数范围本身很窄，可以适当增大到 0.30。整数参数会自动使用稍小的扰动比例（×0.75）。

**Q: LLM 识别敏感参数失败怎么办？**

A: 系统会自动 fallback 到扰动全部参数。也可以用 `--sensitive-params` 手动指定。

**Q: 衰减率为负数是什么意思？**

A: 说明邻域参数的中位数表现反而优于最优参数，这通常意味着优化器找到的"最优"可能是局部最优，或者参数空间比较平坦。

**Q: 和 v2.0 的三维度分析有什么区别？**

A: v3.0 去掉了全局均匀采样和逐年分析，聚焦于最优参数邻域的蒙特卡洛采样。计算量减少 70%+，同时新增了参数敏感度分析和 LLM 辅助功能。
