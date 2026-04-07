# 并行加速优化方案

本文档详细描述了对策略参数优化器的并行加速改进方案，基于 GRPO (Group Relative Policy Optimization) 思路，将串行贝叶斯优化改造为批量并行优化。

## 1. 当前算法分析

### 1.1 现有流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    当前两阶段优化流程                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  阶段1: 探索 (30%) - 正态分布采样                                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  for i in range(exploration_trials):  # 串行!             │  │
│  │      params = normal_sample(search_space)                 │  │
│  │      result = backtest(params)         # 耗时主要在这里    │  │
│  │      update_history(result)                               │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  阶段2: 利用 (70%) - TPE 贝叶斯采样                              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  for i in range(exploitation_trials):  # 串行!            │  │
│  │      params = tpe_sample(history)      # 依赖历史结果      │  │
│  │      result = backtest(params)         # 耗时主要在这里    │  │
│  │      update_model(result)              # 更新代理模型      │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 耗时分布

| 阶段 | 占比 | 可并行性 | 说明 |
|-----|------|---------|------|
| 探索阶段回测 | ~30% | ✅ 完全可并行 | N次回测相互独立 |
| 利用阶段回测 | ~65% | ⚠️ 有限并行 | TPE采样依赖历史 |
| 其他开销 | ~5% | - | 采样、更新模型等 |

### 1.3 性能瓶颈

- **99% 时间花在 Backtrader 回测**
- 每次回测是 CPU 单核运算
- 串行执行，无法利用多核

## 2. 并行加速方案

### 2.1 方案概述

采用 **GRPO 风格的批量并行贝叶斯优化**：

```
┌─────────────────────────────────────────────────────────────────┐
│                    改进后的批量并行流程                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  for iteration in range(n_iterations):                          │
│                                                                 │
│      ┌───────────────────────────────────────────────────────┐  │
│      │ 1. 批量采样 K 个候选点                                 │  │
│      │    points = [sample() for _ in range(K)]              │  │
│      └───────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│      ┌───────────────────────────────────────────────────────┐  │
│      │ 2. 并行回测 (K 个同时运行)                             │  │
│      │                                                       │  │
│      │    ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ... ┌─────┐       │  │
│      │    │ x₁  │ │ x₂  │ │ x₃  │ │ x₄  │     │ xₖ  │       │  │
│      │    │回测 │ │回测 │ │回测 │ │回测 │     │回测 │       │  │
│      │    └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘     └──┬──┘       │  │
│      │       └───────┴───────┼───────┴───────────┘           │  │
│      │                       ▼                               │  │
│      │              收集 K 个结果                             │  │
│      └───────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│      ┌───────────────────────────────────────────────────────┐  │
│      │ 3. 批量更新代理模型                                    │  │
│      │    model.update([(x₁,y₁), (x₂,y₂), ..., (xₖ,yₖ)])    │  │
│      └───────────────────────────────────────────────────────┘  │
│                                                                 │
│  总迭代次数 = N / K (如 100次优化, K=8, 只需 13 轮)             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 核心改进点

#### 2.2.1 探索阶段并行化

**修改位置**: `Optimizer/bayesian_optimizer.py` 第 408-544 行

**当前代码**:
```python
for i, params in enumerate(samples):  # 串行循环
    result = self.backtest_engine.run_backtest(strategy_class, data, params)
    value = self.backtest_engine.evaluate_objective(result, objective)
    # ...
```

**改进代码**:
```python
from joblib import Parallel, delayed

def eval_single(params):
    result = self.backtest_engine.run_backtest(strategy_class, data, params)
    if result is None:
        return params, float('-inf'), None
    value = self.backtest_engine.evaluate_objective(result, objective)
    return params, value, result

# 并行执行所有探索试验
results = Parallel(n_jobs=-1)(
    delayed(eval_single)(p) for p in samples
)

# 处理结果
for params, value, result in results:
    history_list.append({...})
    if value > best_exploration_value:
        best_exploration_value = value
        best_exploration_params = params
```

#### 2.2.2 利用阶段批量并行化

**修改位置**: `Optimizer/bayesian_optimizer.py` 第 555-620 行

**当前代码**:
```python
study.optimize(objective_fn, n_trials=exploitation_trials, n_jobs=1)
```

**改进代码**:
```python
batch_size = min(8, exploitation_trials // 5)

for batch_idx in range(0, exploitation_trials, batch_size):
    # 1. 批量采样
    batch_trials = []
    batch_params = []
    for _ in range(batch_size):
        trial = study.ask()
        params = suggest_params(trial, search_space)
        batch_trials.append(trial)
        batch_params.append(params)
    
    # 2. 并行回测
    results = Parallel(n_jobs=-1)(
        delayed(eval_single)(p) for p in batch_params
    )
    
    # 3. 批量更新
    for trial, (params, value, result) in zip(batch_trials, results):
        study.tell(trial, value)
        update_history(params, value, result)
```

#### 2.2.3 SPP 蒙特卡洛并行化

**修改位置**: `Optimizer/spp_analyzer.py` 第 185-210 行

**当前代码**:
```python
def _evaluate_batch(self, param_list, desc=""):
    records = []
    for i, params in enumerate(param_list):  # 串行
        result = self.engine.run_backtest(...)
        # ...
    return pd.DataFrame(records)
```

**改进代码**:
```python
def _evaluate_batch_parallel(self, param_list, desc=""):
    from joblib import Parallel, delayed
    
    def single_eval(params):
        result = self.engine.run_backtest(
            self.strategy_class, self.data, params, calculate_yearly=False)
        if result is None:
            return None
        return {
            'params': params,
            self.config.objective: self.engine.evaluate_objective(result, self.config.objective),
            'annual_return': result.annual_return,
            'max_drawdown': result.max_drawdown,
            'sharpe_ratio': result.sharpe_ratio,
        }
    
    results = Parallel(n_jobs=-1, verbose=10 if self.verbose else 0)(
        delayed(single_eval)(p) for p in param_list
    )
    
    return pd.DataFrame([r for r in results if r is not None])
```

### 2.3 批量采样多样性策略

为避免批量采样点聚集，采用多样化采样：

```python
def diverse_batch_sample(study, search_space, batch_size):
    """多样化批量采样"""
    points = []
    
    # 50% 用 TPE 采样 (exploitation)
    for _ in range(batch_size // 2):
        trial = study.ask()
        points.append((trial, suggest_params(trial, search_space)))
    
    # 30% 用不同 exploration 参数的采样
    for kappa in [1.0, 2.0, 4.0]:
        if len(points) < batch_size:
            trial = study.ask()
            # 增加随机扰动
            params = suggest_params_with_noise(trial, search_space, kappa)
            points.append((trial, params))
    
    # 剩余用纯随机采样 (diversity)
    while len(points) < batch_size:
        trial = study.ask()
        params = random_sample(search_space)
        points.append((trial, params))
    
    return points
```

### 2.4 自适应批量大小

```python
def adaptive_batch_size(iteration, total_iterations, max_batch=8):
    """自适应批量大小: 前期大批量探索，后期小批量精细搜索"""
    progress = iteration / total_iterations
    
    if progress < 0.3:      # 前 30%: 大批量探索
        return max_batch
    elif progress < 0.7:    # 中 40%: 中等批量
        return max(4, max_batch // 2)
    else:                   # 后 30%: 小批量精细搜索
        return max(2, max_batch // 4)
```

## 3. 预期性能提升

### 3.1 单次优化加速

| 场景 | 当前耗时 | 并行后 (8核) | 加速比 |
|-----|---------|-------------|-------|
| 日线 + 简单策略 | 30 秒 | ~5 秒 | **6x** |
| 日线 + 复杂策略 | 2-3 分钟 | ~30 秒 | **5-6x** |
| 分钟 + 简单策略 | 5 分钟 | ~1 分钟 | **5x** |
| 分钟 + 复杂策略 | 30-60 分钟 | ~5-10 分钟 | **6x** |

### 3.2 批量优化加速

| 场景 | 当前耗时 | 并行后 (8核) | 加速比 |
|-----|---------|-------------|-------|
| 61 标的 (日线) | 2.5 小时 | ~25 分钟 | **6x** |
| 61 标的 (分钟) | 33 小时 | ~5.5 小时 | **6x** |

### 3.3 SPP 分析加速

| 场景 | 当前耗时 | 并行后 (8核) | 加速比 |
|-----|---------|-------------|-------|
| 300 次蒙特卡洛 | 10 分钟 | ~1.5 分钟 | **7x** |

## 4. 实现细节

### 4.1 依赖要求

```bash
pip install joblib  # 并行执行库
```

### 4.2 新增配置参数

在 `Optimizer/config.py` 中添加:

```python
@dataclass
class ParallelConfig:
    """并行优化配置"""
    enable_parallel: bool = True           # 是否启用并行
    n_workers: int = -1                    # 工作进程数, -1 表示自动检测
    batch_size: int = 8                    # 批量大小
    adaptive_batch: bool = True            # 是否使用自适应批量大小
    diversity_ratio: float = 0.3           # 多样性采样比例
```

### 4.3 命令行参数

```bash
# 启用并行优化
python run_optimizer.py -d data/AG.csv -s strategy/Aberration.py --parallel

# 指定并行度
python run_optimizer.py -d data/AG.csv -s strategy/Aberration.py --parallel --n-workers 4

# 指定批量大小
python run_optimizer.py -d data/AG.csv -s strategy/Aberration.py --parallel --batch-size 8
```

## 5. 注意事项

### 5.1 内存考虑

- 每个进程会加载数据副本
- 8 核并行约需 8 倍内存
- 对于大数据集可考虑共享内存

### 5.2 随机种子

```python
def set_parallel_seeds(base_seed, n_workers):
    """为每个进程设置不同的随机种子"""
    return [base_seed + i * 1000 for i in range(n_workers)]
```

### 5.3 进程安全

当前实现是进程安全的:
- 每次回测创建新的 Cerebro 实例
- 无共享状态

### 5.4 收敛性

批量更新可能略微影响收敛精度:
- 使用自适应批量大小缓解
- 后期使用小批量精细搜索

## 6. 实施路线图

### Phase 1: 探索阶段并行 (简单)
- 修改 `bayesian_optimizer.py` 探索阶段
- 预期加速: ~1.3x
- 工作量: 1 小时

### Phase 2: SPP 分析并行 (简单)
- 修改 `spp_analyzer.py` 批量回测
- 预期加速: ~7x (SPP 场景)
- 工作量: 1 小时

### Phase 3: 利用阶段批量并行 (中等)
- 修改 `bayesian_optimizer.py` 利用阶段
- 实现批量采样和更新
- 预期加速: ~5-6x (整体)
- 工作量: 3-4 小时

### Phase 4: 完善与优化
- 添加配置参数
- 实现自适应批量大小
- 添加命令行参数
- 工作量: 2 小时

## 7. 参考资料

- [Optuna 并行优化文档](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html)
- [Batch Bayesian Optimization](https://arxiv.org/abs/1712.01815)
- [GRPO: Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)
- [Joblib 并行计算](https://joblib.readthedocs.io/)
