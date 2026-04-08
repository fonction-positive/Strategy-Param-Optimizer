# 并行优化技术文档

## 概述

**探索阶段**采用进程池实现的并行

**贝叶斯阶段**采用batch实现，更新还是采用原来串行的TPE采样，然后我加了一点扰动进去，防止他局部最优


我跑了一个IWF_1D,结果在optimization_results里，您看看这结果是正常的吗，但是其他的有些好像收敛不了，是不是因为完全不适合这个策略，我有点不太懂嘿嘿
然后我测了一下12核的性能，结果也在那个目录里，您看看,我拿我mac跑的，肯能因为有6个是能效核心，其他的性能核心跑完了得等一会，平均cpu占用率应该还能高一些

### 性能测试结果

以下是在 12 核 MacBook Pro 上的测试结果：

```
╔══════════════════════════════════════════════════════════════════╗
║                       性能监控报告                                ║
╠══════════════════════════════════════════════════════════════════╣
║  时间统计                                                         ║
║  ────────────────────────────────────────────────────────────    ║
║  总耗时 (并行后):     2.8 分钟                                   ║
║  估算串行耗时:        20.8 分钟                                  ║
║  总试验次数:          80                                         ║
║  单次回测时间:        15.56 秒                                   ║
║  探索阶段:            0 trials                                   ║
║  利用阶段:            80 trials                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  资源使用                                                         ║
║  ────────────────────────────────────────────────────────────    ║
║  CPU 平均使用率:      619.9%                                     ║
║  CPU 峰值使用率:      1176.0%                                    ║
║  内存初始值:          227.1 MB                                   ║
║  内存平均值:          1707.2 MB                                  ║
║  内存峰值:            2003.9 MB                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  并行效率                                                         ║
║  ────────────────────────────────────────────────────────────    ║
║  并行进程数:          12                                         ║
║  理论加速比:          12.0x                                      ║
║  实际加速比:          7.5x                                       ║
║  并行效率:            62.6%                                      ║
╠══════════════════════════════════════════════════════════════════╣
```

---

## 并行技术栈

```
┌─────────────────────────────────────────────────────────────┐
│                   并行技术栈                                 │
├─────────────────────────────────────────────────────────────┤
│ Level 1: 进程池并行 (ProcessPool)                           │
│   技术: multiprocessing.Pool + spawn context                │
│   特点: 完全隔离，最佳性能，CPU密集型任务首选               │
│   使用场景: 探索阶段、贝叶斯批量回测                        │
├─────────────────────────────────────────────────────────────┤
│ Level 2: 线程池并行 (ThreadPool)                            │
│   技术: concurrent.futures.ThreadPoolExecutor               │
│   特点: 共享内存，中等性能，作为进程池的回退方案            │
│   使用场景: 进程池失败时的备选方案                          │
├─────────────────────────────────────────────────────────────┤
│ Level 3: 串行执行 (Serial)                                  │
│   技术: 单线程顺序执行                                      │
│   特点: 最高稳定性，调试友好，兼容性保底                    │
│   使用场景: 贝叶斯精细搜索阶段、调试模式                    │
└─────────────────────────────────────────────────────────────┘
```

### 三级容错机制

 `bayesian_optimizer.py` ：

```python
# 容错机制设计思路（实际实现分布在多处代码中）

# 1. 探索阶段：检查路径信息
if not strategy_path or not data_path:
    # 缺少路径信息，回退到线程池模式
    use_process_pool = False

# 2. 进程池执行失败时
try:
    ctx = get_context('spawn')
    with ctx.Pool(processes=n_workers) as pool:
        results = pool.imap_unordered(run_single_backtest, tasks)
except Exception as e:
    # 进程池执行失败，回退到线程池
    print(f"进程池失败: {e}，回退到线程池")
    use_process_pool = False

# 3. 批量回测失败时
try:
    return process_pool_batch_backtest(batch_data)
except Exception as e:
    # 回退到线程池批量回测
    return thread_pool_batch_backtest(batch_data)
```

---

## 探索阶段并行实现

线程池有GIL限制肯能没办法真的并行，我直接用进程池了，把策略直接传给每个进程了，肯能开销会大一点，跑的时候占得内存会多一点，但参数多的时候，速度提升应该比开销划算

### 进程池架构

```python
# 核心实现 - bayesian_optimizer.py
def parallel_exploration(samples, strategy_path, data_path, objective):
    """探索阶段并行回测"""
    from multiprocessing import get_context
    from parallel_worker import run_single_backtest
    
    # 计算进程数
    n_workers = parallel_config.n_workers
    if n_workers == -1:
        n_workers = multiprocessing.cpu_count()
    
    # 构建任务列表（可序列化的字典）
    tasks = []
    for i, params in enumerate(samples):
        task_data = {
            'idx': i,
            'params': params,
            'is_default': (i == 0),
            'strategy_path': strategy_path,      # 传路径，不传对象
            'data_path': data_path,              # 传路径，不传对象
            'objective': objective,
            'data_frequency': data_frequency,
            'broker_config_dict': broker_config_dict,  # 预序列化为字典
        }
        tasks.append(task_data)
    
    # 使用 spawn 方式创建进程（跨平台兼容）
    ctx = get_context('spawn')
    
    with ctx.Pool(processes=n_workers) as pool:
        # 使用 imap_unordered 配合 tqdm 显示进度
        results = []
        with tqdm(total=len(tasks), desc="探索阶段") as pbar:
            for result in pool.imap_unordered(run_single_backtest, tasks):
                results.append(result)
                pbar.update(1)
        
        # 按索引排序恢复顺序
        results.sort(key=lambda x: x['idx'])
    
    return results
```

### 子进程工作器设计

```python
# parallel_worker.py
def run_single_backtest(task_data: dict) -> dict:
    """子进程独立执行回测任务"""
    
    idx = task_data['idx']
    params = task_data['params']
    strategy_path = task_data['strategy_path']
    data_path = task_data['data_path']
    objective = task_data['objective']
    
    try:
        # 1. 独立加载策略模块（每个子进程重新加载）
        strategy_class, module, custom_data_class, custom_commission_class = \
            _load_strategy_from_path(strategy_path)
        
        # 2. 独立加载数据
        data = pd.read_csv(data_path)
        data['datetime'] = pd.to_datetime(data['datetime'])
        
        # 3. 重建经纪商配置（从字典恢复）
        broker_config = None
        if task_data.get('broker_config_dict'):
            broker_config = BrokerConfig(**task_data['broker_config_dict'])
        
        # 4. 创建回测引擎
        engine = BacktestEngine(
            data=data,
            strategy_class=strategy_class,
            broker_config=broker_config
        )
        
        # 5. 执行回测
        result = engine.run_backtest(strategy_class, data, params)
        
        # 6. 计算目标值
        value = engine.evaluate_objective(result, objective)
        
        # 7. 返回可序列化的结果
        return {
            'idx': idx,
            'params': params,
            'value': value,
            'result_dict': {
                'sharpe_ratio': result.sharpe_ratio,
                'annual_return': result.annual_return,
                'max_drawdown': result.max_drawdown,
                'total_return': result.total_return,
                'trades_count': result.trades_count,
                'win_rate': result.win_rate,
            },
            'error': None
        }
        
    except Exception as e:
        return {
            'idx': idx,
            'params': params,
            'value': float('-inf'),
            'result_dict': None,
            'error': str(e)
        }
```

---

## 贝叶斯并行 Batch 实现


### Batch 并行方案

**批量采样 + 并行回测 + 串行更新**：

```
┌────────────────────────────────────────────────────────────────┐
│                    Batch 并行贝叶斯优化                         │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─────────────┐                                               │
│  │ TPE 模型    │                                               │
│  │ (当前状态)  │                                               │
│  └──────┬──────┘                                               │
│         │                                                      │
│         ▼                                                      │
│  ┌─────────────────────────────────────────────┐               │
│  │          批量采样 (Batch Sampling)           │               │
│  │                                             │               │
│  │   TPE采样 (70%)      扰动采样 (30%)         │               │
│  │   ┌───┐ ┌───┐       ┌───┐ ┌───┐           │               │
│  │   │P1 │ │P2 │ ...   │P6 │ │P7 │ │P8 │     │               │
│  │   └───┘ └───┘       └───┘ └───┘ └───┘     │               │
│  └─────────────────────────────────────────────┘               │
│         │                                                      │
│         ▼                                                      │
│  ┌─────────────────────────────────────────────┐               │
│  │          并行回测 (Parallel Backtest)        │               │
│  │                                             │               │
│  │   Worker1  Worker2  Worker3  ...  WorkerN   │               │
│  │     ↓        ↓        ↓            ↓        │               │
│  │   [回测]   [回测]   [回测]  ...  [回测]     │               │
│  └─────────────────────────────────────────────┘               │
│         │                                                      │
│         ▼                                                      │
│  ┌─────────────────────────────────────────────┐               │
│  │          串行更新 (Sequential Update)        │               │
│  │                                             │               │
│  │   for result in batch_results:              │               │
│  │       study.tell(trial, value)  # 更新模型  │               │
│  └─────────────────────────────────────────────┘               │
│         │                                                      │
│         ▼                                                      │
│  ┌─────────────┐                                               │
│  │ TPE 模型    │                                               │
│  │ (更新后)    │  ──→ 下一个 Batch                             │
│  └─────────────┘                                               │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 批量采样的同质化问题

因为同一个batch是根据相同的历史数据和模型采样的（都是基于当前最佳区域进行采样），同一批次里的参数肯能会过于相似。
原来串行就没这个问题，他是一个一个跟新的，每次都会更新历史数据和模型，所以每次采样的参数都不一样。

##### 举了个例子，大概这个意思
 ```
  纯TPE采样：
  批次1的8个trial：
  period: [20, 21, 20, 19, 21, 20, 19, 20]  ← 高度相似
  threshold: [1.8, 1.9, 1.7, 1.8, 1.9, 1.8, 1.7, 1.8]

  加了扰动的采样：
  TPE采样 (6个): period: [20, 21, 20, 19, 21, 20]
                 threshold: [1.8, 1.9, 1.7, 1.8, 1.9, 1.8]

  扰动采样 (2个): period: [15, 28]  ← 扩展探索范围
                 threshold: [2.3, 1.2]  ← 发现新区域
```

### 自适应批量大小

根据优化进度动态调整批量大小，前期大批量快速探索，后期小批量精细收敛：

```python
# bayesian_optimizer.py - _adaptive_batch_size 方法
def _adaptive_batch_size(self, iteration: int, total_iterations: int) -> int:
    """
    根据优化进度动态调整批量大小
    
    优化策略:
    - 前30%: 大批量探索 (快速覆盖空间)
    - 中40%: 中等批量 (平衡探索与收敛) 
    - 后30%: 小批量精细搜索 (局部优化)
    """
    if not self.batch_parallel_config.adaptive_batch:
        return self.batch_parallel_config.batch_size
    
    progress = iteration / total_iterations if total_iterations > 0 else 0
    max_batch = self.batch_parallel_config.max_batch_size  # 默认 12
    min_batch = self.batch_parallel_config.min_batch_size  # 默认 2
    
    if progress < 0.3:      # 前30%: 大批量探索
        return max_batch        # 12
    elif progress < 0.7:    # 中40%: 中等批量
        return max_batch // 2   # 6
    else:                   # 后30%: 小批量精细搜索
        return min_batch        # 2
```

### 多样性增强采样

70% 使用 TPE 模型采样，30% 添加随机扰动，防止过度收敛陷入局部最优：

```python
# bayesian_optimizer.py - _diverse_batch_sampling 方法
def _diverse_batch_sampling(self, study, search_space, batch_size):
    """
    多样化批量采样策略
    
    防止批量采样的同质化问题:
    - 70%: TPE智能采样 (基于贝叶斯模型)
    - 30%: 扰动采样 (增加探索多样性)
    
    注：比例由 BatchParallelConfig.diversity_ratio 控制，默认 0.3
    """
    params_batch = []
    diversity_count = int(batch_size * self.batch_parallel_config.diversity_ratio)
    
    # 主要部分: TPE 采样
    for _ in range(batch_size - diversity_count):
        trial = study.ask()
        params = self._suggest_params(trial, search_space)
        params_batch.append({'trial': trial, 'params': params})
    
    # 多样性部分: 扰动采样
    for _ in range(diversity_count):
        trial = study.ask()
        params = self._suggest_params(trial, search_space)
        # 关键: 添加随机扰动防止过度收敛
        perturbed_params = self._add_perturbation(params, search_space)
        params_batch.append({'trial': trial, 'params': perturbed_params})
    
    return params_batch
```

### 扰动机制实现

```python
# bayesian_optimizer.py - _add_perturbation 方法
def _add_perturbation(self, params, search_space):
    """
    为参数添加智能扰动
    
    Args:
        params: 原始参数
        search_space: 搜索空间定义  
    
    扰动强度由 BatchParallelConfig.perturbation_strength 控制，默认 0.1 (10%)
    """
    perturbed_params = params.copy()
    strength = self.batch_parallel_config.perturbation_strength  # 默认 0.1

    for name, value in params.items():
        if name in search_space:
            space_config = search_space[name]
            param_range = space_config.max_value - space_config.min_value

            if space_config.param_type == "int":
                # 整数参数：随机扰动 ±10% 范围
                max_perturbation = max(1, int(param_range * strength))
                perturbation = np.random.randint(
                    -max_perturbation, 
                    max_perturbation + 1
                )
                new_value = np.clip(
                    value + perturbation,
                    space_config.min_value,
                    space_config.max_value
                )
                perturbed_params[name] = int(new_value)

            elif space_config.param_type == "float":
                # 浮点参数：均匀分布扰动
                perturbation = np.random.uniform(
                    -param_range * strength, 
                    param_range * strength
                )
                new_value = np.clip(
                    value + perturbation,
                    space_config.min_value,
                    space_config.max_value
                )
                perturbed_params[name] = new_value

    return perturbed_params
```

### 混合模式：批量并行 + 串行精细搜索

利用阶段分为两部分：
- **批量并行阶段 (70%)**：大批量采样 + 并行回测，快速收敛
- **串行精细搜索 (30%)**：传统串行贝叶斯，精细调优

```python
# bayesian_optimizer.py - _batch_parallel_exploitation 方法
def _batch_parallel_exploitation(self, study, strategy_class, data, 
                                   search_space, objective, exploitation_trials, ...):
    """批量并行利用阶段优化"""
    
    # 计算并行/串行试验分配
    if self.batch_parallel_config.hybrid_mode:
        # 混合模式：70% 批量并行 + 30% 串行精细搜索
        parallel_trials = int(exploitation_trials * self.batch_parallel_config.parallel_ratio)
        serial_trials = exploitation_trials - parallel_trials
    else:
        # 全批量并行模式
        parallel_trials = exploitation_trials
        serial_trials = 0
    
    # === 第一部分：批量并行优化 ===
    if parallel_trials > 0:
        batch_size = self.batch_parallel_config.batch_size
        n_batches = (parallel_trials + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            # 1. 自适应批量大小
            current_batch_size = self._adaptive_batch_size(batch_idx, n_batches)
            
            # 2. 多样化批量采样
            batch_data = self._diverse_batch_sampling(study, search_space, current_batch_size)
            
            # 3. 并行回测（进程池）
            batch_results = self._parallel_batch_backtest(batch_data, ...)
            
            # 4. 串行更新 TPE 模型
            for i, result in enumerate(batch_results):
                trial = batch_data[i]['trial']
                value = result['value']
                study.tell(trial, value)  # 更新模型
    
    # === 第二部分：串行精细搜索 ===
    if serial_trials > 0:
        # 传统串行贝叶斯优化
        study.optimize(objective_fn, n_trials=serial_trials, n_jobs=1)
    
    return current_best_value
```

### 批量并行回测实现

```python
# bayesian_optimizer.py - _parallel_batch_backtest 方法
def _parallel_batch_backtest(self, batch_data, strategy_class, data, objective, verbose):
    """并行批量回测"""
    from multiprocessing import get_context
    from parallel_worker import run_single_backtest
    
    # 准备任务数据
    tasks = []
    for i, item in enumerate(batch_data):
        task_data = {
            'idx': i,
            'params': item['params'],
            'strategy_path': self.backtest_engine.strategy_path,
            'data_path': self.backtest_engine.data_path,
            'objective': objective,
            'broker_config_dict': serialize_broker_config(self.backtest_engine.broker_config),
        }
        tasks.append(task_data)
    
    # 计算进程数
    n_workers = self.parallel_config.n_workers
    if n_workers == -1:
        n_workers = multiprocessing.cpu_count()
    
    try:
        # 使用 spawn 进程池（跨平台统一）
        ctx = get_context('spawn')
        with ctx.Pool(processes=n_workers) as pool:
            if verbose:
                results = []
                with tqdm(total=len(tasks), desc="批量回测") as pbar:
                    for result in pool.imap_unordered(run_single_backtest, tasks):
                        results.append(result)
                        pbar.update(1)
                results.sort(key=lambda x: x['idx'])
            else:
                results = pool.map(run_single_backtest, tasks)
        return results
        
    except Exception as e:
        # 进程池失败，回退到线程池
        print(f"进程池批量回测失败: {e}，回退到线程池")
        return self._thread_batch_backtest(batch_data, ...)
```

---

## 配置管理

### 并行配置参数

```python
# config.py
@dataclass
class ParallelConfig:
    """基础并行配置"""
    enable_parallel: bool = True           # 是否启用并行
    n_workers: int = -1                    # 工作进程数 (-1=自动检测CPU核心数)
    batch_size: int = 8                    # 批量大小（已移至 BatchParallelConfig）
    verbose_joblib: int = 0                # 日志级别

@dataclass 
class BatchParallelConfig:
    """批量并行配置"""
    enable_batch_parallel: bool = True     # 是否启用批量并行
    batch_size: int = 8                    # 批量大小
    adaptive_batch: bool = True            # 自适应批量大小
    max_batch_size: int = 12               # 最大批量大小（内部固定值）
    min_batch_size: int = 2                # 最小批量大小（内部固定值）
    diversity_ratio: float = 0.3           # 多样性采样比例（30% 扰动）
    hybrid_mode: bool = True               # 混合模式（并行+串行）
    parallel_ratio: float = 0.7            # 并行阶段占比（70%）
    perturbation_strength: float = 0.1     # 扰动强度（10%）
```

### 自动化配置

```python
# CPU核心数自动检测
n_workers = config.n_workers
if n_workers == -1:
    n_workers = multiprocessing.cpu_count()
    print(f"[并行] 自动检测到 {n_workers} 个CPU核心")
```

### 进程创建方式

当前实现**统一使用 spawn 方式**创建进程，确保跨平台兼容性：

```python
# 统一使用 spawn（macOS、Windows、Linux 都支持）
ctx = multiprocessing.get_context('spawn')
with ctx.Pool(processes=n_workers) as pool:
    ...
```

> 注：虽然 Linux 上 fork 更高效，但 spawn 更安全，避免了 fork 在多线程环境下的潜在问题。

---

## 使用方法

### 基本并行优化

```bash
# 启用并行优化（默认）
python run_optimizer.py -d data/QQQ.csv -s strategy/MyStrategy.py

# 指定进程数
python run_optimizer.py -d data/QQQ.csv -s strategy/MyStrategy.py \
  --n-workers 8

# 禁用并行（调试模式）
python run_optimizer.py -d data/QQQ.csv -s strategy/MyStrategy.py \
  --no-parallel
```

### 批量并行配置

```bash
# 自定义批量大小
python run_optimizer.py -d data/QQQ.csv -s strategy/MyStrategy.py \
  --batch-size 12

# 禁用混合模式（全批量并行，不使用串行精细搜索）
python run_optimizer.py -d data/QQQ.csv -s strategy/MyStrategy.py \
  --no-hybrid-mode

# 自定义并行比例（默认0.7，即70%并行+30%串行）
python run_optimizer.py -d data/QQQ.csv -s strategy/MyStrategy.py \
  --parallel-ratio 0.8

# 禁用自适应批量大小
python run_optimizer.py -d data/QQQ.csv -s strategy/MyStrategy.py \
  --no-adaptive-batch
```

### 禁用批量并行（传统串行模式）

```bash
# 完全回到传统串行贝叶斯优化
python run_optimizer.py -d data/QQQ.csv -s strategy/MyStrategy.py \
  --no-batch-parallel
```

### 完整示例

```bash
# 高性能配置（16核服务器）
python run_optimizer.py -d data/QQQ.csv -s strategy/MyStrategy.py \
  --n-workers 16 \
  --batch-size 16 \
  --parallel-ratio 0.8 \
  --trials 200

# 低内存配置（笔记本电脑）
python run_optimizer.py -d data/QQQ.csv -s strategy/MyStrategy.py \
  --n-workers 4 \
  --batch-size 4 \
  --parallel-ratio 0.6
```

---

## 命令行参数一览

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--parallel` | 启用 | 启用并行优化（探索阶段） |
| `--no-parallel` | - | 禁用并行优化 |
| `--n-workers` | -1 | 工作进程数（-1=自动检测CPU核心数） |
| `--batch-parallel` | 启用 | 启用批量并行优化（利用阶段） |
| `--no-batch-parallel` | - | 禁用批量并行优化 |
| `--batch-size` | 8 | 批量大小 |
| `--no-adaptive-batch` | - | 禁用自适应批量大小 |
| `--no-hybrid-mode` | - | 禁用混合模式 |
| `--parallel-ratio` | 0.7 | 批量并行阶段占比 |

---
---
## 性能优化建议

### 1. 根据 CPU 核心数调整配置

```bash
# 查看 CPU 核心数
python -c "import multiprocessing; print(multiprocessing.cpu_count())"

# 建议：n_workers = CPU核心数，batch_size = n_workers 或稍大
```

### 2. 内存管理

每个工作进程会独立加载数据和策略，内存占用约为：
```
总内存 ≈ 单进程内存 × n_workers + 主进程内存
```

如果内存不足，可以减少 `--n-workers` 或 `--batch-size`。

### 3. 大批量 vs 小批量

| 批量大小 | 优点 | 缺点 |
|----------|------|------|
| 大批量 | 并行效率高，探索速度快 | TPE 模型更新不及时，可能错过最优 |
| 小批量 | 模型更新及时，收敛更精确 | 并行效率低，总耗时长 |

建议使用默认的自适应批量大小策略，前期大批量快速探索，后期小批量精细收敛。

### 4. 混合模式的优势

混合模式（默认 70% 并行 + 30% 串行）兼顾了效率和精度：
- 并行阶段快速收敛到高性能区域
- 串行阶段精细搜索，确保找到最优参数


---

## 深入理解：batch_size 与 n_workers 的关系

### 为什么建议 `batch_size ≈ n_workers`？

核心原因是**最大化 CPU 利用率**：

```
场景1: batch_size = 8, n_workers = 8
┌─────────────────────────────────────┐
│ Worker1: [任务1] ████████           │
│ Worker2: [任务2] ████████           │
│ Worker3: [任务3] ████████           │
│ Worker4: [任务4] ████████           │
│ Worker5: [任务5] ████████           │
│ Worker6: [任务6] ████████           │
│ Worker7: [任务7] ████████           │
│ Worker8: [任务8] ████████           │
└─────────────────────────────────────┘
→ 8个任务同时跑，8个核心都在工作 ✅

场景2: batch_size = 4, n_workers = 8
┌─────────────────────────────────────┐
│ Worker1: [任务1] ████████           │
│ Worker2: [任务2] ████████           │
│ Worker3: [任务3] ████████           │
│ Worker4: [任务4] ████████           │
│ Worker5: [空闲]                     │  ← 浪费了！
│ Worker6: [空闲]                     │  ← 浪费了！
│ Worker7: [空闲]                     │  ← 浪费了！
│ Worker8: [空闲]                     │  ← 浪费了！
└─────────────────────────────────────┘
→ 只有4个核心在工作，另外4个闲着 ❌
```

### 为什么说"或稍大"？

因为不同参数组合的回测时间可能不同，会有**负载不均衡**：

```
batch_size = 8, n_workers = 8（刚好相等）
┌─────────────────────────────────────┐
│ Worker1: [任务1] ████                │ 完成
│ Worker2: [任务2] ██████              │ 完成
│ Worker3: [任务3] ████████████████    │ 还在跑...
│ Worker4: [任务4] ████████            │ 完成
│ ...                                 │
└─────────────────────────────────────┘
→ Worker1,2,4 已完成但在等 Worker3，CPU 利用率下降

batch_size = 12, n_workers = 8（稍大一点）
┌─────────────────────────────────────┐
│ Worker1: [任务1] ████ → [任务9] ██   │ 
│ Worker2: [任务2] ██████ → [任务10]   │
│ Worker3: [任务3] ████████████████    │ 慢任务
│ Worker4: [任务4] ████ → [任务11] ██  │
│ ...                                 │
└─────────────────────────────────────┘
→ 快的 Worker 完成后立即领取新任务，减少空闲等待
```

### 但 batch_size 也不能太大

太大的问题：
1. **TPE 模型更新不及时**：同一批的参数都是基于同一个模型采样的，批量越大，模型越"过时"
2. **内存开销**：要同时准备更多任务数据

---

## 多核服务器场景（如 120 核）

### 问题：batch_size = 120 会怎样？

```
┌────────────────────────────────────────────────────────────┐
│                 batch_size = 120 的问题                     │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  TPE 模型状态: [基于前N次试验]                              │
│                     │                                      │
│                     ▼                                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  一次性采样 120 个参数（全部基于同一个模型！）         │  │
│  │                                                      │  │
│  │  param_1, param_2, param_3 ... param_120             │  │
│  │     ↓        ↓        ↓            ↓                 │  │
│  │   相似     相似     相似    ...   相似               │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
│  问题：120个参数可能都集中在同一个区域！                    │
│       相当于在同一个地方挖了120次，浪费计算资源             │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 串行 vs 大批量的对比

```
串行贝叶斯（理想情况）：
Trial 1 → 更新模型 → Trial 2 → 更新模型 → Trial 3 → ...
  ↓           ↓           ↓
探索区域A   发现A不好    转向区域B   发现B更好   深挖B
                        （及时调整方向）

batch_size = 120：
Trial 1-120 全部基于同一个模型采样
  ↓
可能 100 个都在区域A，只有 20 个碰巧在区域B
（浪费了大量计算在已知不好的区域）
```

### 建议：多核场景用较小的 batch_size + 多轮

```python
# 120 核的推荐配置
n_workers = 120        # 充分利用所有核心
batch_size = 16~24     # 但每批只采样 16~24 个参数

# 这样的流程：
# Batch 1: 采样24个 → 120核并行跑（部分核空闲） → 更新模型
# Batch 2: 采样24个 → 120核并行跑 → 更新模型  
# Batch 3: 采样24个 → ...
# 
# 虽然每批没跑满120核，但模型更新更频繁，搜索更智能
```

### 不同核心数的推荐配置

| 核心数 | 推荐 batch_size | 原因 |
|--------|-----------------|------|
| 4-8 核 | = n_workers | 核少，跑满优先 |
| 12-16 核 | = n_workers | 基本平衡 |
| 32-64 核 | 16-24 | 开始需要控制批量大小 |
| 120+ 核 | 16-32 | 模型更新频率更重要 |

### 大批量的替代方案：增加 diversity_ratio

如果一定要大批量，可以增加扰动比例来增加多样性：

```python
# 默认 30% 扰动可能不够
# 大批量场景可以考虑增加到 50%
diversity_ratio = 0.5  # 50% TPE + 50% 扰动
```

这样即使批量大，也有一半的参数是随机扰动的，能覆盖更广的搜索区域。

