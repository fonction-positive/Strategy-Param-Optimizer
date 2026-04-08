# 并行优化技术文档

## 概述
探索阶段采用进程池实现的并行

贝叶斯阶段采用batch实现，更新还是采用原来串行的TPE采样，然后我加了一点扰动进去，让他能去探索别的区域，不会局部最优


我跑了一个IWF_1D,结果在optimization_results里，您看看这结果是正常的吗，但是其他的有些好像收敛不了，是不是因为完全不适合这个策略，我有点不太懂嘿嘿
然后我测了一下12核的性能，结果也在那个目录里，您看看,我拿我mac跑的，肯能因为有6个是能效核心，其他的性能核心跑完了得等一会，平均cpu占用率应该还能高一些


```
╔══════════════════════════════════════════════════════════════════╗
║                       性能监控报告                                ║
╠══════════════════════════════════════════════════════════════════╣
║  时间统计                                                         ║
║  ────────────────────────────────────────────────────────────    ║
║  总耗时 (并行后):     2.8 分钟                                   ║
║  估算串行耗时:        20.8 分钟                                  ║
║  总试验次数:          80                                       ║
║  单次回测时间:        15.56 秒                                  ║
║  探索阶段:            0 trials                                  ║
║  利用阶段:            80 trials                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  资源使用                                                         ║
║  ────────────────────────────────────────────────────────────    ║
║  CPU 平均使用率:      619.9%                                      ║
║  CPU 峰值使用率:      1176.0%                                      ║
║  内存初始值:          227.1 MB                                   ║
║  内存平均值:          1707.2 MB                                   ║
║  内存峰值:            2003.9 MB                                   ║
╠══════════════════════════════════════════════════════════════════╣
║  并行效率                                                         ║
║  ────────────────────────────────────────────────────────────    ║
║  并行进程数:          12                                       ║
║  理论加速比:          12.0x                                      ║
║  实际加速比:          7.5x                                       ║
║  并行效率:            62.6%                                      ║
╠══════════════════════════════════════════════════════════════════╣
```

### 探索阶段并行实现
线程池有GIL限制肯能没办法真的并行，我直接用进程池了，把策略直接传给每个进程了，肯能开销会大一点，跑的时候占得内存会多一点，但参数多的时候，速度提升应该比开销划算

```
┌─────────────────────────────────────────────────────────┐
│                   并行技术栈                              │
├─────────────────────────────────────────────────────────┤
│ Level 1: 进程池并行 (ProcessPool)                        │
│   技术: multiprocessing.Pool + spawn context             │
│   特点: 完全隔离，最佳性能，CPU密集型任务首选               │
├─────────────────────────────────────────────────────────┤
│ Level 2: 线程池并行 (ThreadPool)                         │
│   技术: concurrent.futures.ThreadPoolExecutor            │
│   特点: 共享内存，中等性能，I/O密集型任务适用               │
├─────────────────────────────────────────────────────────┤
│ Level 3: 串行执行 (Serial)                               │
│   技术: 单线程顺序执行                                     │
│   特点: 最高稳定性，调试友好，兼容性保底                   │
└─────────────────────────────────────────────────────────┘
```

```python
def robust_parallel_execution(tasks):
    """
    三级容错机制确保最大兼容性
    
    Level 1: 进程池 (最佳性能)
    Level 2: 线程池 (中等性能)
    Level 3: 串行执行 (保底方案)
    """
    try:
        # 尝试进程池执行
        return process_pool_execution(tasks)
        
    except Exception as e1:
        logger.warning(f"进程池失败: {e1}，回退到线程池")
        
        try:
            # 回退到线程池
            return thread_pool_execution(tasks)
            
        except Exception as e2:
            logger.warning(f"线程池失败: {e2}，回退到串行执行")
            
            # 最终保底: 串行执行
            return serial_execution(tasks)
```

### 2. 进程池架构 (主要模式)

```python
# 核心实现原理
def process_pool_parallel():
    # 1. 创建隔离的子进程池
    ctx = multiprocessing.get_context('spawn')
    
    with ctx.Pool(processes=n_workers) as pool:
        # 2. 异步分发任务
        results = []
        with tqdm(total=len(tasks)) as pbar:
            for result in pool.imap_unordered(run_single_backtest, tasks):
                results.append(result)
                pbar.update(1)
        
        # 3. 按索引排序恢复顺序
        results.sort(key=lambda x: x['idx'])
    
    return results
```


### 3. 子进程工作器设计

```python
def run_single_backtest(task_data: dict) -> dict:
    """子进程独立执行回测任务"""
    
    # 1. 独立加载策略模块
    strategy_class = load_strategy_from_path(task_data['strategy_path'])
    
    # 2. 独立加载数据
    data = pd.read_csv(task_data['data_path'])
    
    # 3. 独立创建回测引擎
    engine = BacktestEngine()
    
    # 4. 执行回测计算
    result = engine.run_backtest(strategy_class, data, task_data['params'])
    
    # 5. 返回序列化结果
    return {
        'idx': task_data['idx'],
        'value': evaluate_objective(result, task_data['objective']),
        'result_dict': extract_metrics(result),
        'error': None
    }
```


## 配置管理

### 1. 并行配置参数

```python
@dataclass
class ParallelConfig:
    """基础并行配置"""
    enable_parallel: bool = True           # 是否启用并行
    n_workers: int = -1                    # 工作进程数 (-1=自动检测CPU核心数)
    batch_size: int = 8                    # 批量大小
    verbose_joblib: int = 0                # 日志级别

@dataclass 
class BatchParallelConfig:
    """批量并行配置"""
    enable_batch_parallel: bool = True     # 是否启用批量并行
    batch_size: int = 8                    # 批量大小
    adaptive_batch: bool = True            # 自适应批量大小
    max_batch_size: int = 12               # 最大批量大小
    min_batch_size: int = 2                # 最小批量大小
    diversity_ratio: float = 0.3           # 多样性采样比例
    hybrid_mode: bool = True               # 混合模式
    parallel_ratio: float = 0.7            # 并行阶段占比
    perturbation_strength: float = 0.1     # 扰动强度
```

### 2. 自动化配置

```python
# CPU核心数自动检测
n_workers = config.n_workers
if n_workers == -1:
    n_workers = multiprocessing.cpu_count()
    print(f"[并行] 自动检测到 {n_workers} 个CPU核心")

# 内存自适应调整
available_memory = get_available_memory()
if available_memory < 4 * 1024**3:  # 小于4GB
    batch_size = min(batch_size, 4)
    print(f"[并行] 内存受限，调整批量大小为 {batch_size}")
```

### 3. 平台兼容性

```python
# 跨平台进程创建方式
if sys.platform == 'darwin':  # macOS
    context = 'spawn'
elif sys.platform == 'win32':  # Windows  
    context = 'spawn'
else:  # Linux
    context = 'fork'  # 更高效

ctx = multiprocessing.get_context(context)
```

---

## 贝叶斯并行batch实现

### 1. 自适应批量大小

```python
def adaptive_batch_size(iteration: int, total_iterations: int) -> int:
    """
    根据优化进度动态调整批量大小
    
    优化策略:
    - 前30%: 大批量探索 (快速覆盖空间)
    - 中40%: 中等批量 (平衡探索与收敛) 
    - 后30%: 小批量精细搜索 (局部优化)
    """
    progress = iteration / total_iterations
    
    if progress < 0.3:      # 探索阶段
        return max_batch_size        # 12
    elif progress < 0.7:    # 收敛阶段
        return max_batch_size // 2   # 6
    else:                   # 精细阶段
        return min_batch_size        # 2
```

### 2. 多样性增强采样
80%是按照原先串行贝叶斯的TPE采样，20%是增加扰动的采样，防止过度收敛只能找到局部最优解。
因为同一个batch是根据相同的历史数据和模型采样的（都是基于当前最佳区域进行采样），同一批次里的参数肯能会过于相似。
原来串行就没这个问题，他是一个一个跟新的，每次都会更新历史数据和模型，所以每次采样的参数都不一样。
#### 举了个例子，大概这个意思
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


```python
def diverse_batch_sampling(study, search_space, batch_size):
    """
    多样化批量采样策略
    
    防止批量采样的同质化问题:
    - 80%: TPE智能采样 (基于贝叶斯模型)
    - 20%: 扰动采样 (增加探索多样性)
    """
    params_batch = []
    diversity_count = int(batch_size * 0.2)
    
    # 主要部分: TPE采样
    for _ in range(batch_size - diversity_count):
        trial = study.ask()
        params = suggest_params(trial, search_space)
        params_batch.append({'trial': trial, 'params': params})
    
    # 多样性部分: 扰动采样
    for _ in range(diversity_count):
        trial = study.ask()
        params = suggest_params(trial, search_space)
        # 关键: 添加随机扰动防止过度收敛
        perturbed_params = add_perturbation(params, search_space)
        params_batch.append({'trial': trial, 'params': perturbed_params})
    
    return params_batch
```

```python
# 扰动机制的技术实现

  def add_perturbation(params, search_space, strength=0.1):
      """
      为参数添加智能扰动
      
      Args:
          params: 原始参数
          search_space: 搜索空间定义  
          strength: 扰动强度 (0.1 = 10%)
      """
      perturbed_params = params.copy()

      for name, value in params.items():
          if name in search_space:
              space_config = search_space[name]
              param_range = space_config.max_value - space_config.min_value

              if space_config.param_type == "int":
                  # 整数参数：随机扰动±10%范围
                  max_perturbation = max(1, int(param_range * strength))
                  perturbation = np.random.randint(-max_perturbation,
  max_perturbation + 1)
                  new_value = np.clip(
                      value + perturbation,
                      space_config.min_value,
                      space_config.max_value
                  )
                  perturbed_params[name] = int(new_value)

              elif space_config.param_type == "float":
                  # 浮点参数：高斯扰动
                  perturbation = np.random.normal(0, param_range * strength)
                  new_value = np.clip(
                      value + perturbation,
                      space_config.min_value,
                      space_config.max_value
                  )
                  perturbed_params[name] = new_value

      return perturbed_params
```



---

## 性能优化策略

### 1. CPU利用率最大化

```python
# CPU亲和性设置 (Linux)
import os
def set_cpu_affinity():
    if hasattr(os, 'sched_setaffinity'):
        # 绑定进程到特定CPU核心
        available_cpus = os.sched_getaffinity(0)
        os.sched_setaffinity(0, available_cpus)

# NUMA优化 (多路CPU系统)
def optimize_numa():
    try:
        import numa
        # 绑定内存分配到本地NUMA节点
        numa.set_preferred(numa.get_run_node_mask())
    except ImportError:
        pass
```

### 2. 内存管理优化

```python
# 任务数据预处理
def prepare_task_data(params_list, strategy_path, data_path):
    """
    优化任务数据结构，减少序列化开销
    """
    # 预序列化配置对象
    broker_config_dict = serialize_broker_config(broker_config)
    
    tasks = []
    for i, params in enumerate(params_list):
        task = {
            'idx': i,
            'params': params,                    # 小对象，序列化快
            'strategy_path': strategy_path,      # 字符串，序列化快
            'data_path': data_path,             # 字符串，序列化快
            'broker_config_dict': broker_config_dict,  # 预序列化
        }
        tasks.append(task)
    
    return tasks

# 内存监控和自适应调整
def monitor_memory_usage():
    import psutil
    
    memory_percent = psutil.virtual_memory().percent
    if memory_percent > 85:
        # 内存使用率过高，降低并行度
        return max(1, current_workers // 2)
    return current_workers
```

### 3. I/O优化策略

```python
# 数据预加载缓存
class DataCache:
    def __init__(self):
        self._cache = {}
    
    def get_data(self, data_path):
        if data_path not in self._cache:
            # 仅在首次访问时加载数据
            self._cache[data_path] = pd.read_csv(data_path)
        return self._cache[data_path].copy()

# 异步I/O优化
async def async_load_data(data_paths):
    import aiofiles
    import asyncio
    
    async def load_single(path):
        async with aiofiles.open(path, 'r') as f:
            content = await f.read()
            return pd.read_csv(io.StringIO(content))
    
    # 并发加载所有数据文件
    tasks = [load_single(path) for path in data_paths]
    return await asyncio.gather(*tasks)
```

---

## 使用方法

### 1. 基本并行优化

```bash
# 启用并行优化 (默认)
python run_optimizer.py -d data/QQQ.csv -s strategy/MyStrategy.py

# 指定进程数
python run_optimizer.py -d data/QQQ.csv -s strategy/MyStrategy.py \
  --n-workers 8

# 禁用并行 (调试模式)
python run_optimizer.py -d data/QQQ.csv -s strategy/MyStrategy.py \
  --no-parallel
```

### 2. 高级并行配置

```bash
# 自定义批量配置
python run_optimizer.py -d data/QQQ.csv -s strategy/MyStrategy.py \
  --batch-size 12 \
  --max-batch-size 16 \
  --min-batch-size 4

# 混合模式配置
python run_optimizer.py -d data/QQQ.csv -s strategy/MyStrategy.py \
  --hybrid-mode \
  --parallel-ratio 0.8

# 禁用混合模式 (全并行)
python run_optimizer.py -d data/QQQ.csv -s strategy/MyStrategy.py \
  --no-hybrid-mode
```

### 3. 性能调优参数

```bash
python run_optimizer.py -d data/QQQ.csv -s strategy/MyStrategy.py \
  --n-workers 16 \
  --batch-size 20 \
  --max-batch-size 32 \
  --diversity-ratio 0.15

python run_optimizer.py -d data/QQQ.csv -s strategy/MyStrategy.py \
  --n-workers 4 \
  --batch-size 6 \
  --max-batch-size 8 \
  --diversity-ratio 0.25
```

