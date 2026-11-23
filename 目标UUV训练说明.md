# 目标 UUV 训练说明

## 当前任务定义

- 目标 UUV 不执行交叉机动，训练配置固定令 `cross_probability_per_second=0`。
- 目标 UUV 使用与我方 UUV 相同的 120° 前视声呐：15° 间隔、九束、8 m 量程。
- 声呐探测地图边界、固定障碍物和同侧活动 UUV；真实接触使用相同的 1 m 硬碰撞阈值与穿透分离。
- 目标观测为 24 维；当前代码只接受这一观测结构。

## 唯一奖励

目标奖励严格采用论文式 (30)：

```text
r_target = sum_i(d_i,t - d_i,t-1) + min(l_1, ..., l_9) - L
```

其中 `d_i,t` 是当前目标与第 `i` 艘我方 UUV 在时刻 `t` 的距离，`l_j` 是第 `j` 束声呐的实际量程，`L=8 m` 是最大量程。实现中没有存活奖励、交叉奖励、捕获罚分或额外避碰奖励。

## 训练方式

仅训练目标侧：

```powershell
.\.venv\Scripts\python.exe main.py --mode train-evader --episodes 1000
```

输出模型为 `checkpoints/evader_sonar_latest.pt`。

双方同步训练（默认总计 1000 个 episode，每批并行运行 2 个 episode）：

```powershell
.\.venv\Scripts\python.exe main.py --mode train-sync --run-dir runs/synchronous_current
```

同步训练只共享环境状态与仿真时间步。我方和目标方分别持有独立的 Actor、目标网络、Critic、优化器、更新计数和经验池，并分别保存为：

- `checkpoints/pursuer_latest.pt`
- `checkpoints/target_latest.pt`

