# UUV 协同围捕仿真

当前代码仅保留最新链路：`encirclement-v14`、`paper-task-v11`、RI-DARA 角度角色，以及 6 艘围捕 UUV / 2 艘目标 UUV 的同步独立 MATD3 训练。旧版本检查点、奖励分支、测试代码和实验产物已移除。

完整设计、训练流程、现有评估证据和后续计划见 [仿真方案.md](仿真方案.md)。

## 当前模型

- 围捕侧观测：71 维，包含艇体坐标特征、相对速度、九束前视声呐和 RI-DARA 状态。
- 目标侧观测：24 维，使用九束前视声呐，不再支持旧 25 维模型。
- 围捕奖励：跟踪、环绕、收缩、角色、避碰、保持、捕获、碰撞八项。
- 角色状态权重：默认 `0.2`。
- 当前保留模型：`runs/synchronous_role_state_04/checkpoints/`。

## AutoDL 运行

进入项目目录并确认 CUDA：

```bash
cd /1
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

启动优化后的同步训练：

```bash
CUDA_VISIBLE_DEVICES=0 python -u main.py --mode train-sync \
  --episodes 1000 \
  --num-envs 8 \
  --batch-size 1024 \
  --update-interval 10 \
  --threads 1 \
  --role-state-weight 0.2 \
  --run-dir runs/synchronous_fast_01
```

同步训练会在 CPU 子进程计算下一步环境状态时，并行执行 GPU 学习更新。Linux `top`
中的 CPU `500%` 表示约 5 个逻辑核心正在工作，并不表示异常；实际速度以训练输出和
`training.csv` 中的 `transitions_per_second` 为准。经验池达到 5000 条之前只有推理、
没有批量学习更新，因此这一阶段 GPU 占用通常较低。

评估保留的最新双侧模型：

```bash
python main.py --mode eval-sync \
  --checkpoint runs/synchronous_role_state_04/checkpoints/pursuer_latest.pt \
  --evader-checkpoint runs/synchronous_role_state_04/checkpoints/target_latest.pt \
  --episodes 100 --output-dir outputs/sync_eval_current
```

渲染指定评估回合：

```bash
python scripts/render_sync_eval.py \
  --pursuer-checkpoint runs/synchronous_role_state_04/checkpoints/pursuer_latest.pt \
  --target-checkpoint runs/synchronous_role_state_04/checkpoints/target_latest.pt \
  --episodes 1 10 19
```

绘制同步训练曲线：

```bash
python scripts/plot_sync_training.py runs/synchronous_role_state_04/training.csv
```

Windows 本地运行时将 `python` 替换为 `.\.venv\Scripts\python.exe`，并使用 PowerShell 反引号或单行命令；AutoDL Bash 多行命令必须使用反斜杠 `\`。

只训练目标侧时使用 `--mode train-evader`；只训练围捕侧时使用 `--mode train`，默认冻结当前保留的 24 维目标模型。

## 保留内容

- `uuv_sim/`：当前环境、奖励、RI-DARA、MATD3 与训练代码。
- `scripts/`：当前同步评估渲染、训练曲线和 RI-DARA 诊断工具。
- `runs/synchronous_role_state_04/`：最新完整训练组。
- `outputs/sync_eval_ep800/`：最新保留评估组。
- `analysis/ri_dara_ep800_failures/`：失败回合诊断及其数据、图表。
- `仿真方案.md`、`奖励函数说明.md`、`角色状态奖励实验说明.md`、`目标UUV训练说明.md`：当前文档组。

`sources/` 是 ChatGPT 项目的只读同步目录，清理时不会修改。
