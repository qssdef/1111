import argparse
import random
import os
import numpy as np
from config import AppConfig


def main():
    parser = argparse.ArgumentParser(description="Modular 6v2 UUV pursuit-evasion simulator")
    parser.add_argument("--mode", choices=["demo", "train", "eval", "train-evader", "eval-evader", "train-sync", "eval-sync"], default="demo")
    parser.add_argument("--episodes", type=int, default=None,
                        help="episode count (default: 1000 for training, 1 for demo, 10 for evaluation)")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--actor-init", default=None, help="initialize a new pursuer actor from a compatible checkpoint")
    parser.add_argument("--evader-checkpoint", default=None, help="frozen target model for pursuer train/eval")
    parser.add_argument("--run-dir", default=None, help="training output directory")
    parser.add_argument("--update-interval", type=int, default=10, help="joint transitions per pursuer critic update (default: 10)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="learner batch size (train-sync default: 1024; other modes use config default)")
    parser.add_argument("--exploration-mode", choices=["persistent_uniform", "gaussian"], default="persistent_uniform")
    parser.add_argument("--eval-stage", choices=["full","current"], default="full", help="full difficulty (default), or checkpoint curriculum stage")
    parser.add_argument("--assignment-mode", choices=["fixed", "dynamic"], default="fixed",
                        help="fixed initial groups (default), or dynamic reassignment for comparison")
    parser.add_argument("--no-curriculum", action="store_true", help="train pursuers on full scenes from episode one")
    parser.add_argument("--no-per", action="store_true", help="use uniform replay instead of PER for pursuer training")
    parser.add_argument("--max-steps", type=int, default=None, help="pursuer training episode limit (default: 480)")
    parser.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads (default: 1)")
    parser.add_argument("--num-envs", type=int, default=8, help="parallel simulation environments (default: 8)")
    parser.add_argument("--role-state-weight", type=float, default=None,
                        help="absolute RI-DARA role cost; train-sync default is 0.2")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--no-gif", action="store_true", help="skip GIF export; keep plots and data")
    parser.add_argument("--output-dir", default=None, help="directory for GIF and plots")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if (args.threads < 1 or args.num_envs < 1 or args.update_interval < 1
            or (args.batch_size is not None and args.batch_size < 1)
            or (args.episodes is not None and args.episodes < 1)
            or (args.max_steps is not None and args.max_steps < 1)):
        parser.error("threads, num-envs, update-interval, batch-size, episodes and max-steps must be positive")
    if args.actor_init and (args.mode != "train" or args.checkpoint):
        parser.error("--actor-init is for a new pursuer training run and cannot be combined with --checkpoint")
    # Static plots and GIF export should work on machines without a Tk desktop runtime.
    if not args.render:
        os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))
        import matplotlib
        matplotlib.use("Agg")
    app = AppConfig(); app.seed = args.seed
    if args.role_state_weight is not None and args.role_state_weight < 0:
        parser.error("role-state-weight must be nonnegative")
    if args.role_state_weight is not None:
        app.reward.role_state_weight = args.role_state_weight
    elif args.mode == "train-sync":
        app.reward.role_state_weight = 0.2
    if args.no_curriculum:
        app.pursuit.curriculum = False
    app.pursuit.assignment_mode = args.assignment_mode
    app.pursuit.prioritized_replay = not args.no_per
    app.pursuit.num_envs = args.num_envs
    app.pursuit.update_interval = args.update_interval
    app.pursuit.exploration_mode = args.exploration_mode
    if args.batch_size is not None:
        app.matd3.batch_size = args.batch_size
    elif args.mode == "train-sync":
        app.matd3.batch_size = 1024
    if args.max_steps is not None:
        app.pursuit.max_steps = args.max_steps
    import torch
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        # TF32 is designed for high-throughput float32 training on Ampere/Ada
        # GPUs and keeps the public model/replay dtypes unchanged.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    output_dir = args.output_dir if args.mode == "eval" else (args.output_dir or "outputs")
    episodes = args.episodes
    if episodes is None:
        if args.mode in ("train", "train-evader", "train-sync"):
            episodes = 1000
        else:
            episodes = 100 if args.mode == "eval-sync" else (10 if args.mode in ("eval", "eval-evader") else 1)
    random.seed(args.seed); np.random.seed(args.seed)
    # Keep torch and the trainer out of module-level imports: Windows spawn
    # workers import this module but only need the NumPy simulation.
    from uuv_sim.runner import (demo, evaluate, evaluate_evader, evaluate_synchronous,
                                train, train_evader, train_synchronous)
    if args.mode == "demo":
        demo(app, episodes, args.render, not args.no_gif, output_dir)
    elif args.mode == "train":
        train(app, episodes, args.checkpoint, args.evader_checkpoint, args.run_dir, args.actor_init)
    elif args.mode == "eval":
        if not args.checkpoint:
            parser.error("--checkpoint is required in eval mode")
        evaluate(app, args.checkpoint, episodes, args.render, not args.no_gif, output_dir, args.evader_checkpoint,args.eval_stage)
    elif args.mode == "train-evader":
        train_evader(app, episodes, args.checkpoint)
    elif args.mode == "train-sync":
        train_synchronous(app, episodes, args.checkpoint, args.evader_checkpoint, args.run_dir)
    elif args.mode == "eval-sync":
        if not args.checkpoint or not args.evader_checkpoint:
            parser.error("--checkpoint and --evader-checkpoint are required in eval-sync mode")
        evaluate_synchronous(app, args.checkpoint, args.evader_checkpoint, episodes,
                             args.output_dir or "outputs/sync_eval")
    else:
        if not args.checkpoint:
            parser.error("--checkpoint is required in eval-evader mode")
        evaluate_evader(app, args.checkpoint, episodes, args.render, not args.no_gif, output_dir)


if __name__ == "__main__":
    main()
