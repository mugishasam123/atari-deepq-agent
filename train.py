"""
Train a DQN agent on an Atari environment using Stable Baselines3.
Supports CnnPolicy and MlpPolicy; tune hyperparameters via CLI.
"""
import argparse
import os
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
ENV_ID = "ALE/Pong-v5"
MODEL_DIR = "models"
LOG_DIR = "logs"


def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN on Atari (Pong)")
    parser.add_argument("--policy", type=str, default="CnnPolicy",
                        choices=["CnnPolicy", "MlpPolicy"],
                        help="Policy: CnnPolicy (images) or MlpPolicy (vector)")
    parser.add_argument("--lr", type=float, default=2.5e-4,
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for updates")
    parser.add_argument("--epsilon_start", type=float, default=1.0,
                        help="Initial epsilon (exploration)")
    parser.add_argument("--epsilon_end", type=float, default=0.05,
                        help="Final epsilon")
    parser.add_argument("--epsilon_decay", type=float, default=250_000,
                        help="Timesteps over which epsilon decays")
    parser.add_argument("--steps", type=int, default=500_000,
                        help="Total training timesteps")
    parser.add_argument("--n_envs", type=int, default=4,
                        help="Parallel environments")
    parser.add_argument("--save_as_best", action="store_true",
                        help="Also save final model as models/dqn_model.zip (for submission)")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    run_name = (
        f"policy-{args.policy}_lr-{args.lr}_gamma-{args.gamma}_"
        f"batch-{args.batch_size}_eps{args.epsilon_start}-{args.epsilon_end}_decay-{int(args.epsilon_decay)}"
    )
    model_path = os.path.join(MODEL_DIR, f"{run_name}.zip")
    log_path = os.path.join(LOG_DIR, run_name)

    print("Training DQN on", ENV_ID)
    print("Policy:", args.policy, "| Steps:", args.steps)
    print("Model path:", model_path)
    print("Log path:", log_path)

    env = make_atari_env(ENV_ID, n_envs=args.n_envs)
    env = VecFrameStack(env, n_stack=4)

    exploration_fraction = args.epsilon_decay / args.steps

    model = DQN(
        args.policy,
        env,
        learning_rate=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        exploration_initial_eps=args.epsilon_start,
        exploration_final_eps=args.epsilon_end,
        exploration_fraction=exploration_fraction,
        buffer_size=100_000,
        learning_starts=50_000,
        target_update_interval=1_000,
        train_freq=4,
        gradient_steps=1,
        verbose=1,
        tensorboard_log=log_path,
    )

    model.learn(total_timesteps=args.steps)
    model.save(model_path)
    print("Saved:", model_path)

    if args.save_as_best:
        best_zip = os.path.join(MODEL_DIR, "dqn_model.zip")
        model.save(best_zip)
        print("Also saved as:", best_zip)

    env.close()


if __name__ == "__main__":
    main()
