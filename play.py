"""
Load a trained DQN model and play in the Atari environment.
Uses greedy action selection (no exploration) and renders the game.
"""
import argparse
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")
import ale_py  # registers ALE/* env ids with Gymnasium before gym.make
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env

ENV_ID = "ALE/SpaceInvaders-v5"


def parse_args():
    parser = argparse.ArgumentParser(description="Play with trained DQN agent")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model .zip (e.g. models/dqn_model.zip)")
    parser.add_argument("--n_episodes", type=int, default=5,
                        help="Number of episodes to run")
    parser.add_argument("--fps", type=float, default=30,
                        help="Rendering speed (frames per second)")
    return parser.parse_args()


def main():
    args = parse_args()
    print("Loading model:", args.model)

    env = make_atari_env(ENV_ID, n_envs=1, env_kwargs={"render_mode": "human"})
    env = VecFrameStack(env, n_stack=4)

    try:
        model = DQN.load(args.model, env=env)
    except Exception as e:
        print("Error loading model:", e)
        print("Ensure the path is correct and the model was trained on", ENV_ID)
        env.close()
        return

    print("Model loaded. Playing with greedy policy (deterministic=True).")
    delay = 1.0 / args.fps

    for ep in range(args.n_episodes):
        obs = env.reset()
        total_reward = 0.0
        print(f"\n--- Episode {ep + 1}/{args.n_episodes} ---")

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += float(reward[0])
            env.render()
            time.sleep(delay)
            if done.any():
                break

        print(f"Episode reward: {total_reward}")

    env.close()
    print("Done.")


if __name__ == "__main__":
    main()
