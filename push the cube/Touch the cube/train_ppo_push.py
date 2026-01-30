"""Train PPO to push the cube to the LEFT goal.

Run from the directory that contains:
  - scene.xml
  - so_arm100.xml
  - assets/ (mesh files)

Example:
  python3 train_ppo_push.py
  python3 visualize_ppo_push.py
"""

import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from so_arm_push_env import SoArmPushEnv


def make_env():
    
    return SoArmPushEnv(xml_path="scene.xml", max_steps=200, goal_delta_x=0.10, goal_tolerance=0.03)


if __name__ == "__main__":
    vec_env = DummyVecEnv([make_env])

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./tb_push_ppo",
    )

    checkpoint = CheckpointCallback(save_freq=50_000, save_path="./checkpoints_push", name_prefix="ppo_push")

    total_timesteps = int(os.environ.get("TOTAL_TIMESTEPS", "1000000"))
    model.learn(total_timesteps=total_timesteps, callback=checkpoint)

    model.save("ppo_soarm_push")
    print("Saved: ppo_soarm_push.zip")

    vec_env.close()
