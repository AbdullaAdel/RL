import torch
import mujoco
import mujoco.viewer
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from so_arm_curriculum_env import SoArmCurriculumEnv

def render_preview(model, stage_idx, stage_names, num_episodes=2):
    """Opens a window for a limited time to show current performance."""
    test_env = SoArmCurriculumEnv()
    test_env.current_stage = stage_idx
    print(f"\n>>> VISUAL PREVIEW: {stage_names[stage_idx]} <<<")
    
    # launch_passive opens the window in a separate thread
    with mujoco.viewer.launch_passive(test_env.sim.model, test_env.sim.data) as viewer:
        for ep in range(num_episodes):
            obs, _ = test_env.reset()
            done = False
            step_limit = 0 
            
            while not done and viewer.is_running() and step_limit < 400:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = test_env.step(action)
                
                viewer.sync()
                time.sleep(0.01) # Faster playback for the preview
                
                done = terminated or truncated
                step_limit += 1
            
            if not viewer.is_running():
                break
    
    # Explicitly close to release the MuJoCo context
    test_env.close()
    print(">>> Preview Closed. Resuming Training/Eval...")

def evaluate_success(model, env, n_episodes=15):
    """Checks the success rate without visuals to keep training fast."""
    print("Checking success rate...")
    successes = 0
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = env.step(action) 
            done = dones[0]
            if infos[0].get("success", False):
                successes += 1
                done = True # VecEnv auto-resets, but we break the loop
    return successes / n_episodes

if __name__ == "__main__":
    env = DummyVecEnv([lambda: SoArmCurriculumEnv()])
    
    # Initialize model with a standard entropy coefficient
    model = PPO("MlpPolicy", env, learning_rate=2e-4, ent_coef=0.01, verbose=1, device="cpu")

    stage_names = ["TOUCH", "PUSH", "DRAG", "RANDOM"]
    
    for stage_idx in range(4):
        env.envs[0].current_stage = stage_idx
        print(f"\n{'='*50}\nSTAGING: {stage_names[stage_idx]}\n{'='*50}")

        success_streak = 0
        required_streak = 2
        
        # --- NEW: FAILURE TRACKING FOR EXPLORATION ---
        consecutive_failures = 0 
        
        while True:
            # 1. Adaptive Entropy Hack
            # If the robot is consistently failing (SR < 5%), triple the exploration
            if consecutive_failures >= 2:
                print(">>> LOCAL MINIMUM DETECTED: Jacking up Entropy for exploration! <<<")
                model.ent_coef = 0.05  # Force high randomness
            else:
                model.ent_coef = 0.01  # Normal refined training
            
            # 2. Train
            print(f"\nTraining for 70,000 steps... (Streak: {success_streak}/{required_streak})")
            model.learn(total_timesteps=70000, reset_num_timesteps=False)
            
            # 3. Preview
            render_preview(model, stage_idx, stage_names)
            
            # 4. Eval
            sr = evaluate_success(model, env)
            print(f"[{stage_names[stage_idx]}] Current Success Rate: {sr*100:.1f}%")
            
            # --- UPDATED FAILURE LOGIC ---
            if sr < 0.05:  # If the robot is effectively doing nothing or just dying
                consecutive_failures += 1
            else:
                consecutive_failures = 0 # Reset if we see even a little progress
            
            # --- STREAK LOGIC ---
            if sr >= 0.90:
                success_streak += 1
                print(f"STREAK INCREASED: {success_streak}/{required_streak}")
            else:
                if success_streak > 0:
                    print(f"STREAK BROKEN! Resetting to 0.")
                success_streak = 0

            if success_streak >= required_streak:
                print(f"PROMOTED! {required_streak} consecutive successes achieved.")
                model.save(f"ppo_soarm_stage_{stage_idx}")
                break
            else:
                print(f"Continuing training in {stage_names[stage_idx]}...")

    model.save("ppo_soarm_final")
    env.close()