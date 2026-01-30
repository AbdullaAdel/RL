import time
import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from so_arm_push_env import SoArmPushEnv

# --- ADJUST THIS FOR PLAYBACK SPEED ---
SLOW_MOTION_FACTOR = 2.0  # 1.0 = Real-time, 2.0 = Half speed
# --------------------------------------

if __name__ == "__main__":
    # 1. Initialize Environment
    env = SoArmPushEnv(xml_path="scene.xml", max_steps=200)
    
    # 2. Load Model (Force CPU to avoid GPU warnings)
    print("Loading model...")
    model = PPO.load("ppo_soarm_push", device="cpu")

    sim = env.sim
    
    # 3. Standard Reset - This gives us the initial 21-dim observation
    obs, _ = env.reset()

    # --- MANUAL GOAL OVERRIDE ---
    # Define our target
    manual_goal_xy = np.array([0.1, 0.07])
    
    # Update environment's internal goal
    env.goal_pos[:2] = manual_goal_xy
    
    # Update visual marker in the simulator
    if hasattr(env, "goal_site_id") and env.goal_site_id is not None:
        sim.model.site_pos[env.goal_site_id] = env.goal_pos
        mujoco.mj_forward(sim.model, sim.data)
        
    # FIX: Manually inject the new goal into the existing observation array
    # Since obs is [qpos(6), qvel(6), cube_pos(3), cube_vel(3), goal(3)]
    # The goal is at the very end (last 3 indices)
    obs[-3:-1] = manual_goal_xy  # Overwrite X and Y in the observation
    
    print(f"Goal set to: {env.goal_pos[:2]}")
    print(f"Observation goal updated to: {obs[-3:]}")
    # ----------------------------

    # 4. Launch Viewer and Run Loop
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        done = False
        while viewer.is_running() and not done:
            # AI predicts based on the 'obs' we just modified
            action, _ = model.predict(obs, deterministic=True)
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)

            # Log progress
            if env.step_count % 20 == 0 or info.get("success", False):
                print(f"t={env.step_count:3d} | dist_goal={info['dist_cube_goal']:.4f}")

            done = terminated or truncated
            viewer.sync()
            
            time.sleep((sim.sim_dt * sim.ctrl_steps) * SLOW_MOTION_FACTOR)

    print("\nSimulation Finished.")
    env.close()