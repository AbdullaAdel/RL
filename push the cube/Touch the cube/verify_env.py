import time
import numpy as np
import mujoco
import mujoco.viewer
from so_arm_curriculum_env import SoArmCurriculumEnv

def verify_safe_env():
    # 1. Initialize the environment
    env = SoArmCurriculumEnv(xml_path="scene.xml")
    
    print("--- Safe Arc Verification (3cm Goal Variance) ---")
    stage_names = ["DRAGGING", "PUSHING", "RANDOM (SAFE ARC)"]
    
    with mujoco.viewer.launch_passive(env.sim.model, env.sim.data) as viewer:
        for stage_idx in range(3):
            env.current_stage = stage_idx
            print(f"\nVerifying Stage {stage_idx}: {stage_names[stage_idx]}")
            
            for example in range(5):
                # --- APPLY THE NEW SAFE ARC LOGIC MANUALLY FOR VERIFICATION ---
                env.sim.reset()
                cube_id = mujoco.mj_name2id(env.sim.model, mujoco.mjtObj.mjOBJ_JOINT, "cube")
                q_addr = env.sim.model.jnt_qposadr[cube_id]
                
                if stage_idx == 0: # DRAGGING
                    cube_p = [0.30, 0.0]
                    goal_p = [0.18, 0.0]
                elif stage_idx == 1: # PUSHING
                    cube_p = [0.18, 0.0]
                    goal_p = [0.30, 0.0]
                else: # RANDOM with 0.03 (3cm) Variance
                    # 1. Cube in a safe reachable arc (18cm to 30cm)
                    r_cube = np.random.uniform(0.18, 0.30)
                    theta_cube = np.random.uniform(-0.5, 0.5) 
                    cube_p = [r_cube * np.cos(theta_cube), r_cube * np.sin(theta_cube)]
                    
                    # 2. Goal strictly within 0.03m of the cube
                    # This is your requested 0.03 change
                    r_goal_offset = 0.03 
                    theta_goal = np.random.uniform(0, 2*np.pi)
                    goal_p = [
                        cube_p[0] + r_goal_offset * np.cos(theta_goal),
                        cube_p[1] + r_goal_offset * np.sin(theta_goal)
                    ]

                # Update Simulation
                env.sim.data.qpos[q_addr:q_addr+2] = cube_p
                env.goal_pos = np.array([goal_p[0], goal_p[1], 0.02], dtype=np.float32)
                
                goal_site_id = mujoco.mj_name2id(env.sim.model, mujoco.mjtObj.mjOBJ_SITE, "push_goal")
                env.sim.model.site_pos[goal_site_id] = env.goal_pos
                mujoco.mj_forward(env.sim.model, env.sim.data)
                # -------------------------------------------------------------

                print(f"  [Ex {example+1}] Cube: {np.round(cube_p, 3)} | Goal: {np.round(goal_p, 3)}")
                
                start_time = time.time()
                while time.time() - start_time < 2.5: # 2.5s per view
                    if not viewer.is_running(): return
                    viewer.sync()
                    time.sleep(0.01)

    print("\nVerification complete.")
    env.close()

if __name__ == "__main__":
    verify_safe_env()