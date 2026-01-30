import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from so_arm_simple_env import SoArmSimpleEnv

class SoArmCurriculumEnv(gym.Env):
    def __init__(self, xml_path="scene.xml", max_steps=400, goal_tolerance=0.04):
        super().__init__()
        # Ensure the simulation setup is initialized
        self.sim = SoArmSimpleEnv(xml_path=xml_path, control_hz=20, action_scale=0.05)
        self.n_joints = self.sim.model.nu 
        self.max_steps = max_steps
        self.goal_tolerance = goal_tolerance
        
        # 0: Touch, 1: Push, 2: Drag, 3: Random
        self.current_stage = 0 
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32)
        # Obs (21): [q(6), dq(6), cube_pos(3), ee_to_cube(3), cube_to_goal(3)]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32)

        self.goal_pos = np.zeros(3, dtype=np.float32)
        self.prev_goal_dist = 0.0
        self.step_count = 0

    def _get_obs(self):
        q, dq = self.sim.data.qpos[:6], self.sim.data.qvel[:6]
        cube_id = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_JOINT, "cube")
        cube_pos = self.sim.data.qpos[self.sim.model.jnt_qposadr[cube_id] : self.sim.model.jnt_qposadr[cube_id]+3]
        ee_pos = self.sim.data.site_xpos[mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")]
        
        return np.concatenate([q, dq, cube_pos, cube_pos - ee_pos, self.goal_pos - cube_pos]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset(seed=seed)
        
        # --- 1. INITIAL JOINT NOISE ---
        # Forces the robot to start from slightly different angles
        for i in range(self.n_joints):
            noise = np.random.uniform(-0.05, 0.05)
            self.sim.data.qpos[i] += noise
        
        # --- 2. CENTERED STAGE COORDINATES ---
        # Workspace Rectangle: X [0.2, 0.4], Y [-0.15, 0.15]
        if self.current_stage == 0:   # TOUCH (Centered)
            cube_p, goal_p = [0.3, 0.0], [0.3, 0.0]
        elif self.current_stage == 1: # PUSH (Straight line forward)
            cube_p, goal_p = [0.25, 0.0], [0.35, 0.0]
        elif self.current_stage == 2: # DRAG (Straight line backward)
            cube_p, goal_p = [0.35, 0.0], [0.25, 0.0]
        else:                         # RANDOM (Full Rectangle)
            cube_x = np.random.uniform(0.2, 0.4)
            cube_y = np.random.uniform(-0.15, 0.15)
            cube_p = [cube_x, cube_y]
            
            # Goal within 10cm, clipped to the rectangle bounds
            goal_x = np.clip(cube_x + np.random.uniform(-0.1, 0.1), 0.2, 0.4)
            goal_y = np.clip(cube_y + np.random.uniform(-0.1, 0.1), -0.15, 0.15)
            goal_p = [goal_x, goal_y]

        # Apply to simulation
        cube_id = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_JOINT, "cube")
        q_addr = self.sim.model.jnt_qposadr[cube_id]
        self.sim.data.qpos[q_addr:q_addr+2] = cube_p
        self.goal_pos = np.array([goal_p[0], goal_p[1], 0.02], dtype=np.float32)

        # Update visual goal site
        sid = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_SITE, "push_goal")
        self.sim.model.site_pos[sid] = self.goal_pos
        
        self.prev_goal_dist = np.linalg.norm(np.array(cube_p) - np.array(goal_p))
        self.step_count = 0
        self.contact_bonus_accumulator = 0.0 
        
        mujoco.mj_forward(self.sim.model, self.sim.data)
        return self._get_obs(), {}


    def step(self, action):
        self.sim.step(action)
        self.step_count += 1
        obs = self._get_obs()
        
        ee_pos = self.sim.data.site_xpos[mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")]
        dist_ee_cube = np.linalg.norm(obs[15:18])
        dist_cube_goal = np.linalg.norm(obs[18:21])
        
        # 2. GLOBAL KILL SWITCH: Floor Collision
        if ee_pos[2] < 0.015:
            return obs, -500.0, True, False, {"success": False}

        # Check for active physical contact
        is_touching = False
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            n1, n2 = self.sim.model.geom(contact.geom1).name, self.sim.model.geom(contact.geom2).name
            if ("cube" in n1 or "cube" in n2) and any(x in n1 or x in n2 for x in ["jaw", "finger", "claw"]):
                is_touching = True; break

        # --- 3. PROGRESSIVE CONTACT REWARD ---
        # Reward actively touching the cube, but weight it by progress
        if is_touching:
            # Small incremental bonus for maintaining contact
            self.contact_bonus_accumulator += 0.1 
        else:
            self.contact_bonus_accumulator *= 0.9 # Decay if contact is lost

        success = False
        reward = 0.0
        r_reach = float(1.0 - np.tanh(10.0 * dist_ee_cube))

        if self.current_stage == 0:
            success = is_touching
            reward = r_reach + self.contact_bonus_accumulator
        else:
            success = dist_cube_goal < self.goal_tolerance
            progress = self.prev_goal_dist - dist_cube_goal
            self.prev_goal_dist = dist_cube_goal
            
            # r_move is significantly boosted if actively touching
            multiplier = 2.0 if is_touching else 1.0
            r_move = (500.0 * progress * multiplier) if dist_ee_cube < 0.07 else 0.0
            
            # Combine all factors
            reward = float(r_reach + r_move + self.contact_bonus_accumulator)

        if success:
            reward += 150.0

        return obs, float(reward), bool(success), self.step_count >= self.max_steps, {"success": success}