import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from so_arm_simple_env import SoArmSimpleEnv


class SoArmPushEnv(gym.Env):
    """
    Push the cube to the LEFT goal.

    Obs (21): [q(6), dq(6), cube_pos(3), ee_pos(3), goal_pos(3)]
    Action (6): joint delta in [-1,1]
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        xml_path: str = "scene.xml",
        max_steps: int = 200,
        goal_delta_x: float = 0.10,
        goal_tolerance: float = 0.03,
    ):
        super().__init__()

        self.sim = SoArmSimpleEnv(
            xml_path=xml_path,
            control_hz=20,
            episode_seconds=max_steps / 20,
            action_scale=0.05,
        )

        self.max_steps = int(max_steps)
        self.step_count = 0

        self.goal_delta_x = float(goal_delta_x)
        self.goal_tolerance = float(goal_tolerance)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.sim.n_joints,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.sim.n_obs + 3,), dtype=np.float32)

        try:
            self.goal_site_id = self.sim.model.site("push_goal").id
        except KeyError:
            self.goal_site_id = None

        self.goal_pos = np.zeros(3, dtype=np.float32)
        self.prev_cube_x = 0.0
        self.prev_goal_dist = 0.0

    def _with_goal(self, base_obs: np.ndarray) -> np.ndarray:
        return np.concatenate([base_obs, self.goal_pos]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        base_obs = self.sim.reset(seed=seed).astype(np.float32)

        n = self.sim.n_joints
        cube_pos = base_obs[n*2:n*2+3]

        # goal is left of initial cube x
        self.goal_pos = np.array([cube_pos[0] - self.goal_delta_x, cube_pos[1], cube_pos[2]], dtype=np.float32)

        # visualize goal marker
        if self.goal_site_id is not None:
            self.sim.model.site_pos[self.goal_site_id] = self.goal_pos
            mujoco.mj_forward(self.sim.model, self.sim.data)

        self.prev_cube_x = float(cube_pos[0])
        self.prev_goal_dist = float(np.linalg.norm(cube_pos - self.goal_pos))
        self.step_count = 0

        return self._with_goal(base_obs), {}

    def step(self, action):
        base_obs = self.sim.step(action).astype(np.float32)
        self.step_count += 1

        n = self.sim.n_joints
        cube_pos = base_obs[n*2:n*2+3]
        ee_pos   = base_obs[n*2+3:n*2+6]

        dist_ee_cube = float(np.linalg.norm(ee_pos - cube_pos))
        dist_cube_goal = float(np.linalg.norm(cube_pos - self.goal_pos))

        # cube moved left if x decreased
        cube_x = float(cube_pos[0])
        delta_x_left = self.prev_cube_x - cube_x
        self.prev_cube_x = cube_x

        # contact/near-cube gate (near=1, far->0)
        gate = float(np.exp(-12.0 * dist_ee_cube))

        # goal distance progress
        progress = self.prev_goal_dist - dist_cube_goal
        self.prev_goal_dist = dist_cube_goal

        # ---- Reward terms ----
        # (1) reach: learn to go to cube
        r_reach = 1.0 - float(np.tanh(6.0 * dist_ee_cube))

        # (2) push-left: ONLY if close to cube (gate)
        # make it strong so PPO prefers pushing instead of hovering
        r_push_left = 60.0 * delta_x_left * gate

        # (3) progress toward goal (also helpful)
        r_progress = 15.0 * progress

        # (4) final shaping
        r_goal = -1.5 * dist_cube_goal

        # (5) action smoothness
        a = np.asarray(action, dtype=np.float32)
        r_act = -0.01 * float(np.sum(np.abs(a)))

        reward = r_reach + r_push_left + r_progress + r_goal + r_act

        # success
        success = dist_cube_goal < self.goal_tolerance
        if success:
            reward += 80.0

        terminated = bool(success)
        truncated = self.step_count >= self.max_steps

        info = {
            "success": success,
            "dist_ee_cube": dist_ee_cube,
            "dist_cube_goal": dist_cube_goal,
            "delta_x_left": delta_x_left,
            "gate": gate,
        }

        return self._with_goal(base_obs), reward, terminated, truncated, info
