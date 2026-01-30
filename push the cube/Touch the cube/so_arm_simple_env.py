"""Low-level MuJoCo wrapper used by the high-level Gym environments.

This repo originally referenced a `so_arm_simple_env.py` that wasn't included
in the uploaded files. This implementation is intentionally minimal but
feature-complete for PPO/DDPG style training with Stable-Baselines3.

Key features:
  - Loads MuJoCo XML (scene.xml that includes so_arm100.xml).
  - Position-control actuators: actions are joint *deltas* scaled by
    `action_scale` and applied to `data.ctrl`.
  - Fixed control rate (control_hz). Internally performs multiple mujoco
    simulation steps per control step.
  - Provides a standard 18D observation used by your envs:
      [q(6), dq(6), cube_pos(3), ee_pos(3)]
  - Exposes `model`, `data`, `sim_dt`, and `ctrl_steps` for visualization.

Assumptions (based on the provided XML):
  - Robot joints are named: Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw
  - Cube body is named: cube (geom: cube_geom)
  - End-effector site is named: ee_site
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import mujoco


@dataclass
class JointSpec:
    name: str
    joint_id: int
    qpos_adr: int
    dof_adr: int
    range_min: float
    range_max: float


class SoArmSimpleEnv:
    def __init__(
        self,
        xml_path: str = "scene.xml",
        control_hz: int = 20,
        episode_seconds: float = 10.0,
        action_scale: float = 0.05,
        joint_names: Optional[Iterable[str]] = None,
        cube_body: str = "cube",
        ee_site: str = "ee_site",
        cube_init_xy: Tuple[float, float] = (0.25, 0.10),
        cube_init_z: float = 0.02,
        cube_xy_noise: float = 0.02,
    ):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.control_hz = int(control_hz)
        self.episode_seconds = float(episode_seconds)
        self.action_scale = float(action_scale)

        self.sim_dt = float(self.model.opt.timestep)
        self.ctrl_steps = max(1, int(round((1.0 / self.control_hz) / self.sim_dt)))
        self.max_steps = int(round(self.episode_seconds * self.control_hz))

        # --- IDs ---
        self.cube_body_id = self.model.body(cube_body).id
        self.ee_site_id = self.model.site(ee_site).id

        # --- Actuated joints ---
        if joint_names is None:
            joint_names = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
        self.joints: List[JointSpec] = []
        for name in joint_names:
            j = self.model.joint(name)
            jid = j.id
            qpos_adr = int(self.model.jnt_qposadr[jid])
            dof_adr = int(self.model.jnt_dofadr[jid])
            rmin, rmax = self.model.jnt_range[jid]
            self.joints.append(JointSpec(name, jid, qpos_adr, dof_adr, float(rmin), float(rmax)))

        self.n_joints = len(self.joints)

        # Observation: q(6)+dq(6)+cube_pos(3)+ee_pos(3)
        self.n_obs = self.n_joints * 2 + 6

        # --- Reset defaults ---
        self._home_q = np.array([0.0, -1.57, 1.57, 1.57, 1.57, 0.0], dtype=np.float64)
        self._cube_init_xy = np.array(cube_init_xy, dtype=np.float64)
        self._cube_init_z = float(cube_init_z)
        self._cube_xy_noise = float(cube_xy_noise)

        # Precompute cube free-joint qpos address (7 values for free joint)
        self._cube_joint_qpos_adr = int(self.model.joint("cube").qposadr)

        self._rng = np.random.default_rng()

        # Make sure state is valid
        self.reset()

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------
    def _get_q(self) -> np.ndarray:
        q = np.empty(self.n_joints, dtype=np.float64)
        for i, js in enumerate(self.joints):
            q[i] = self.data.qpos[js.qpos_adr]
        return q

    def _get_dq(self) -> np.ndarray:
        dq = np.empty(self.n_joints, dtype=np.float64)
        for i, js in enumerate(self.joints):
            dq[i] = self.data.qvel[js.dof_adr]
        return dq

    def _set_ctrl(self, ctrl: np.ndarray) -> None:
        self.data.ctrl[: self.n_joints] = ctrl

    def _clip_to_ranges(self, q: np.ndarray) -> np.ndarray:
        out = q.copy()
        for i, js in enumerate(self.joints):
            out[i] = float(np.clip(out[i], js.range_min, js.range_max))
        return out

    def get_obs(self) -> np.ndarray:
        q = self._get_q()
        dq = self._get_dq()
        cube_pos = self.data.xpos[self.cube_body_id].copy()  # (3,)
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()  # (3,)
        obs = np.concatenate([q, dq, cube_pos, ee_pos])
        return obs.astype(np.float32)

    # ---------------------------------------------------------------------
    # Public API (used by your Gym envs)
    # ---------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)

        # --- Place cube on the table with small XY noise ---
        xy = self._cube_init_xy + self._rng.uniform(-self._cube_xy_noise, self._cube_xy_noise, size=(2,))
        x, y = float(xy[0]), float(xy[1])
        z = self._cube_init_z

        adr = self._cube_joint_qpos_adr
        # free joint: [x, y, z, qw, qx, qy, qz]
        self.data.qpos[adr : adr + 7] = np.array([x, y, z, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # --- Set robot to home pose ---
        for i, js in enumerate(self.joints):
            self.data.qpos[js.qpos_adr] = self._home_q[i]

        self.data.qvel[:] = 0.0

        # ctrl targets should match pose for a calm reset
        self._set_ctrl(self._clip_to_ranges(self._home_q))

        mujoco.mj_forward(self.model, self.data)
        return self.get_obs()

    def step(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64).reshape(self.n_joints)
        action = np.clip(action, -1.0, 1.0)

        # Delta position control around the current ctrl target (stable)
        cur_ctrl = self.data.ctrl[: self.n_joints].copy()
        target = cur_ctrl + self.action_scale * action
        target = self._clip_to_ranges(target)
        self._set_ctrl(target)

        for _ in range(self.ctrl_steps):
            mujoco.mj_step(self.model, self.data)

        return self.get_obs()
