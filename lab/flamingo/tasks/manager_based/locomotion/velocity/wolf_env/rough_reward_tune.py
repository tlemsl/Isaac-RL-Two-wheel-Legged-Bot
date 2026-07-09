"""Rough-terrain reward tuning profiles (watcher-driven)."""

from __future__ import annotations

import json
import os
from pathlib import Path


def _profile_path() -> Path:
    env = os.environ.get("WOLF_ROUGH_REWARD_PROFILE")
    if env:
        return Path(env)
    # repo root: .../wolf-quadruped/docs/experiments/artifacts/wolf_rough_reward_profile.json
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "docs" / "experiments" / "artifacts" / "wolf_rough_reward_profile.json"
        if candidate.is_file():
            return candidate
    return Path("/workspace/Documents/wolf-quadruped/docs/experiments/artifacts/wolf_rough_reward_profile.json")


def load_profile_id() -> int:
    path = _profile_path()
    if not path.is_file():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return int(data.get("profile", 0))
    except (json.JSONDecodeError, TypeError, ValueError):
        return 0


def apply_rough_reward_profile(env_cfg) -> int:
    """Mutate env_cfg rewards/events in-place. Returns profile id."""
    profile = load_profile_id()
    if profile <= 0:
        return profile

    rewards = env_cfg.rewards
    if profile >= 1:
        # Survival-first: less harsh termination/contact, stronger tracking, easier terrain.
        rewards.termination_penalty.weight = -100.0
        rewards.undesired_contacts.weight = -10.0
        rewards.undesired_contacts.params["sensor_cfg"].body_names = ["base_link"]
        rewards.track_lin_vel_xy_exp.weight = 5.0
        rewards.track_ang_vel_z_exp.weight = 3.5
        rewards.action_rate_l2.weight = -0.01
        rewards.dof_acc_l2.weight = -5.0e-7
        env_cfg.events.push_robot = None
        env_cfg.scene.terrain.max_init_terrain_level = 1

    if profile >= 2:
        rewards.termination_penalty.weight = -50.0
        rewards.undesired_contacts.weight = -5.0
        rewards.track_lin_vel_xy_exp.weight = 6.0
        rewards.track_ang_vel_z_exp.weight = 4.0
        rewards.action_rate_l2.weight = -0.005
        rewards.dof_torques_l2.weight = -5.0e-6
        rewards.joint_deviation_hip_front.weight = -0.5
        rewards.joint_deviation_hip_back.weight = -1.0
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (-0.75, 0.75)

    return profile
