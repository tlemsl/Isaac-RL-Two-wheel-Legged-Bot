# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Measure Wolf joint torques at q=0 with zero policy actions (PD holds target=0).

Uses current WOLF_CFG DelayedPD gains from wolf_rev01_0_0.py.
Run inside nvidia-ros2 after: source scripts/env_rl.sh

Example:
  python scripts/co_rl/measure_zero_pose_torque.py --headless --steps 500
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Wolf q=0 standing torque measurement (Isaac Lab).")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Wolf-v1-ppo")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=500, help="Total simulation steps.")
parser.add_argument(
    "--settle_steps",
    type=int,
    default=200,
    help="Steps to skip before collecting torque statistics.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Directory for CSV export (default: logs/diagnostics/q0_torque_<timestamp>).",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import lab.flamingo.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def _configure_env_for_q0_measurement(env_cfg) -> None:
    """Deterministic spawn: all joints at 0, flat origin, no domain randomization."""
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.events.add_base_mass = None
    env_cfg.events.physics_material = None
    env_cfg.events.push_robot = None
    env_cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
    env_cfg.events.reset_base.params = {
        "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
        "velocity_range": {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        },
    }
    env_cfg.commands.base_velocity.rel_standing_envs = 1.0
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.terminations.base_contact = None
    env_cfg.terminations.terrain_out_of_bounds = None
    env_cfg.episode_length_s = 60.0


def _pd_torque_estimate(q: float, dq: float, kp: float, kd: float, q_des: float = 0.0) -> float:
    return kp * (q_des - q) - kd * dq


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    _configure_env_for_q0_measurement(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    robot = env.unwrapped.scene["robot"]
    device = env.unwrapped.device
    num_actions = env.unwrapped.action_manager.total_action_dim
    zero_action = torch.zeros((args_cli.num_envs, num_actions), device=device)

    # Force exact q=0 after reset (events may still jitter base pose).
    joint_pos = torch.zeros_like(robot.data.joint_pos)
    joint_vel = torch.zeros_like(robot.data.joint_vel)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    env.unwrapped.scene.write_data_to_sim()

    tau_samples: list[np.ndarray] = []
    q_samples: list[np.ndarray] = []
    dq_samples: list[np.ndarray] = []
    base_z_samples: list[float] = []

    print(f"[measure] task={args_cli.task}, steps={args_cli.steps}, settle={args_cli.settle_steps}")
    print(f"[measure] action_dim={num_actions}, joint_names ({len(robot.joint_names)}):")
    for i, name in enumerate(robot.joint_names):
        print(f"  [{i:2d}] {name}")

    for step in range(args_cli.steps):
        env.step(zero_action)
        if step < args_cli.settle_steps:
            continue
        tau_samples.append(robot.data.applied_torque[0].detach().cpu().numpy())
        q_samples.append(robot.data.joint_pos[0].detach().cpu().numpy())
        dq_samples.append(robot.data.joint_vel[0].detach().cpu().numpy())
        base_z_samples.append(float(robot.data.root_pos_w[0, 2].detach().cpu()))

    tau = np.stack(tau_samples, axis=0)
    q = np.stack(q_samples, axis=0)
    dq = np.stack(dq_samples, axis=0)
    joint_names = list(robot.joint_names)

    # PD gains from wolf_rev01_0_0.py (hip/KFE/AFE groups).
    pd_gains = {
        "HAA": (150.0, 5.0),
        "HFE": (150.0, 5.0),
        "KFE": (300.0, 20.0),
        "AFE": (150.0, 5.0),
    }

    def _gains_for_joint(name: str) -> tuple[float, float]:
        for key, gains in pd_gains.items():
            if key in name:
                return gains
        return (0.0, 0.0)

    print("\n=== q=0 standing torque (zero actions, current WOLF_CFG PD) ===")
    print(f"base_link z: mean={np.mean(base_z_samples):.4f}  std={np.std(base_z_samples):.4f}")
    print(
        f"{'joint':<30} {'q_mean':>8} {'dq_rms':>8} {'tau_mean':>10} {'tau_std':>8} "
        f"{'|tau|max':>8} {'tau_pd':>10}"
    )
    rows = []
    for i, name in enumerate(joint_names):
        kp, kd = _gains_for_joint(name)
        q_mean = float(q[:, i].mean())
        dq_rms = float(np.sqrt(np.mean(dq[:, i] ** 2)))
        tau_mean = float(tau[:, i].mean())
        tau_std = float(tau[:, i].std())
        tau_max = float(np.abs(tau[:, i]).max())
        tau_pd = _pd_torque_estimate(q_mean, dq_rms, kp, kd)
        print(
            f"{name:<30} {q_mean:8.4f} {dq_rms:8.4f} {tau_mean:10.2f} {tau_std:8.2f} "
            f"{tau_max:8.2f} {tau_pd:10.2f}"
        )
        rows.append([name, q_mean, dq_rms, tau_mean, tau_std, tau_max, tau_pd, kp, kd])

    out_dir = args_cli.output_dir
    if out_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("logs", "diagnostics", f"q0_torque_{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    summary_path = os.path.join(out_dir, "q0_torque_summary.csv")
    summary = np.array([[r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]] for r in rows])
    header = "q_mean,dq_rms,tau_mean,tau_std,tau_abs_max,tau_pd_est,kp,kd"
    np.savetxt(
        summary_path,
        summary,
        delimiter=",",
        header=header,
        comments="",
        fmt="%.6f",
    )
    with open(os.path.join(out_dir, "joint_names.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(joint_names) + "\n")

    ts_path = os.path.join(out_dir, "timeseries.csv")
    ts_header = ["step"] + [f"tau_{n}" for n in joint_names] + [f"q_{n}" for n in joint_names]
    steps = np.arange(args_cli.settle_steps, args_cli.steps)[:, None]
    ts_data = np.hstack([steps, tau, q])
    np.savetxt(ts_path, ts_data, delimiter=",", header=",".join(ts_header), comments="")
    print(f"\n[measure] Saved summary: {summary_path}")
    print(f"[measure] Saved timeseries: {ts_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
