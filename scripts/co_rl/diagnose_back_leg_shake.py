# Quick back-leg shake diagnostic: contact force vs joint oscillation (front vs back).
# Pass criteria (plant PD 120/2 hip/AFE): AFE dq_rms < 0.5, AFE sat% = 0 — see
# docs/experiments/rl/20260615_wolf_isaac_pd_damping_shake.md
# Run: source scripts/env_rl.sh && python scripts/co_rl/diagnose_back_leg_shake.py --headless

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--settle", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args)
simulation_app = app.app

import gymnasium as gym
import numpy as np
import torch

import lab.flamingo.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

FOOT_BODIES = [
    "Foot_front_left_link",
    "Foot_front_right_link",
    "Foot_back_left_link",
    "Foot_back_right_link",
]


def _rms(vals: list[float]) -> float | None:
    if not vals:
        return None
    return float(np.sqrt(np.mean(np.square(vals))))


def _max_abs(vals: list[float]) -> float:
    if not vals:
        return 0.0
    return float(max(abs(v) for v in vals))


def main() -> None:
    if args.steps <= args.settle:
        raise ValueError(
            f"--steps ({args.steps}) must be greater than --settle ({args.settle}) "
            "to collect samples."
        )
    env_cfg = parse_env_cfg(
        "Isaac-Velocity-Flat-Wolf-v1-ppo",
        device=args.device,
        num_envs=1,
        use_fabric=True,
    )
    env_cfg.scene.num_envs = 1
    env_cfg.events.add_base_mass = None
    env_cfg.events.physics_material = None
    env_cfg.events.push_robot = None
    env_cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
    env_cfg.terminations.base_contact = None
    env_cfg.terminations.terrain_out_of_bounds = None

    env = gym.make("Isaac-Velocity-Flat-Wolf-v1-ppo", cfg=env_cfg)
    env.reset()
    robot = env.unwrapped.scene["robot"]
    contact = env.unwrapped.scene.sensors["contact_forces"]
    zero = torch.zeros((1, env.unwrapped.action_manager.total_action_dim), device=env.unwrapped.device)

    joint_pos = torch.zeros_like(robot.data.joint_pos)
    joint_vel = torch.zeros_like(robot.data.joint_vel)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    env.unwrapped.scene.write_data_to_sim()

    body_ids = [robot.body_names.index(n) for n in FOOT_BODIES if n in robot.body_names]
    foot_names = [n for n in FOOT_BODIES if n in robot.body_names]
    print(f"Foot bodies tracked: {foot_names}")

    joint_names = list(robot.joint_names)
    dq_hist: dict[str, list[float]] = {n: [] for n in joint_names}
    fz_hist: dict[str, list[float]] = {n: [] for n in foot_names}
    tau_hist: dict[str, list[float]] = {
        n: [] for n in joint_names if "AFE" in n or "KFE" in n
    }

    for step in range(args.steps):
        env.step(zero)
        if step < args.settle:
            continue
        for idx, jn in enumerate(joint_names):
            dq_hist[jn].append(float(robot.data.joint_vel[0, idx].cpu()))
        for i, bn in enumerate(foot_names):
            bid = body_ids[i]
            # net contact force magnitude on foot body
            f = contact.data.net_forces_w[0, bid].cpu().numpy()
            fz_hist[bn].append(float(np.linalg.norm(f)))
        for jn in tau_hist:
            idx = joint_names.index(jn)
            tau_hist[jn].append(float(robot.data.applied_torque[0, idx].cpu()))

    n_samples = len(next(iter(dq_hist.values()), []))
    print(f"Collected {n_samples} samples (steps {args.settle}..{args.steps - 1})")

    print("\n=== Joint velocity RMS (rad/s) — shake indicator ===")
    groups = {"front": [], "back": [], "afe": [], "kfe_back": [], "kfe_front": []}
    for jn, vals in dq_hist.items():
        rms = _rms(vals)
        if rms is None:
            continue
        if "front" in jn:
            groups["front"].append(rms)
        if "back" in jn:
            groups["back"].append(rms)
        if "AFE" in jn:
            groups["afe"].append(rms)
        if "KFE" in jn and "back" in jn:
            groups["kfe_back"].append(rms)
        if "KFE" in jn and "front" in jn:
            groups["kfe_front"].append(rms)
        if rms > 2.0 or "AFE" in jn:
            print(f"  {jn:<32} rms={rms:7.2f}  max|dq|={_max_abs(vals):7.2f}")

    print("\n=== Group means (dq RMS) ===")
    for k, v in groups.items():
        if v:
            print(f"  {k:<12} mean={np.mean(v):.2f}  max={np.max(v):.2f}")

    print("\n=== Foot contact force |F| (N) — chatter indicator ===")
    for bn, vals in fz_hist.items():
        if not vals:
            print(f"  {bn:<28} (no samples)")
            continue
        arr = np.asarray(vals, dtype=np.float64)
        mean_f = float(arr.mean())
        print(
            f"  {bn:<28} mean={mean_f:8.1f}  std={arr.std():8.1f}  "
            f"max={arr.max():8.1f}  cv={arr.std() / max(mean_f, 1.0):.2f}"
        )

    print("\n=== AFE/KFE torque (Nm) ===")
    for jn, vals in tau_hist.items():
        if not vals:
            print(f"  {jn:<32} (no samples)")
            continue
        arr = np.asarray(vals, dtype=np.float64)
        sat_frac = float(np.mean(np.abs(arr) >= 119.0))
        print(
            f"  {jn:<32} mean={arr.mean():8.1f}  std={arr.std():8.1f}  "
            f"|max|={np.abs(arr).max():8.1f}  sat%={100 * sat_frac:.0f}"
        )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
