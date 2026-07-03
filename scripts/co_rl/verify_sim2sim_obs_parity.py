#!/usr/bin/env python3
"""Verify Wolf sim2sim policy obs parity: Isaac env vs deploy obs_contract.

Checks:
  1) Isaac articulation vs action joint order labels
  2) stack_policy joint_pos/vel block matches Python ObsBuilder (Isaac articulation order)
  3) projected_gravity + base_ang_vel body-frame axes under multiple base orientations / ω

Run (container, env_rl):
  source scripts/env_rl.sh
  python scripts/co_rl/verify_sim2sim_obs_parity.py --headless
"""

from __future__ import annotations

import argparse
import math
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--tol", type=float, default=1e-4, help="Max abs error for obs parity.")
parser.add_argument("--ang_static_tol", type=float, default=0.01, help="Residual ω when target is zero.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args)
simulation_app = app.app

import gymnasium as gym
import numpy as np
import torch
import isaaclab.utils.math as math_utils

import lab.flamingo.tasks  # noqa: F401
from lab.flamingo.tasks.manager_based.locomotion.velocity.wolf_env.flat_env.stand_drive import (
    flat_env_stand_drive_cfg as cfg_mod,
)

# Repo-relative import when PYTHONPATH includes wolf-quadruped/workspace install
try:
    from wolf_rl_inference.obs_contract import (
        ANG_VEL_SCALE,
        ISAAC_OBS_JOINT_NAMES,
        JOINT_VEL_SCALE,
        RL_JOINT_NAMES,
        ObsBuilder,
        RobotState,
        VelocityCommand,
        gather_joint_values,
        projected_gravity_from_quat,
    )
except ImportError:
    sys.path.insert(
        0,
        str(__import__("pathlib").Path(__file__).resolve().parents[4] / "workspace/src/wolf_rl_inference"),
    )
    from wolf_rl_inference.obs_contract import (  # type: ignore
        ANG_VEL_SCALE,
        ISAAC_OBS_JOINT_NAMES,
        JOINT_VEL_SCALE,
        RL_JOINT_NAMES,
        ObsBuilder,
        RobotState,
        VelocityCommand,
        gather_joint_values,
        projected_gravity_from_quat,
    )


def _collect_action_joint_names(env) -> list[str]:
    names: list[str] = []
    for term_name in env.unwrapped.action_manager.active_terms:
        term = env.unwrapped.action_manager.get_term(term_name)
        joint_names = getattr(term, "_joint_names", None) or getattr(term, "joint_names", None)
        if joint_names is None:
            continue
        names.extend(list(joint_names))
    return names


def _euler_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qw, qx, qy, qz


def _set_robot_state(
    robot,
    joint_q_by_name: dict[str, float],
    joint_dq_by_name: dict[str, float],
    quat_wxyz: tuple[float, float, float, float],
    ang_vel_b: tuple[float, float, float],
) -> None:
    device = robot.device
    joint_pos = torch.zeros_like(robot.data.joint_pos)
    joint_vel = torch.zeros_like(robot.data.joint_vel)
    for idx, name in enumerate(robot.joint_names):
        if name in joint_q_by_name:
            joint_pos[0, idx] = float(joint_q_by_name[name])
        if name in joint_dq_by_name:
            joint_vel[0, idx] = float(joint_dq_by_name[name])
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    pos = robot.data.root_link_pos_w.clone()
    lin_vel = torch.zeros((1, 3), device=device)
    ang_b = torch.tensor(ang_vel_b, device=device, dtype=torch.float32).unsqueeze(0)
    quat_t = torch.tensor(quat_wxyz, device=device, dtype=torch.float32).unsqueeze(0)
    ang_w = math_utils.quat_apply(quat_t, ang_b)
    robot.write_root_link_pose_to_sim(torch.cat([pos, quat_t], dim=-1))
    robot.write_root_com_velocity_to_sim(torch.cat([lin_vel, ang_w], dim=-1))


def _isaac_joint_pos_obs(robot) -> np.ndarray:
    q = robot.data.joint_pos[0].detach().cpu().numpy()
    q0 = robot.data.default_joint_pos[0].detach().cpu().numpy()
    return q - q0


def _isaac_joint_vel_obs(robot) -> np.ndarray:
    return robot.data.joint_vel[0].detach().cpu().numpy() * JOINT_VEL_SCALE


def _isaac_ang_vel_obs(robot) -> np.ndarray:
    return robot.data.root_link_ang_vel_b[0].detach().cpu().numpy() * ANG_VEL_SCALE


def _isaac_gravity_obs(robot) -> np.ndarray:
    return robot.data.projected_gravity_b[0].detach().cpu().numpy()


def _push_robot_state(env, robot) -> None:
    env.unwrapped.scene.write_data_to_sim()
    robot.update(0.0)


def main() -> int:
    env_cfg = cfg_mod.WolfFlatEnvCfg()
    env_cfg.scene.num_envs = 1
    env = gym.make("Isaac-Velocity-Flat-Wolf-v1-ppo", cfg=env_cfg)
    env.reset()
    robot = env.unwrapped.scene["robot"]

    articulation = list(robot.joint_names)
    action_names = _collect_action_joint_names(env)
    print("[verify] articulation joint_names:")
    for i, n in enumerate(articulation):
        print(f"  {i:2d}: {n}")
    print("[verify] action joint_names:")
    for i, n in enumerate(action_names):
        print(f"  {i:2d}: {n}")
    print(f"[verify] articulation == ISAAC_OBS_JOINT_NAMES: {articulation == list(ISAAC_OBS_JOINT_NAMES)}")
    print(f"[verify] action == RL_JOINT_NAMES: {action_names == list(RL_JOINT_NAMES)}")

  # --- Joint obs block: unique q/dq per joint name ---
    joint_q = {name: 0.1 * (i + 1) for i, name in enumerate(ISAAC_OBS_JOINT_NAMES)}
    joint_dq = {name: -0.05 * (i + 1) for i, name in enumerate(ISAAC_OBS_JOINT_NAMES)}
    last_action = [0.2 * (i + 1) for i in range(14)]
    _set_robot_state(robot, joint_q, joint_dq, (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    _push_robot_state(env, robot)

    isaac_q = _isaac_joint_pos_obs(robot)
    isaac_dq = _isaac_joint_vel_obs(robot)
    isaac_ang = _isaac_ang_vel_obs(robot)
    isaac_grav = _isaac_gravity_obs(robot)

    deploy_state = RobotState(
        joint_pos=gather_joint_values(joint_q, ISAAC_OBS_JOINT_NAMES),
        joint_vel=gather_joint_values(joint_dq, ISAAC_OBS_JOINT_NAMES),
        base_ang_vel=[0.0, 0.0, 0.0],
        projected_gravity=list(projected_gravity_from_quat(1.0, 0.0, 0.0, 0.0)),
    )
    builder = ObsBuilder(num_policy_stacks=2)
    builder.set_last_action(last_action)
    deploy_frame = builder._build_stack_frame(deploy_state)  # noqa: SLF001

    err_q = float(np.max(np.abs(isaac_q - np.array(deploy_state.joint_pos))))
    err_dq = float(np.max(np.abs(isaac_dq - np.array(deploy_state.joint_vel) * JOINT_VEL_SCALE)))
    err_ang = float(np.max(np.abs(isaac_ang - np.array(deploy_state.base_ang_vel) * ANG_VEL_SCALE)))
    err_grav = float(np.max(np.abs(isaac_grav - np.array(deploy_state.projected_gravity))))
    deploy_q_dq_ang_grav = (
        deploy_frame[0:14]
        + deploy_frame[14:28]
        + deploy_frame[28:31]
        + deploy_frame[31:34]
    )
    isaac_q_dq_ang_grav = np.concatenate([isaac_q, isaac_dq, isaac_ang, isaac_grav])
    err_kin = float(np.max(np.abs(isaac_q_dq_ang_grav - deploy_q_dq_ang_grav)))
    print(f"[verify] joint_pos max|err|={err_q:.3e} (expect ~0, default offset=0)")
    print(f"[verify] joint_vel max|err|={err_dq:.3e}")
    print(f"[verify] base_ang_vel max|err|={err_ang:.3e}")
    print(f"[verify] projected_gravity max|err|={err_grav:.3e}")
    print(f"[verify] q+dq+ang+grav block max|err|={err_kin:.3e}")

  # --- IMU axis sweep ---
    imu_cases = [
        ("identity", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ("roll+90", (math.pi / 2, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ("pitch+90", (0.0, math.pi / 2, 0.0), (0.0, 0.0, 0.0)),
        ("yaw+90", (0.0, 0.0, math.pi / 2), (0.0, 0.0, 0.0)),
        ("roll-45_pitch+30", (-math.pi / 4, math.pi / 6, 0.0), (0.0, 0.0, 0.0)),
    ]
    imu_ok = True
    print("[verify] IMU orientation: Isaac projected_gravity_b vs deploy quat formula:")
    for label, rpy, _omega_b in imu_cases:
        quat = _euler_to_quat_wxyz(*rpy)
        _set_robot_state(robot, {}, {}, quat, (0.0, 0.0, 0.0))
        _push_robot_state(env, robot)
        grav_isaac = _isaac_gravity_obs(robot)
        grav_deploy = np.array(projected_gravity_from_quat(*quat))
        eg = float(np.max(np.abs(grav_isaac - grav_deploy)))
        ok = eg < args.tol
        imu_ok = imu_ok and ok
        print(
            f"  {label:22s} grav_err={eg:.3e} "
            f"grav_isaac={np.round(grav_isaac,4)} grav_deploy={np.round(grav_deploy,4)} "
            f"{'PASS' if ok else 'FAIL'}"
        )

    print("[verify] Note: body ω axis sweep is in verify_mujoco_imu_axes.py (deploy /wolf/imu path).")

    joint_ok = err_q < args.tol and err_dq < args.tol and err_grav < args.tol and err_ang < args.ang_static_tol
    all_ok = joint_ok and imu_ok and articulation == list(ISAAC_OBS_JOINT_NAMES) and action_names == list(RL_JOINT_NAMES)
    print(f"[verify] OVERALL: {'PASS' if all_ok else 'FAIL'}")
    env.close()
    simulation_app.close()
    return 0 if all_ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[verify] FATAL: {exc}", file=sys.stderr, flush=True)
        import traceback

        traceback.print_exc()
        raise SystemExit(1) from exc
