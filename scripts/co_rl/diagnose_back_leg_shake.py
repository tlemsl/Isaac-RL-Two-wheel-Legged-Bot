# Quick back-leg shake diagnostic: contact force vs joint oscillation (front vs back).
# Pass criteria (plant PD 120/2 hip/AFE): AFE dq_rms < 0.5, AFE sat% = 0 — see
# docs/experiments/rl/20260615_wolf_isaac_pd_damping_shake.md
# Run: source scripts/env_rl.sh && python scripts/co_rl/diagnose_back_leg_shake.py --headless
# Foot heights + standing spawn:
#   python scripts/co_rl/diagnose_back_leg_shake.py --headless --pose zero
#   python scripts/co_rl/diagnose_back_leg_shake.py --headless --pose standing

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

# Synced with workspace/src/wolf_mujoco_sim/wolf_mujoco_sim/initial_pose.py
WOLF_ZERO_JOINT_Q: dict[str, float] = {
    "HAA_front_left_joint": 0.0,
    "HFE_front_left_joint": 0.0,
    "KFE_front_left_joint": 0.0,
    "HAA_front_right_joint": 0.0,
    "HFE_front_right_joint": 0.0,
    "KFE_front_right_joint": 0.0,
    "HAA_back_left_joint": 0.0,
    "HFE_back_left_joint": 0.0,
    "KFE_back_left_joint": 0.0,
    "AFE_back_left_joint": 0.0,
    "HAA_back_right_joint": 0.0,
    "HFE_back_right_joint": 0.0,
    "KFE_back_right_joint": 0.0,
    "AFE_back_right_joint": 0.0,
}

WOLF_STANDING_JOINT_Q: dict[str, float] = {
    "HAA_front_left_joint": -0.0231,
    "HFE_front_left_joint": 0.3946,
    "KFE_front_left_joint": -0.2001,
    "HAA_front_right_joint": 0.0231,
    "HFE_front_right_joint": 0.3925,
    "KFE_front_right_joint": -0.1970,
    "HAA_back_left_joint": -0.0951,
    "HFE_back_left_joint": 0.4503,
    "KFE_back_left_joint": 0.1231,
    "AFE_back_left_joint": -0.0208,
    "HAA_back_right_joint": 0.0934,
    "HFE_back_right_joint": 0.4254,
    "KFE_back_right_joint": 0.1030,
    "AFE_back_right_joint": -0.0220,
}

POSE_TABLE = {
    "zero": WOLF_ZERO_JOINT_Q,
    "standing": WOLF_STANDING_JOINT_Q,
}

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--settle", type=int, default=100)
parser.add_argument(
    "--pose",
    type=str,
    choices=sorted(POSE_TABLE.keys()),
    default="zero",
    help="Joint spawn pose: URDF zero or MuJoCo-calibrated WOLF_STANDING_JOINT_Q.",
)
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


def _quat_wxyz_to_roll_pitch(quat_wxyz: np.ndarray) -> tuple[float, float]:
    """Return roll, pitch [rad] from Isaac root quaternion (w, x, y, z)."""
    w, x, y, z = [float(v) for v in quat_wxyz]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    return roll, pitch


def _apply_pose_q(robot, pose_q: dict[str, float]) -> None:
    joint_pos = torch.zeros_like(robot.data.joint_pos)
    for idx, name in enumerate(robot.joint_names):
        if name in pose_q:
            joint_pos[0, idx] = float(pose_q[name])
    joint_vel = torch.zeros_like(robot.data.joint_vel)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)


def _build_hold_actions(env, pose_q: dict[str, float], scale: float = 0.5) -> torch.Tensor:
    """Policy actions so PD target q_des matches pose_q (JointPositionAction scale only)."""
    unwrapped = env.unwrapped
    values: list[float] = []
    for term_name in unwrapped.action_manager.active_terms:
        term = unwrapped.action_manager.get_term(term_name)
        joint_names = getattr(term, "_joint_names", None) or getattr(term, "joint_names", None)
        if joint_names is None:
            continue
        for jn in joint_names:
            values.append(float(pose_q.get(jn, 0.0)) / scale)
    return torch.tensor([values], device=unwrapped.device, dtype=torch.float32)


def _foot_heights_world(robot, foot_names: list[str], body_ids: list[int]) -> dict[str, float]:
    heights: dict[str, float] = {}
    for name, bid in zip(foot_names, body_ids):
        heights[name] = float(robot.data.body_pos_w[0, bid, 2].cpu())
    return heights


def _resolve_contact_body_ids(contact, foot_names: list[str]) -> dict[str, int]:
    """Map foot link name -> index into contact_sensor.data.net_forces_w (not articulation body id)."""
    out: dict[str, int] = {}
    for name in foot_names:
        ids, _ = contact.find_bodies(name)
        if len(ids) != 1:
            raise RuntimeError(f"contact.find_bodies({name!r}) -> {ids} (expected exactly one body)")
        out[name] = int(ids[0])
    return out


def _contact_force_vec(contact, contact_body_id: int) -> np.ndarray:
    return contact.data.net_forces_w[0, contact_body_id].detach().cpu().numpy()


def _print_contact_index_audit(
    robot,
    contact,
    foot_names: list[str],
    robot_body_ids: list[int],
    contact_body_ids: dict[str, int],
) -> None:
    print("\n=== Contact sensor index audit ===")
    print(
        "  net_forces_w is indexed by ContactSensor body id, NOT ArticulationData body id."
    )
    print(f"  contact sensor tracks {len(contact.body_names)} bodies; robot has {len(robot.body_names)} bodies")
    print(f"  {'foot':<28} {'robot_bid':>9} {'contact_id':>10} {'match':>6}")
    for i, name in enumerate(foot_names):
        rbid = robot_body_ids[i]
        cid = contact_body_ids[name]
        print(f"  {name:<28} {rbid:9d} {cid:10d} {'yes' if rbid == cid else 'NO':>6}")

    print("\n  Force compare (wrong=robot_bid, correct=contact_id) after settle window:")
    print(f"  {'foot':<28} {'|F| wrong':>10} {'|F| correct':>12} {'Fz correct':>12}")
    for i, name in enumerate(foot_names):
        rbid = robot_body_ids[i]
        cid = contact_body_ids[name]
        f_wrong = contact.data.net_forces_w[0, rbid].detach().cpu().numpy()
        f_correct = _contact_force_vec(contact, cid)
        print(
            f"  {name:<28} {np.linalg.norm(f_wrong):10.2f} {np.linalg.norm(f_correct):12.2f} "
            f"{f_correct[2]:12.2f}"
        )

    print("\n  All contact sensor bodies with |F| > 1 N (correct indexing):")
    any_hit = False
    for cid, name in enumerate(contact.body_names):
        f = _contact_force_vec(contact, cid)
        mag = float(np.linalg.norm(f))
        if mag > 1.0:
            any_hit = True
            print(f"    [{cid:3d}] {name:<32} |F|={mag:7.1f} N  Fz={f[2]:7.1f} N")
    if not any_hit:
        print("    (none)")


def _print_foot_heights(
    title: str,
    heights: dict[str, float],
    contact_forces: dict[str, float] | None = None,
    contact_fz: dict[str, float] | None = None,
) -> None:
    vals = list(heights.values())
    spread = max(vals) - min(vals) if vals else 0.0
    print(f"\n=== {title} ===")
    print(f"  foot link-origin z spread (max-min): {spread * 1000.0:.2f} mm")
    print("  (link origin z != sole height; compare trends only)")
    for name in FOOT_BODIES:
        if name not in heights:
            continue
        extra = ""
        if contact_forces is not None:
            extra = f"  |F|={contact_forces.get(name, 0.0):7.1f} N"
        if contact_fz is not None:
            extra += f"  Fz={contact_fz.get(name, 0.0):7.1f} N"
        print(f"  {name:<28} z={heights[name]:8.4f} m{extra}")


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

    pose_q = POSE_TABLE[args.pose]
    print(f"Spawn pose: {args.pose} (PD target = spawn q via hold actions)")

    env = gym.make("Isaac-Velocity-Flat-Wolf-v1-ppo", cfg=env_cfg)
    env.reset()
    robot = env.unwrapped.scene["robot"]
    contact = env.unwrapped.scene.sensors["contact_forces"]
    hold_action = _build_hold_actions(env, pose_q)

    _apply_pose_q(robot, pose_q)
    env.unwrapped.scene.write_data_to_sim()

    body_ids = [robot.body_names.index(n) for n in FOOT_BODIES if n in robot.body_names]
    foot_names = [n for n in FOOT_BODIES if n in robot.body_names]
    contact_body_ids = _resolve_contact_body_ids(contact, foot_names)
    print(f"Foot bodies tracked: {foot_names}")

    joint_names = list(robot.joint_names)
    dq_hist: dict[str, list[float]] = {n: [] for n in joint_names}
    fz_hist: dict[str, list[float]] = {n: [] for n in foot_names}
    fz_comp_hist: dict[str, list[float]] = {n: [] for n in foot_names}
    foot_z_hist: dict[str, list[float]] = {n: [] for n in foot_names}
    tau_hist: dict[str, list[float]] = {
        n: [] for n in joint_names if "AFE" in n or "KFE" in n
    }
    base_z_hist: list[float] = []
    base_roll_hist: list[float] = []
    base_pitch_hist: list[float] = []

    env.step(hold_action)
    env.unwrapped.scene.update(dt=env.unwrapped.physics_dt)
    _print_foot_heights(
        "Foot body z after spawn (1 step)",
        _foot_heights_world(robot, foot_names, body_ids),
        contact_forces={
            bn: float(np.linalg.norm(_contact_force_vec(contact, contact_body_ids[bn])))
            for bn in foot_names
        },
        contact_fz={bn: float(_contact_force_vec(contact, contact_body_ids[bn])[2]) for bn in foot_names},
    )
    quat = robot.data.root_quat_w[0].cpu().numpy()
    roll, pitch = _quat_wxyz_to_roll_pitch(quat)
    print(
        f"  base_link z={float(robot.data.root_pos_w[0, 2].cpu()):.4f} m  "
        f"roll={math.degrees(roll):+.2f} deg  pitch={math.degrees(pitch):+.2f} deg"
    )

    for step in range(args.steps):
        env.step(hold_action)
        if step < args.settle:
            continue
        for idx, jn in enumerate(joint_names):
            dq_hist[jn].append(float(robot.data.joint_vel[0, idx].cpu()))
        for i, bn in enumerate(foot_names):
            cid = contact_body_ids[bn]
            f = _contact_force_vec(contact, cid)
            fz_hist[bn].append(float(np.linalg.norm(f)))
            fz_comp_hist[bn].append(float(f[2]))
            foot_z_hist[bn].append(float(robot.data.body_pos_w[0, body_ids[i], 2].cpu()))
        for jn in tau_hist:
            idx = joint_names.index(jn)
            tau_hist[jn].append(float(robot.data.applied_torque[0, idx].cpu()))
        base_z_hist.append(float(robot.data.root_pos_w[0, 2].cpu()))
        roll, pitch = _quat_wxyz_to_roll_pitch(robot.data.root_quat_w[0].cpu().numpy())
        base_roll_hist.append(roll)
        base_pitch_hist.append(pitch)

    n_samples = len(next(iter(dq_hist.values()), []))
    print(f"Collected {n_samples} samples (steps {args.settle}..{args.steps - 1})")

    mean_heights = {bn: float(np.mean(foot_z_hist[bn])) for bn in foot_names}
    mean_contact = {bn: float(np.mean(fz_hist[bn])) for bn in foot_names}
    mean_fz = {bn: float(np.mean(fz_comp_hist[bn])) for bn in foot_names}
    _print_foot_heights(
        f"Foot body z mean (steps {args.settle}..{args.steps - 1})",
        mean_heights,
        contact_forces=mean_contact,
        contact_fz=mean_fz,
    )
    print(
        f"  base_link z mean={np.mean(base_z_hist):.4f} m  "
        f"roll mean={math.degrees(np.mean(base_roll_hist)):+.2f} deg  "
        f"pitch mean={math.degrees(np.mean(base_pitch_hist)):+.2f} deg"
    )
    _print_contact_index_audit(robot, contact, foot_names, body_ids, contact_body_ids)

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

    print("\n=== Foot contact force |F| (N) — chatter indicator (contact sensor ids) ===")
    for bn, vals in fz_hist.items():
        if not vals:
            print(f"  {bn:<28} (no samples)")
            continue
        arr = np.asarray(vals, dtype=np.float64)
        fz_arr = np.asarray(fz_comp_hist[bn], dtype=np.float64)
        mean_f = float(arr.mean())
        print(
            f"  {bn:<28} mean|F|={mean_f:8.1f}  meanFz={fz_arr.mean():8.1f}  std={arr.std():8.1f}  "
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
