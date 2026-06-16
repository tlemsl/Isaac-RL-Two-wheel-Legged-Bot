# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from lab.flamingo.assets.flamingo import FLAMINGO_ASSETS_DATA_DIR

# Shared delayed-PD defaults for Wolf policy training (joint-space, 14 DOF).
# Tuned 2026-06-15: HAA/HFE/AFE 120/2, KFE 300/15 — see
# docs/experiments/rl/20260615_wolf_isaac_pd_damping_shake.md
# (prior 200/10 caused back-leg shake and AFE torque saturation at effort_limit).
_WOLF_PD_DELAY = dict(min_delay=0, max_delay=4)
_WOLF_HIP_PD = dict(stiffness=150.0, damping=2.0)
_WOLF_KFE_PD = dict(
    effort_limit=600.0,
    velocity_limit=25.0,
    stiffness=300.0,
    damping=15.0,
    friction=0.0,
    armature=0.035,
)
_WOLF_AFE_PD = dict(
    effort_limit=60.0,
    velocity_limit=25.0,
    stiffness=150.0,
    damping=2.0,
    friction=0.0,
    armature=0.01,
)


def _wolf_hip_delayed_pd(joint_names: list[str]) -> DelayedPDActuatorCfg:
    return DelayedPDActuatorCfg(
        joint_names_expr=joint_names,
        effort_limit=120.0,
        velocity_limit=20.0,
        stiffness={name: _WOLF_HIP_PD["stiffness"] for name in joint_names},
        damping={name: _WOLF_HIP_PD["damping"] for name in joint_names},
        friction={name: 0.0 for name in joint_names},
        armature={name: 0.01 for name in joint_names},
        **_WOLF_PD_DELAY,
    )


def _wolf_knee_delayed_pd(joint_name: str) -> DelayedPDActuatorCfg:
    return DelayedPDActuatorCfg(
        joint_names_expr=[joint_name],
        effort_limit=_WOLF_KFE_PD["effort_limit"],
        velocity_limit=_WOLF_KFE_PD["velocity_limit"],
        stiffness={joint_name: _WOLF_KFE_PD["stiffness"]},
        damping={joint_name: _WOLF_KFE_PD["damping"]},
        friction={joint_name: _WOLF_KFE_PD["friction"]},
        armature={joint_name: _WOLF_KFE_PD["armature"]},
        **_WOLF_PD_DELAY,
    )


def _wolf_ankle_delayed_pd(joint_name: str) -> DelayedPDActuatorCfg:
    return DelayedPDActuatorCfg(
        joint_names_expr=[joint_name],
        effort_limit=_WOLF_AFE_PD["effort_limit"],
        velocity_limit=_WOLF_AFE_PD["velocity_limit"],
        stiffness={joint_name: _WOLF_AFE_PD["stiffness"]},
        damping={joint_name: _WOLF_AFE_PD["damping"]},
        friction={joint_name: _WOLF_AFE_PD["friction"]},
        armature={joint_name: _WOLF_AFE_PD["armature"]},
        **_WOLF_PD_DELAY,
    )


WOLF_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{FLAMINGO_ASSETS_DATA_DIR}/Robots/Wolf/wolf_rev_01_0_0/asset/Wolf_rev_1_0_0.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
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
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.8,
    actuators={
        "joints_front_left_H": _wolf_hip_delayed_pd(
            ["HAA_front_left_joint", "HFE_front_left_joint"]
        ),
        "joints_front_left_K": _wolf_knee_delayed_pd("KFE_front_left_joint"),
        "joints_front_right_H": _wolf_hip_delayed_pd(
            ["HAA_front_right_joint", "HFE_front_right_joint"]
        ),
        "joints_front_right_K": _wolf_knee_delayed_pd("KFE_front_right_joint"),
        "joints_back_left_H": _wolf_hip_delayed_pd(
            ["HAA_back_left_joint", "HFE_back_left_joint"]
        ),
        "joints_back_left_K": _wolf_knee_delayed_pd("KFE_back_left_joint"),
        "joints_back_left_A": _wolf_ankle_delayed_pd("AFE_back_left_joint"),
        "joints_back_right_H": _wolf_hip_delayed_pd(
            ["HAA_back_right_joint", "HFE_back_right_joint"]
        ),
        "joints_back_right_K": _wolf_knee_delayed_pd("KFE_back_right_joint"),
        "joints_back_right_A": _wolf_ankle_delayed_pd("AFE_back_right_joint"),
    },
)
