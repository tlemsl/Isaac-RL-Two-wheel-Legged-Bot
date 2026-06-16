"""Dump mass/inertia from Wolf USD. Run inside nvidia-ros2 with env_rl.sh."""
from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless"])
AppLauncher(args)

from pxr import Usd, UsdPhysics  # noqa: E402

RL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
USD_PATH = os.path.join(
    RL_ROOT,
    "lab/flamingo/assets/data/Robots/Wolf/wolf_rev_01_0_0/asset/Wolf_rev_1_0_0.usd",
)

JOINT_ORDER = [
    "HAA_front_left_joint",
    "HFE_front_left_joint",
    "KFE_front_left_joint",
    "HAA_front_right_joint",
    "HFE_front_right_joint",
    "KFE_front_right_joint",
    "HAA_back_left_joint",
    "HFE_back_left_joint",
    "KFE_back_left_joint",
    "AFE_back_left_joint",
    "HAA_back_right_joint",
    "HFE_back_right_joint",
    "KFE_back_right_joint",
    "AFE_back_right_joint",
]


def _vec3(attr) -> tuple[float, float, float] | None:
    if attr is None or not attr.HasAuthoredValue():
        return None
    v = attr.Get()
    return float(v[0]), float(v[1]), float(v[2])


def _child_link_name(joint_prim) -> str | None:
    targets = joint_prim.GetRelationship("physics:body1").GetTargets()
    if not targets:
        return None
    return targets[0].name


def main() -> None:
    stage = Usd.Stage.Open(USD_PATH)
    if stage is None:
        raise FileNotFoundError(USD_PATH)

    link_rows: list[dict] = []
    total_mass = 0.0

    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        mass_attr = prim.GetAttribute("physics:mass")
        if not mass_attr or not mass_attr.HasAuthoredValue():
            continue
        mass = float(mass_attr.Get())
        ixx, iyy, izz = _vec3(prim.GetAttribute("physics:diagonalInertia")) or (None, None, None)
        cx, cy, cz = _vec3(prim.GetAttribute("physics:centerOfMass")) or (None, None, None)
        total_mass += mass
        link_rows.append(
            {
                "name": prim.GetName(),
                "path": str(prim.GetPath()),
                "mass": mass,
                "ixx": ixx,
                "iyy": iyy,
                "izz": izz,
                "com": (cx, cy, cz),
            }
        )

    link_rows.sort(key=lambda r: r["name"])
    joint_to_child: dict[str, str] = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        child = _child_link_name(prim)
        if child:
            joint_to_child[prim.GetName()] = child

    link_by_name = {r["name"]: r for r in link_rows}

    print(f"USD: {USD_PATH}")
    print(f"Links with mass: {len(link_rows)}")
    print(f"Total mass (sum of link masses): {total_mass:.6f} kg\n")

    print("=== All links (mass + diagonal inertia + CoM in link frame) ===")
    hdr = (
        f"{'link':<32} {'mass[kg]':>10} {'Ixx':>12} {'Iyy':>12} {'Izz':>12} "
        f"{'CoM x':>10} {'CoM y':>10} {'CoM z':>10}"
    )
    print(hdr)
    for r in link_rows:
        cx, cy, cz = r["com"]
        print(
            f"{r['name']:<32} {r['mass']:10.5f} "
            f"{r['ixx'] or 0:12.6e} {r['iyy'] or 0:12.6e} {r['izz'] or 0:12.6e} "
            f"{(cx or 0):10.5f} {(cy or 0):10.5f} {(cz or 0):10.5f}"
        )

    print("\n=== Per revolute joint → child link inertial properties ===")
    print(hdr.replace("link", "joint → child"))
    for jn in JOINT_ORDER:
        child = joint_to_child.get(jn)
        if child is None:
            # Isaac may name joints without _joint suffix
            alt = jn.replace("_joint", "")
            child = joint_to_child.get(alt)
        if child is None:
            print(f"{jn:<32} (joint not found in USD)")
            continue
        r = link_by_name.get(child)
        if r is None:
            print(f"{jn + ' → ' + child:<32} (child link mass not found)")
            continue
        cx, cy, cz = r["com"]
        label = f"{jn} → {child}"
        print(
            f"{label:<32} {r['mass']:10.5f} "
            f"{r['ixx'] or 0:12.6e} {r['iyy'] or 0:12.6e} {r['izz'] or 0:12.6e} "
            f"{(cx or 0):10.5f} {(cy or 0):10.5f} {(cz or 0):10.5f}"
        )

    # aggregate by leg segment type
    print("\n=== Mass subtotals by segment ===")
    segments = {"base": 0.0, "Hip": 0.0, "Thigh": 0.0, "Shank": 0.0, "Ankle": 0.0, "Foot": 0.0}
    for r in link_rows:
        n = r["name"]
        if n == "base_link":
            segments["base"] += r["mass"]
        elif n.startswith("Hip_"):
            segments["Hip"] += r["mass"]
        elif n.startswith("Thigh_"):
            segments["Thigh"] += r["mass"]
        elif n.startswith("Shank_"):
            segments["Shank"] += r["mass"]
        elif n.startswith("Ankle_"):
            segments["Ankle"] += r["mass"]
        elif n.startswith("Foot_"):
            segments["Foot"] += r["mass"]
    for k, v in segments.items():
        print(f"  {k:<8} {v:8.4f} kg")


if __name__ == "__main__":
    main()
