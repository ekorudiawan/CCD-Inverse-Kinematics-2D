"""
Cyclic Coordinate Descent (CCD) Inverse Kinematics for planar n-link arms.
"""

from dataclasses import dataclass
from typing import Optional
import math
import numpy as np


def rotate_z(theta: float) -> np.ndarray:
    """Homogeneous rotation matrix around Z axis (4x4)."""
    ct, st = math.cos(theta), math.sin(theta)
    return np.array([
        [ct, -st, 0, 0],
        [st,  ct, 0, 0],
        [0,   0,  1, 0],
        [0,   0,  0, 1],
    ])


def translate(dx: float, dy: float = 0.0, dz: float = 0.0) -> np.ndarray:
    """Homogeneous translation matrix (4x4)."""
    return np.array([
        [1, 0, 0, dx],
        [0, 1, 0, dy],
        [0, 0, 1, dz],
        [0, 0, 0, 1],
    ])


def FK(angles: list[float], links: list[float]) -> list[np.ndarray]:
    """
    Forward kinematics: compute joint positions given joint angles.

    Parameters
    ----------
    angles : list[float]
        Joint angles in degrees.
    links : list[float]
        Link lengths.

    Returns
    -------
    list[ndarray]
        4x4 homogeneous transforms for each joint (including base).
        transforms[0] = eye(4) (base), transforms[i] = transform of joint i.
    """
    n = len(links)
    P = [np.eye(4)]
    for i in range(n):
        R = rotate_z(math.radians(angles[i]))
        T = translate(links[i])
        P.append(P[-1] @ R @ T)
    return P


@dataclass
class IKResult:
    """Result of inverse kinematics computation."""

    angles: list[float]
    error: float
    solved: bool
    iterations: int


def IK(
    target: np.ndarray,
    angles: list[float],
    links: list[float],
    max_iter: int = 10000,
    err_min: float = 0.1,
) -> IKResult:
    """
    Cyclic Coordinate Descent IK solver for planar 2D arms.

    Parameters
    ----------
    target : ndarray (3,)
        Desired [x, y, z] position of end-effector.
    angles : list[float]
        Initial joint angles in degrees (modified in place).
    links : list[float]
        Link lengths.
    max_iter : int
        Maximum solver iterations.
    err_min : float
        Convergence threshold (Euclidean distance).

    Returns
    -------
    IKResult
        Solved joint angles, final error, success flag, iteration count.
    """
    solved = False
    err = float("inf")
    n_links = len(links)

    for loop in range(max_iter):
        for i in range(n_links - 1, -1, -1):
            P = FK(angles, links)
            end_eff = P[-1][:3, 3]
            end_to_target = target - end_eff
            err = float(math.sqrt(end_to_target[0] ** 2 + end_to_target[1] ** 2))

            if err < err_min:
                solved = True
                break

            # Vector from current joint i to end-effector
            cur_to_end = end_eff - P[i][:3, 3]
            cur_to_end_mag = float(
                math.sqrt(cur_to_end[0] ** 2 + cur_to_end[1] ** 2)
            )

            # Vector from current joint i to target
            cur_to_target = target - P[i][:3, 3]
            cur_to_target_mag = float(
                math.sqrt(cur_to_target[0] ** 2 + cur_to_target[1] ** 2)
            )

            denom = cur_to_end_mag * cur_to_target_mag
            if denom <= 1e-4:
                cos_rot = 1.0
                sin_rot = 0.0
            else:
                cos_rot = (
                    cur_to_end[0] * cur_to_target[0]
                    + cur_to_end[1] * cur_to_target[1]
                ) / denom
                sin_rot = (
                    cur_to_end[0] * cur_to_target[1]
                    - cur_to_end[1] * cur_to_target[0]
                ) / denom

            rot_ang = math.acos(max(-1.0, min(1.0, cos_rot)))
            if sin_rot < 0.0:
                rot_ang = -rot_ang

            # Update joint angle (clamp to [0, 360))
            angles[i] = (angles[i] + math.degrees(rot_ang)) % 360

        if solved:
            break

    return IKResult(
        angles=list(angles),
        error=err,
        solved=solved,
        iterations=loop + 1,
    )


def demo():
    """Run IK on a few test targets and print results."""
    links = [50, 40, 35, 30]
    angles = [0.0, 0.0, 0.0, 0.0]

    targets = [
        np.array([80.0, 40.0, 0.0]),
        np.array([30.0, 60.0, 0.0]),
        np.array([-20.0, 30.0, 0.0]),
    ]

    for tgt in targets:
        result = IK(tgt, angles, links)
        P = FK(result.angles, links)
        ee = P[-1][:3, 3]
        print(f"Target=({tgt[0]:6.1f}, {tgt[1]:6.1f}) "
              f"solved={result.solved} "
              f"iter={result.iterations:4d} "
              f"err={result.error:.4f} "
              f"EE=({ee[0]:6.1f}, {ee[1]:6.1f})")


if __name__ == "__main__":
    demo()
