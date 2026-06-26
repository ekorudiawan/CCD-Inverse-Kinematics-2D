#!/usr/bin/env python3
"""
CCD IK CLI — solve & visualize inverse kinematics from terminal.
"""

import sys
import os

os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from ccd_ik.kinematics import FK, IK


def plot_arm(angles: list[float], links: list[float], ax, target=None):
    """Plot robot arm on given axes."""
    P = FK(angles, links)
    xs = [p[0, 3] for p in P]
    ys = [p[1, 3] for p in P]
    ax.plot(xs, ys, "b-", lw=4, zorder=3)
    ax.plot(xs, ys, "bo", ms=6, zorder=4)
    ax.plot(xs[-1], ys[-1], "ro", ms=8, zorder=5, label="End Effector")
    if target is not None:
        ax.plot(target[0], target[1], "g*", ms=12, zorder=5, label="Target")
    ax.legend()
    ax.set_xlim(-10, 160)
    ax.set_ylim(-10, 160)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title("CCD Inverse Kinematics — 2D Arm")


def make_animation(links: list[float], targets: list[np.ndarray]) -> str:
    """Generate animated GIF of IK reaching multiple targets."""
    angles = [0.0] * len(links)
    frames = []

    for tgt in targets:
        result = IK(tgt, angles, links)
        # Interpolate intermediate IK states for smooth animation
        n_ik_frames = max(5, result.iterations // 2)
        for _ in range(n_ik_frames):
            ik_res = IK(tgt, angles, links)
            P = FK(ik_res.angles, links)
            frames.append((ik_res.angles.copy(), tgt, P))

    fig, ax = plt.subplots(figsize=(5, 5))

    def update(frame_idx):
        ax.clear()
        ang, tgt, P = frames[frame_idx]
        xs = [p[0, 3] for p in P]
        ys = [p[1, 3] for p in P]
        ax.plot(xs, ys, "b-", lw=4, zorder=3)
        ax.plot(xs, ys, "bo", ms=6, zorder=4)
        ax.plot(xs[-1], ys[-1], "ro", ms=8, zorder=5)
        ax.plot(tgt[0], tgt[1], "g*", ms=12, zorder=5)
        ax.set_xlim(-10, 160)
        ax.set_ylim(-10, 160)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"CCD IK — frame {frame_idx + 1}/{len(frames)}")
        return ax.lines

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=80, blit=False
    )

    out = "images/ccd_ik.gif"
    os.makedirs("images", exist_ok=True)
    anim.save(out, writer="pillow", fps=12)
    plt.close(fig)
    return out


def main():
    import argparse

    parser = argparse.ArgumentParser(description="CCD Inverse Kinematics")
    parser.add_argument(
        "action",
        nargs="?",
        default="demo",
        choices=["demo", "gif"],
        help="demo: print results; gif: generate animation",
    )
    args = parser.parse_args()

    links = [50, 40, 35, 30]

    if args.action == "demo":
        from ccd_ik.kinematics import demo
        demo()
    elif args.action == "gif":
        targets = [
            np.array([80.0, 40.0, 0.0]),
            np.array([30.0, 60.0, 0.0]),
            np.array([100.0, 80.0, 0.0]),
            np.array([40.0, 20.0, 0.0]),
            np.array([120.0, 50.0, 0.0]),
        ]
        out = make_animation(links, targets)
        print(f"GIF saved: {out}")


if __name__ == "__main__":
    main()
