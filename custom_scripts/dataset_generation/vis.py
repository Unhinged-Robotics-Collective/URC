import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes # type:ignore
import numpy as np
from mpl_toolkits.mplot3d import axes3d  # noqa: F401

from pytransform3d.plot_utils import Frame
from pytransform3d.rotations import passive_matrix_from_angle, R_id
from pytransform3d.transformations import transform_from, concat


class PositionVisualizer:
    def __init__(self):
        n_frames = 200

        fig = plt.figure(figsize=(5, 5))

        ax: Axes = fig.add_subplot(111, projection="3d")
        ax.set_xlim((-1, 1))
        ax.set_ylim((-1, 1))
        ax.set_zlim((-1, 1)) # type: ignore[attr-defined]
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z") # type: ignore[attr-defined]

        H = np.eye(4)
        frame = Frame(H, s=0.2)
        frame.add_frame(ax) # type: ignore

        anim = animation.FuncAnimation(
            fig,
            self.update_trajectory,
            n_frames,
            fargs=(self, n_frames, frame),
            interval=50,
            blit=False,
        )

        plt.show()

    def update_trajectory(self, step: int, n_frames: int, frame: Frame):
        progress = float(step + 1) / float(n_frames)
        H = np.eye(4)
        H0 = transform_from(R_id, np.zeros(3))
        H_mod = np.eye(4)
        H0[:3, 3] = np.array([progress, 0, progress])
        H_mod[:3, :3] = passive_matrix_from_angle(2, 8 * np.pi * progress)
        H = concat(H0, H_mod)
        frame.set_data(H)
        return frame

def update_trajectory(step: int, n_frames: int, frame: Frame):
    progress = float(step + 1) / float(n_frames)
    H = np.eye(4)
    H0 = transform_from(R_id, np.zeros(3))
    H_mod = np.eye(4)
    H0[:3, 3] = np.array([progress, 0, progress])
    H_mod[:3, :3] = passive_matrix_from_angle(2, 8 * np.pi * progress)
    H = concat(H0, H_mod)
    frame.set_data(H)
    return frame

if __name__ == "__main__":
    n_frames = 200

    fig = plt.figure(figsize=(5, 5))

    ax: Axes = fig.add_subplot(111, projection="3d")
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    ax.set_zlim((-1, 1)) # type: ignore[attr-defined]
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z") # type: ignore[attr-defined]

    H = np.eye(4)
    frame = Frame(H, s=0.2)
    frame.add_frame(ax) # type: ignore

    anim = animation.FuncAnimation(
        fig,
        update_trajectory,
        n_frames,
        fargs=(n_frames, frame),
        interval=50,
        blit=False,
    )

    plt.show()

