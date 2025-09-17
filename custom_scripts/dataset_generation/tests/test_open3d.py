import numpy as np
import open3d as o3d


def make_frame(scale: float = 0.2) -> o3d.geometry.LineSet:
    points = [
        [0, 0, 0],
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, scale],
    ]
    lines = [[0, 1], [0, 2], [0, 3]]
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    frame = o3d.geometry.LineSet()
    frame.points = o3d.utility.Vector3dVector(points)
    frame.lines = o3d.utility.Vector2iVector(lines)
    frame.colors = o3d.utility.Vector3dVector(colors)
    return frame


if __name__ == "__main__":
    frame = make_frame(0.2)
    vis = o3d.visualization.Visualizer()
    vis.create_window("Test Frame")
    vis.add_geometry(frame)
    vis.run()  # ✅ blocks, handles all events
    vis.destroy_window()

