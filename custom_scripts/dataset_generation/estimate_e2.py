import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from dataclasses import dataclass, field


@dataclass
class CamInfo:
    w: float
    h: float
    x0: float = -1
    y0: float = -1
    f: float = -1
    R: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=float))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))

    def __post_init__(self):
        if self.x0 <= -1:
            self.x0 = self.w / 2.0
        if self.y0 <= -1:
            self.y0 = self.h / 2.0


def K_from_f(f, w, h):
    return np.array([[f, 0, w/2],
                     [0, f, h/2],
                     [0, 0,   1 ]], dtype=float)


def essential_cost(F: np.ndarray, f1: float, f2: float, w1: float, h1: float, w2: float, h2: float):
    K1 = K_from_f(f1, w1, h1)
    K2 = K_from_f(f2, w2, h2)
    E = K2.T @ F @ K1
    _, S, _ = np.linalg.svd(E)
    S = np.sort(S)[::-1]
    return (S[0]-S[1])**2 + (S[2])**2

def estimate_f_from_F(xy1: np.ndarray, xy2: np.ndarray, cam1: CamInfo, cam2: CamInfo):
    xy1 = xy1.astype(np.float32)
    xy2 = xy2.astype(np.float32)
    F, mask = cv2.findFundamentalMat(xy1, xy2, cv2.FM_RANSAC, 1.0, 0.999)
    if F is None or mask is None or mask.sum() < 8:
        raise RuntimeError("F estimation failed or too few inliers.")
    inl = mask.ravel().astype(bool)
    # xy1i, xy2i = xy1[inl], xy2[inl]

    f1_range = np.linspace(0.5*cam1.w, 1.5*cam1.w, 100)
    f2_range = np.linspace(0.5*cam2.w, 1.5*cam2.w, 100)
    best = (np.inf, None, None)
    for f1 in f1_range:
        for f2 in f2_range:
            c = essential_cost(F, f1, f2, cam1.w, cam1.h, cam2.w, cam2.h)
            if c < best[0]:
                best = (c, f1, f2)
    f1_best, f2_best = best[1], best[2]
    return f1_best, f2_best, F, inl

def homogenize(X3xN: np.ndarray) -> np.ndarray:
    # (3,N) -> (4,N)
    assert X3xN.shape[0] == 3
    ones = np.ones((1, X3xN.shape[1]), dtype=X3xN.dtype)
    return np.concatenate([X3xN, ones], axis=0)


def calculate_cam_matrices(cam1: CamInfo, cam2: CamInfo, uv1: np.ndarray, uv2: np.ndarray):
    # Estimate f1,f2 and F
    f1_est, f2_est, F_est, inl = estimate_f_from_F(uv1, uv2, cam1, cam2)
    print("f1_est, f2_est:", f1_est, f2_est)

    # Build E with estimated f's
    K1_est = K_from_f(f1_est, cam1.w, cam1.h)
    K2_est = K_from_f(f2_est, cam2.w, cam2.h)
    E_est = K2_est.T @ F_est @ K1_est

    # Either use pixel coords + K...
    retval, R_est, t_est, mask_pose = cv2.recoverPose(E_est, uv1[inl], uv2[inl], K1_est)
    print("recoverPose inliers:", retval, "/", inl.sum())
    print("R_est:\n", R_est)
    print("t_dir_est:", (t_est / (np.linalg.norm(t_est) + 1e-9)).ravel())

    # Triangulate for sanity
    P1_est = K1_est @ np.hstack([np.eye(3), np.zeros((3,1))])
    P2_est = K2_est @ np.hstack([R_est, t_est])
    pts4d = cv2.triangulatePoints(P1_est, P2_est, uv1[inl].T, uv2[inl].T)  # (4,Ninl)
    pts3d = (pts4d[:3] / pts4d[3]).T

    print("Triangulated Z (first 5):", pts3d[:5, 2])

def test_estimate():
    # Image size
    W, H = 1280, 720
    cam1 = CamInfo(w=W, h=H)
    cam2 = CamInfo(w=W, h=H)

    # Intrinsics (ground truth for synthetic)
    K1_gt = K_from_f(900, W, H)
    K2_gt = K_from_f(1000, W, H)

    # Relative pose (cam2 wrt cam1)
    R_gt = Rotation.from_rotvec(np.array([0.2, -0.1, 0.05])).as_matrix()
    t_gt = np.array([0.2, 0.0, 0.05])  # arbitrary baseline

    # Cameras
    P1 = K1_gt @ np.hstack([np.eye(3), np.zeros((3,1))])      # cam1 canonical
    P2 = K2_gt @ np.hstack([R_gt, t_gt.reshape(3,1)])         # cam2

    # 3D points (same set for both cams)
    N = 21
    X = np.random.uniform([-0.2, -0.2, 1.0], [0.2, 0.2, 1.6], size=(N,3)).T  # (3,N), in front of both cams
    Xh = homogenize(X)  # (4,N)

    # Project
    uv1_h = P1 @ Xh
    uv2_h = P2 @ Xh
    uv1 = (uv1_h[:2] / uv1_h[2]).T  # (N,2)
    uv2 = (uv2_h[:2] / uv2_h[2]).T

    # Add tiny pixel noise
    uv1 += np.random.normal(0, 0.2, uv1.shape)
    uv2 += np.random.normal(0, 0.2, uv2.shape)

    calculate_cam_matrices(cam1, cam2, uv1, uv2)


if __name__ == "__main__":
    test_estimate()
