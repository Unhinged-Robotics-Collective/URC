from matplotlib import pyplot as plt
import numpy as np

from scipy.spatial.transform import Rotation

import cv2
from dataset_generation.config import CamInfo


def umeyama(X: np.ndarray, Y: np.ndarray):
    """
    min_{s,R,t} || s*R*p_1 + t - p_2 ||
    """
    # shapes: (xyz, num_points)
    mu_x = X.mean(axis=1).reshape(-1, 1)
    mu_y = Y.mean(axis=1).reshape(-1, 1)
    var_x = np.square(X - mu_x).sum(axis=0).mean()
    cov_xy = ((Y - mu_y) @ (X - mu_x).T) / X.shape[1]
    U, D, VH = np.linalg.svd(cov_xy)
    S = np.eye(X.shape[0])
    if np.linalg.det(U) * np.linalg.det(VH) < 0:
        S[-1, -1] = -1
    c = np.trace(np.diag(D) @ S) / var_x
    R = U @ S @ VH
    t = mu_y - c * R @ mu_x
    return c, R, t


def rot_mat_2d(angle: float) -> np.ndarray:
    return np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]
    ])

def test_umeyama():
    p1 = np.random.rand(3, 21)
    R = Rotation.from_rotvec(np.random.rand(3)).as_matrix()
    t = np.random.rand(3)
    s = np.random.rand()
    p2 = s * R @ p1 + t[:, None]

    return p1, p2, umeyama(p1, p2)

def umeyama_apply(p: np.ndarray, s: float, R: np.ndarray, t: np.ndarray):
    return s * R @ p + t

# mlops => scale data

def triangulate_point(d1: np.ndarray, d2: np.ndarray, R: np.ndarray, t: np.ndarray):
    """
    Triangulate a 3D point seen by two cameras.

    Parameters
    ----------
    d1 : (3,) array
        Ray direction from C1 (normalized or not).
    d2 : (3,) array
        Ray direction from C2 (normalized or not).
    R : (3,3) array
        Rotation matrix from C1 to C2.
    t : (3,) array
        Translation vector from C1 to C2.

    Returns
    -------
    X1 : (3,) array
        3D point in C1 coordinates.
    X2 : (3,) array
        3D point in C2 coordinates.
    lambdas : (2,) array
        Depth values [lambda1, lambda2].
    """

    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)
    t = np.asarray(t).reshape(3)

    # Build least-squares system: [-R d1 | d2] [λ1, λ2]^T = t
    A = np.column_stack((-R @ d1, d2))
    lambdas, *_ = np.linalg.lstsq(A, t, rcond=None)

    lam1, lam2 = lambdas
    X1 = lam1 * d1
    X2 = lam2 * d2

    return X1, X2, lambdas


def umeyama_transform(p1: np.ndarray, p2: np.ndarray):
    """Return p1 in frame of p2"""
    s, R, t = umeyama(p1, p2)
    p_reconstructed = umeyama_apply(p1, s, R, t)
    return p_reconstructed


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

    f1_range = np.linspace(0.5*cam1.w, 1.5*cam1.w, 30)
    f2_range = np.linspace(0.5*cam2.w, 1.5*cam2.w, 30)
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


def calculate_cam_matrices(cam1: CamInfo, cam2: CamInfo, uv1: np.ndarray, uv2: np.ndarray) -> np.ndarray:
    """
    Accepts: (N,2)

    Returns (N,3)"""
    # Estimate f1,f2 and F
    f1_est, f2_est, F_est, inl = estimate_f_from_F(uv1, uv2, cam1, cam2)
    # print("f1_est, f2_est:", f1_est, f2_est)

    # Build E with estimated f's
    K1_est = K_from_f(f1_est, cam1.w, cam1.h)
    K2_est = K_from_f(f2_est, cam2.w, cam2.h)
    E_est = K2_est.T @ F_est @ K1_est
    pts1n = cv2.undistortPoints(uv1[inl].reshape(-1,1,2), K1_est, np.zeros((5,)))
    pts2n = cv2.undistortPoints(uv2[inl].reshape(-1,1,2), K2_est, np.zeros((5,)))

    # Either use pixel coords + K..
    retval, R_est, t_est, mask_pose = cv2.recoverPose(E_est, pts1n, pts2n)
    # print("recoverPose inliers:", retval, "/", inl.sum())
    # print("R_est:\n", R_est)
    # print("t_dir_est:", (t_est / (np.linalg.norm(t_est) + 1e-9)).ravel())

    # Triangulate for sanity
    P1_est = K1_est @ np.hstack([np.eye(3), np.zeros((3,1))])
    P2_est = K2_est @ np.hstack([R_est, t_est])
    pts4d = cv2.triangulatePoints(P1_est, P2_est, pts1n.squeeze(1).T, pts2n.squeeze(1).T)  # (4,Ninl)
    pts3d = (pts4d[:3] / pts4d[3]).T

    # print("Triangulated Z:", pts3d[:, 2])
    return pts3d


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

def main():
    np.random.seed(42)
    p1, p2, (s, R, t) = test_umeyama()
    p_reconstructed = umeyama_apply(p1, s, R, t)
    print(p_reconstructed)
    print("PARAMS", s, R, t)
    print("DET:", np.linalg.det(R))
    plt.scatter(p1[0], p1[1], color=(1,0,0))
    plt.scatter(p2[0], p2[1], color=(0,1,0), s=100)
    plt.scatter(p_reconstructed[0], p_reconstructed[1], color=(1,1,0))
    plt.show()
    test_estimate()


if __name__ == "__main__":
    main()
