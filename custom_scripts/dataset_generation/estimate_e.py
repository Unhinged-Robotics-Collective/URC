import numpy as np
import cv2
from scipy.spatial.transform import Rotation

"""
For mediapipe:

x0, y0 = 0.5, 0.5

we can get the width, height from the window/image size
"""
from dataclasses import dataclass, field

@dataclass
class CamInfo:
    w: float
    h: float
    x0: float = -1
    y0: float = -1
    f: float = -1
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        if self.x0 < 0:
            self.x0 = self.w / 2.0
        if self.y0 < 0:
            self.y0 = self.h / 2.0


# def center_pixels(xy: np.ndarray, x0: np.ndarray, y0: np.ndarray) -> np.ndarray:
#     xy_c = xy.copy()
#     xy_c[:,0] -= x0
#     xy_c[:,1] -= y0
#     return xy_c


def K_from_f(f, w, h):
    return np.array([[f, 0, w/2],
                     [0, f, h/2],
                     [0, 0,   1 ]], float)


def essential_cost(F: np.ndarray, f1: float, f2: float, w1: float, h1: float, w2: float, h2: float):
    """
    The Essential matrix decomposed using SVD:
    U S V^T
    where S = diag(s,s,0)

    Therefore:
    1. we minimize difference between elements (0,0) & (1,1)
    2. minimize element (2,2)
    """
    K1 = K_from_f(f1, w1, h1)
    K2 = K_from_f(f2, w2, h2)
    E = K2.T @ F @ K1
    U, S, Vt = np.linalg.svd(E)
    S = np.sort(S)[::-1]  # sigma1 >= sigma2 >= sigma3
    return (S[0]-S[1])**2 + (S[2])**2


def estimate_fxy_from_two_views(xy1: np.ndarray, xy2: np.ndarray, cam1: CamInfo, cam2: CamInfo):
    assert xy1.shape[1] == 2, f"xy: {xy1.shape}"
    assert len(xy1.shape) == 2, f"xy: {xy1.shape}"
    # Estimate F with RANSAC
    F, mask = cv2.findFundamentalMat(xy1, xy2, cv2.FM_RANSAC, 1.0, 0.999)
    # Coarse-to-fine search for f (shared for cam1 & cam2)
    f1_range = np.linspace(0.1*cam1.w, 1.5*cam1.w, 30)
    f2_range = np.linspace(0.1*cam2.w, 1.5*cam2.w, 30)

    best = (1e18, None, None)
    for f1 in f1_range:
        for f2 in f2_range:
            c = essential_cost(F, f1, f2, cam1.w, cam1.h, cam2.w, cam2.h)
            if c < best[0]:
                best = (c, f1, f2)
    f1_best, f2_best = best[1], best[2]

    # # Optional: tiny local refinement around f_best
    # local = np.linspace(0.8*f_best, 1.2*f_best, 21)
    # for f in local:
    #     c = essential_cost(F, f, x0, y0)
    #     if c < best[0]:
    #         best = (c, f)
    return f1_best, f2_best, F, mask


def homogenize(arr: np.ndarray):
    # print(arr.shape)
    # print(np.ones(arr.shape[1]))
    assert arr.shape[0] == 3
    return np.concatenate([arr, np.ones(arr.shape[1])[None, :]], axis=0)


def test_estimate():
    from dataset_generation import config
    cam1 = CamInfo(w=config.IMAGE_SIZE[0], h=config.IMAGE_SIZE[1])
    cam2 = CamInfo(w=config.IMAGE_SIZE[0], h=config.IMAGE_SIZE[1])

    # setup fake data
    random_points = np.random.rand(3, 21)
    random_points2 = random_points + np.random.rand(3, 21) * 0.05
    # print("random points", random_points)
    K1_base = K_from_f(f=250, w=config.IMAGE_SIZE[0], h=config.IMAGE_SIZE[1])
    K2_base = K_from_f(f=280, w=config.IMAGE_SIZE[0], h=config.IMAGE_SIZE[1])
    R_rand = Rotation.from_rotvec(np.random.rand(3)).as_matrix()
    # print("R", R_rand)
    t_rand = np.random.rand(3)
    # print("t", t_rand)

    Rt2 = np.concatenate([R_rand, t_rand[:, None]], axis=1)
    Rt = np.concat([np.eye(3), np.zeros(3)[:, None]], axis=1)
    print("Rt2", Rt2)
    P1_base = K1_base @ Rt
    P2_base = K2_base @ Rt2
    uv1 = P1_base @ homogenize(random_points)
    uv2 = P2_base @ homogenize(random_points2)
    print("random points 1", uv1.shape)
    uv1 /= uv1[2, :]
    uv2 /= uv2[2, :]
    uv1 = uv1[:2].T
    uv2 = uv2[:2].T
    print("random points ", uv1.shape, uv2.shape)
    f1_best, f2_best, F, mask = estimate_fxy_from_two_views(uv1, uv2, cam1, cam2)
    print(uv1.shape)
    inl = mask.ravel().astype(bool)
    pts1 = uv1[inl]
    pts2 = uv2[inl]
    print("f1_best", f1_best, " f2_best", f2_best, " F", F)
    K1 = K_from_f(f1_best, cam1.w, cam1.h)
    K2 = K_from_f(f2_best, cam2.w, cam2.h)
    E = K2.T @ F @ K1

    # normalize points into pixel coords for recoverPose
    pts1u = cv2.undistortPoints(pts1.reshape(-1,1,2), K1, np.zeros((5,)))
    pts2u = cv2.undistortPoints(pts2.reshape(-1,1,2), K2, np.zeros((5,)))

    _, R, t, mask_pose = cv2.recoverPose(E, pts1u, pts2u)
    print("R",R)
    print("t",t)


if __name__ == "__main__":
    test_estimate()