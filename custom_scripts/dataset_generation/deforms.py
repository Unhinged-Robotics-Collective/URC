from matplotlib import pyplot as plt
import numpy as np

from scipy.spatial.transform import Rotation
from scipy.ndimage import rotate


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

def deform(dir: np.ndarray, origin: np.ndarray):
    # project onto the line => we get distance from origin
    # IDK even if it's deforming the hands using perspective
    ...

def umeyama_transform(p1: np.ndarray, p2: np.ndarray):
    """Return p1 in frame of p2"""
    s, R, t = umeyama(p1, p2)
    p_reconstructed = umeyama_apply(p1, s, R, t)
    return p_reconstructed

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


if __name__ == "__main__":
    main()
