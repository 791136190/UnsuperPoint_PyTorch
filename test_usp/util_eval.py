import numpy as np
from glob import glob


def warp_kpt(keypoints, H):
    # num_points = keypoints.shape[0]
    # homogeneous_points = np.concatenate([keypoints, np.ones((num_points, 1))], axis=1)
    # warped_points = np.dot(homogeneous_points, np.transpose(H))
    # return warped_points[:, :2] / warped_points[:, 2:]

    r = np.zeros_like(keypoints)

    Denominator = keypoints[:, 0] * H[2, 0] + keypoints[:, 1] * H[2, 1] + H[2, 2]
    r[:, 0] = (keypoints[:, 0] * H[0, 0] + keypoints[:, 1] * H[0, 1] + H[0, 2]) / Denominator
    r[:, 1] = (keypoints[:, 0] * H[1, 0] + keypoints[:, 1] * H[1, 1] + H[1, 2]) / Denominator

    return r

def filter_keypoints(points, prob, des, shape):
    """ Keep only the points whose coordinates are
    inside the dimensions of shape. """
    mask = (points[:, 0] >= 0) & (points[:, 0] < shape[0]) &\
           (points[:, 1] >= 0) & (points[:, 1] < shape[1])
    return points[mask, :], prob[mask, :], des[mask, :]

def keep_true_keypoints(points, prob, des, H, shape):
    """ Keep only the points whose warped coordinates by H
    are still inside shape. """
    warped_points = warp_kpt(points[:, [0, 1]], H)
    warped_points[:, [0, 1]] = warped_points[:, [0, 1]]
    mask = (warped_points[:, 0] >= 0) & (warped_points[:, 0] < shape[0]) &\
           (warped_points[:, 1] >= 0) & (warped_points[:, 1] < shape[1])
    return points[mask, :], prob[mask, :], des[mask, :]

def select_k_best(points, prob, des, k):
    """ Select the k most probable points (and strip their proba).
    points has shape (num_points, 3) where the last coordinate is the proba. """
    # sorted_prob = points[points[:, 2].argsort(), :2]
    prob = np.squeeze(prob)
    sort = prob.argsort(axis=0)[::-1]  # 从大到小的顺序
    sorted_prob = points[sort, :]
    sorted_des = des[sort, :]

    start = min(k, points.shape[0])
    # return sorted_prob[-start:, :], sorted_des[-start:, :]
    return sorted_prob[:start, :], des[:start, :]


def get_paths_out(path_name, mode=''):
    """
    Return a list of paths to the outputs of the experiment.
    """
    # return glob(osp.join(EXPER_PATH, 'outputs/{}/*.npz'.format(exper_name)))

    # base_path = Path(path_name)
    folders = glob(path_name+'/%s*'%mode)
    out_paths = []
    for folder in folders:
        out = glob(str(folder) + '/*.usp')
        out_paths += out
    out_paths = sorted(out_paths)
    return out_paths