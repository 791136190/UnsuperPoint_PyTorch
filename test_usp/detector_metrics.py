import numpy as np
from os import path as osp
from glob import glob

def warp_keypoints(keypoints, H):
    """
    :param keypoints:
    points:
        numpy (N, (x,y))
    :param H:
    :return:
    """
    num_points = keypoints.shape[0]
    homogeneous_points = np.concatenate([keypoints, np.ones((num_points, 1))],
                                        axis=1)
    warped_points = np.dot(homogeneous_points, np.transpose(H))
    return warped_points[:, :2] / warped_points[:, 2:]


def compute_repeatability(keypoints, warped_keypoints, homography, width, height, keep_k_points=300, distance_thresh=3):
    """ Compute the repeatability. The experiment must contain in its output the prediction
    on 2 images, an original image and a warped version of it, plus the homography
    linking the 2 images.

    Args:
        keypoints ([type np.ndarray]): shape N x 3 . Like [[x, y, score], ...]
        warped_keypoints ([type np.ndarray]): shape  N x 3
        homography ([type np.ndarray]): 3 x 3
        width ([type int]): width of the image
        height ([type int]): height of the image
        keep_k_points (int, optional):  Defaults to 300.
        distance_thresh (int, optional):  Defaults to 3.
    """

    def filter_keypoints(points, shape):
        """ Keep only the points whose coordinates are
        inside the dimensions of shape. """
        """
        points:
            numpy (N, (x,y))
        shape:
            (y, x)
        """
        mask = (points[:, 0] >= 0) & (points[:, 0] < shape[1]) &\
               (points[:, 1] >= 0) & (points[:, 1] < shape[0])
        return points[mask, :]

    def keep_true_keypoints(points, H, shape):
        """ Keep only the points whose warped coordinates by H
        are still inside shape. """
        """
        input:
            points: numpy (N, (x,y))
            shape: (height, width)
        return:
            points: numpy (N, (x,y))
        """
        # warped_points = warp_keypoints(points[:, [1, 0]], H)
        warped_points = warp_keypoints(points[:, [0, 1]], H)
        # warped_points[:, [0, 1]] = warped_points[:, [1, 0]]
        mask = (warped_points[:, 0] >= 0) & (warped_points[:, 0] < shape[1]) &\
               (warped_points[:, 1] >= 0) & (warped_points[:, 1] < shape[0])
        return points[mask, :]

    def select_k_best(points, k):
        """ Select the k most probable points (and strip their proba).
        points has shape (num_points, 3) where the last coordinate is the proba. """
        sorted_prob = points
        if points.shape[1] > 2:
            sorted_prob = points[points[:, 2].argsort(), :2]
            start = min(k, points.shape[0])
            sorted_prob = sorted_prob[-start:, :]
        return sorted_prob

    # paths = get_paths(exper_name)
    localization_err = -1
    repeatability = []
    N1s = []
    N2s = []
    # for path in paths:
    # data = np.load(path)
    shape = (height, width)
    H = homography

    # Filter out predictions
    # keypoints = np.where(data['prob'] > 0)
    # prob = data['prob'][keypoints[0], keypoints[1]]
    # keypoints = np.stack([keypoints[0], keypoints[1]], axis=-1)
    # warped_keypoints = np.where(data['warped_prob'] > 0)
    # warped_prob = data['warped_prob'][warped_keypoints[0], warped_keypoints[1]]
    # warped_keypoints = np.stack([warped_keypoints[0],
    #                              warped_keypoints[1],
    #                              warped_prob], axis=-1)
    # keypoints = data['prob'][:, :2]
    # warped_keypoints = data['warped_prob'][:, :2]
    keypoints = keypoints.copy()
    warped_keypoints = warped_keypoints.copy()
    
    warped_keypoints = keep_true_keypoints(warped_keypoints, np.linalg.inv(H), shape)

    # Warp the original keypoints with the true homography
    true_warped_keypoints = keypoints
    # true_warped_keypoints[:,:2] = warp_keypoints(keypoints[:, [1, 0]], H)
    true_warped_keypoints[:,:2] = warp_keypoints(keypoints[:, :2], H) # make sure the input fits the (x,y)
    # true_warped_keypoints = np.stack([true_warped_keypoints[:, 1],
    #                                   true_warped_keypoints[:, 0],
    #                                   prob], axis=-1)
    true_warped_keypoints = filter_keypoints(true_warped_keypoints, shape)

    # Keep only the keep_k_points best predictions
    warped_keypoints = select_k_best(warped_keypoints, keep_k_points)
    true_warped_keypoints = select_k_best(true_warped_keypoints, keep_k_points)

    # Compute the repeatability
    N1 = true_warped_keypoints.shape[0]
    # print('true_warped_keypoints: ', true_warped_keypoints[:2,:])
    N2 = warped_keypoints.shape[0]
    # print('warped_keypoints: ', warped_keypoints[:2,:])
    N1s.append(N1)
    N2s.append(N2)
    true_warped_keypoints = np.expand_dims(true_warped_keypoints, 1)
    warped_keypoints = np.expand_dims(warped_keypoints, 0)
    # shapes are broadcasted to N1 x N2 x 2:
    norm = np.linalg.norm(true_warped_keypoints - warped_keypoints,
                          ord=None, axis=2) # N1 x N2
    count1 = 0
    count2 = 0
    local_err1, local_err2 = None, None
    if N2 != 0:
        min1 = np.min(norm, axis=1)
        count1 = np.sum(min1 <= distance_thresh)
        # print("count1: ", count1)
        local_err1 = min1[min1 <= distance_thresh]
        # print("local_err1: ", local_err1)
    if N1 != 0:
        min2 = np.min(norm, axis=0)
        count2 = np.sum(min2 <= distance_thresh)
        local_err2 = min2[min2 <= distance_thresh]

    if N1 + N2 > 0:
        # repeatability.append((count1 + count2) / (N1 + N2))
        repeatability = (count1 + count2) / (N1 + N2)
    if count1 + count2 > 0:
        localization_err = 0
        if local_err1 is not None:
            localization_err += (local_err1.sum())/ (count1 + count2)
        if local_err2 is not None:
            localization_err += (local_err2.sum())/ (count1 + count2)
    else:
        repeatability = 0
    # if verbose:
    #     print("Average number of points in the first image: " + str(np.mean(N1s)))
    #     print("Average number of points in the second image: " + str(np.mean(N2s)))
    # return np.mean(repeatability)
    return repeatability, localization_err
