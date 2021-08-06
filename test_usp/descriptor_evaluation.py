import numpy as np
import cv2
from os import path as osp
from glob import glob
from test_usp.util_eval import warp_kpt, filter_keypoints, keep_true_keypoints, select_k_best, get_paths_out


def compute_homography(data, keep_k_points=1000, correctness_thresh=3, orb=False, size_eval=(320,240)):
    """
    Compute the homography between 2 sets of detections and descriptors inside data.
    """
    real_H = data['mat']
    size = data['img_wh'] # original image size

    prob = data['src_score'].reshape(-1, 1)
    keypoints = data['src_point']
    des = data['src_des'].astype('float32')

    warped_prob = data['dst_score'].reshape(-1, 1)
    warped_keypoints = data['dst_point']
    warped_des = data['dst_des'].astype('float32')

    # Keeps only the points shared between the two views
    warped_keypoints, warped_prob, warped_des = keep_true_keypoints(warped_keypoints,
                                                                    warped_prob,
                                                                    warped_des,
                                                                    np.linalg.inv(real_H),
                                                                    size)

    keypoints, prob, des = keep_true_keypoints(keypoints,
                                                prob,
                                                des,
                                                real_H,
                                                size)
    result = {'correctness': 0.,
                'keypoints1': keypoints,
                'keypoints2': warped_keypoints,
                'matches': [],
                'inliers': [],
                'homography': None}
    # Keep only the keep_k_points best predictions
    if warped_keypoints.shape[0] < 1 or keypoints.shape[0] < 1:
        return result


    warped_keypoints, warped_desc = select_k_best(warped_keypoints, warped_prob, warped_des, keep_k_points)
    keypoints, desc = select_k_best(keypoints, prob, des, keep_k_points)


    # Match the keypoints with the warped_keypoints with nearest neighbor search
    if orb:
        desc = desc.astype(np.uint8)
        warped_desc = warped_desc.astype(np.uint8)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc, warped_desc)
    matches_idx = np.array([m.queryIdx for m in matches])
    if len(matches_idx) == 0:  # No match found
        return result
    m_keypoints = keypoints[matches_idx, :]
    matches_idx = np.array([m.trainIdx for m in matches])
    m_warped_keypoints = warped_keypoints[matches_idx, :]

    # mapping to eval size
    scale = 1. / np.asarray(size) * np.asarray(size_eval)
    # m_keypoints = m_keypoints * scale
    # m_warped_keypoints = m_warped_keypoints * scale

    # Estimate the homography between the matches using RANSAC
    H, inliers = cv2.findHomography(m_keypoints[:, [0, 1]],
                                    m_warped_keypoints[:, [0, 1]],
                                    cv2.RANSAC)
    if H is None:
        return {'correctness': 0.,
                'keypoints1': keypoints,
                'keypoints2': warped_keypoints,
                'matches': matches,
                'inliers': inliers,
                'homography': H}

    inliers = inliers.flatten()

    # Compute correctness
    shape = size_eval[::-1]
    corners = np.array([[0,            0],
                        [size[0] - 1, 0],
                        [0,            size[1] - 1],
                        [size[0] - 1, size[1] - 1]])
    real_warped_corners = warp_kpt(corners, real_H)*scale
    warped_corners = warp_kpt(corners, H)*scale
    mean_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))
    correctness = float(mean_dist <= correctness_thresh)

    return {'correctness': correctness,
            'keypoints1': keypoints,
            'keypoints2': warped_keypoints,
            'matches': matches,
            'inliers': inliers,
            'homography': H}

def homography_estimation(eval_out, keep_k_points=1000,
                          correctness_thresh=3, orb=False, size_eval=(320,240), verbose=True):
    """
    Estimates the homography between two images given the predictions.
    The experiment must contain in its output the prediction on 2 images, an original
    image and a warped version of it, plus the homography linking the 2 images.
    Outputs the correctness score.
    """
    paths_usp = get_paths_out(eval_out)
    cur_index = 0
    correctness = []
    for path in paths_usp:
        data = np.load(path)
        estimates = compute_homography(data, keep_k_points, correctness_thresh, orb, size_eval)
        correctness.append(estimates['correctness'])

    if verbose:
        print('******eval homo******')
        print(('keep_k_points:%d, correctness_thresh:%d' % (keep_k_points, correctness_thresh)))
        print("eval image size by W x H: ", size_eval)
        print("correctness is ", np.mean(correctness))
    return np.mean(correctness)


def get_homography_matches(exper_name, keep_k_points=1000,
                           correctness_thresh=3, num_images=1, orb=False):
    """
    Estimates the homography between two images given the predictions.
    The experiment must contain in its output the prediction on 2 images, an original
    image and a warped version of it, plus the homography linking the 2 images.
    Outputs the keypoints shared between the two views,
    a mask of inliers points in the first image, and a list of matches meaning that
    keypoints1[i] is matched with keypoints2[matches[i]]
    """
    paths = get_paths(exper_name)
    outputs = []
    for path in paths[:num_images]:
        data = np.load(path)
        output = compute_homography(data, keep_k_points, correctness_thresh, orb)
        output['image1'] = data['image']
        output['image2'] = data['warped_image']
        outputs.append(output)
    return outputs

if __name__ == '__main__':
    eval_out = '/home/bodong/hpatches_result_sp/'
    correctness_thresh = 1
    keep_k_points = 1000

    correct = homography_estimation(eval_out=eval_out, keep_k_points=keep_k_points,
                                                          correctness_thresh=correctness_thresh, verbose=True)

