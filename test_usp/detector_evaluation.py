# -*- coding: utf-8 -*-

import numpy as np
from os import path as osp
from glob import glob
from pathlib import Path
import re
import os
import cv2
from test_usp.util_eval import warp_kpt, filter_keypoints, keep_true_keypoints, select_k_best, get_paths_out

eval_root_path = '../output/eval/eval_all/'  # epoch_2/HPatch/
data_root_path = '/opt/train_data/COCO_data/'  # HPatch/


def get_paths_data(path_name):
    """
    Return a list of paths to the outputs of the experiment.
    """
    # return glob(osp.join(EXPER_PATH, 'outputs/{}/*.npz'.format(exper_name)))

    base_path = Path(path_name)
    folders = list(base_path.iterdir())
    out_paths = []
    for folder in folders:
        out = glob(str(folder) + '/H_*')
        out_paths += out
    out_paths = sorted(out_paths)
    return out_paths

def compute_tp_fp(data, remove_zero=1e-4, distance_thresh=2, simplified=False):
    """
    Compute the true and false positive rates.
    """
    # Read data
    gt = np.where(data['keypoint_map'])
    gt = np.stack([gt[0], gt[1]], axis=-1)
    n_gt = len(gt)
    prob = data['prob_nms'] if 'prob_nms' in data.files else data['prob']

    # Filter out predictions with near-zero probability
    mask = np.where(prob > remove_zero)
    prob = prob[mask]
    pred = np.array(mask).T

    # When several detections match the same ground truth point, only pick
    # the one with the highest score  (the others are false positive)
    sort_idx = np.argsort(prob)[::-1]
    prob = prob[sort_idx]
    pred = pred[sort_idx]

    diff = np.expand_dims(pred, axis=1) - np.expand_dims(gt, axis=0)
    dist = np.linalg.norm(diff, axis=-1)
    matches = np.less_equal(dist, distance_thresh)

    tp = []
    matched = np.zeros(len(gt))
    for m in matches:
        correct = np.any(m)
        if correct:
            gt_idx = np.argmax(m)
            tp.append(not matched[gt_idx])
            matched[gt_idx] = 1
        else:
            tp.append(False)
    tp = np.array(tp, bool)
    if simplified:
        tp = np.any(matches, axis=1)  # keeps multiple matches for the same gt point
        n_gt = np.sum(np.minimum(np.sum(matches, axis=0), 1))  # buggy
    fp = np.logical_not(tp)
    return tp, fp, prob, n_gt


def div0(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        idx = ~np.isfinite(c)
        c[idx] = np.where(a[idx] == 0, 1, 0)  # -inf inf NaN
    return c


def compute_pr(exper_name, **kwargs):
    """
    Compute precision and recall.
    """
    # Gather TP and FP for all files
    paths = get_paths_out(exper_name)
    tp, fp, prob, n_gt = [], [], [], 0
    for path in paths:
        t, f, p, n = compute_tp_fp(np.load(path), **kwargs)
        tp.append(t)
        fp.append(f)
        prob.append(p)
        n_gt += n
    tp = np.concatenate(tp)
    fp = np.concatenate(fp)
    prob = np.concatenate(prob)

    # Sort in descending order of confidence
    sort_idx = np.argsort(prob)[::-1]
    tp = tp[sort_idx]
    fp = fp[sort_idx]
    prob = prob[sort_idx]

    # Cumulative
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = div0(tp_cum, n_gt)
    precision = div0(tp_cum, tp_cum + fp_cum)
    recall = np.concatenate([[0], recall, [1]])
    precision = np.concatenate([[0], precision, [0]])
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    return precision, recall, prob


def compute_mAP(precision, recall):
    """
    Compute average precision.
    """
    return np.sum(precision[1:] * (recall[1:] - recall[:-1]))


def compute_loc_error(eval_out, prob_thresh=0.5, distance_thresh=2):
    """
    Compute the localization error.
    """
    def loc_error_per_image(data):
        # Read data
        gt = np.where(data['keypoint_map'])
        gt = np.stack([gt[0], gt[1]], axis=-1)
        prob = data['prob']

        # Filter out predictions
        mask = np.where(prob > prob_thresh)
        pred = np.array(mask).T
        prob = prob[mask]

        if not len(gt) or not len(pred):
            return []

        diff = np.expand_dims(pred, axis=1) - np.expand_dims(gt, axis=0)
        dist = np.linalg.norm(diff, axis=-1)
        dist = np.min(dist, axis=1)
        correct_dist = dist[np.less_equal(dist, distance_thresh)]
        return correct_dist
    paths = get_paths_out(eval_out)
    error = []
    for path in paths:
        error.append(loc_error_per_image(np.load(path)))
    return np.mean(np.concatenate(error))

def compute_repeatability(eval_out, eval_data='', keep_k_points=300,
                          distance_thresh=3, size_eval=[320,240], verbose=False):
    """
    Compute the repeatability. The experiment must contain in its output the prediction
    on 2 images, an original image and a warped version of it, plus the homography
    linking the 2 images.
    """
    paths_usp = get_paths_out(eval_out, mode=eval_data)
    repeatability = []
    error = []
    sim = []
    N1s = []
    N2s = []
    cur_index = 0
    # bf = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=True)

    for path in paths_usp:
        # name = path.split('/')[-2]
        # num_list = re.findall(r'\d+', path.split('/')[-1])[0]
        # h_data_path = eval_data + name + '/' + 'H_1_' + num_list
        cur_index += 1

        # H = None
        # if os.path.exists(h_data_path):
        #     print(h_data_path)
        #     H = np.loadtxt(h_data_path)
        #
        #     data = np.load(path)
        #     warped_prob = data['scores'].reshape(-1, 1)
        #     warped_keypoints = data['keypoints']
        #     shape = data['imsize']
        #     # warped_keypoints = np.stack([warped_keypoints, warped_prob], axis=-1)
        # else:
        #     data = np.load(path)
        #     prob = data['scores'].reshape(-1, 1)
        #     keypoints = data['keypoints']
        #     shape = data['imsize']
        #
        #     continue

        data = np.load(path)
        prob = data['src_score'].reshape(-1, 1)
        keypoints = data['src_point']
        des = data['src_des']

        warped_prob = data['dst_score'].reshape(-1, 1)
        warped_keypoints = data['dst_point']
        warped_des = data['dst_des']

        H = data['mat']

        size = data['img_wh']

        # print('cur_index:', cur_index)
        # shape = data['warped_prob'].shape
        # H = data['homography']
        # if cur_index == 461:
        #     print(cur_index)
        # Filter out predictions
        # keypoints = np.where(data['prob'] > 0)
        # prob = data['prob'][keypoints[0], keypoints[1]]
        # keypoints = np.stack([keypoints[0], keypoints[1]], axis=-1)
        # warped_keypoints = np.where(data['warped_prob'] > 0)
        # warped_prob = data['warped_prob'][warped_keypoints[0], warped_keypoints[1]]
        # warped_keypoints = np.stack([warped_keypoints[0],
        #                              warped_keypoints[1],
        #                              warped_prob], axis=-1)
        warped_keypoints, warped_prob, warped_des = keep_true_keypoints(warped_keypoints,
                                                                        warped_prob,
                                                                        warped_des,
                                                                        np.linalg.inv(H),
                                                                        size)
        # warped_keypoints = keep_true_keypoints(warped_keypoints, H, shape)
        # warped_keypoints, warped_prob, warped_des = filter_keypoints(warped_keypoints, warped_prob, warped_des, shape)

        # Warp the original keypoints with the true homography
        true_warped_keypoints = warp_kpt(keypoints[:, [0, 1]], H)
        # true_warped_keypoints = np.stack([true_warped_keypoints[:, 1], true_warped_keypoints[:, 0], prob], axis=-1)
        true_warped_keypoints, prob, des = filter_keypoints(true_warped_keypoints, prob, des, size)

        # Keep only the keep_k_points best predictions
        if warped_keypoints.shape[0] < 1 or true_warped_keypoints.shape[0] < 1:
            continue

        warped_keypoints, warped_des = select_k_best(warped_keypoints, warped_prob, warped_des, keep_k_points)
        true_warped_keypoints, des = select_k_best(true_warped_keypoints, prob, des, keep_k_points)

        # Compute the repeatability
        N1 = true_warped_keypoints.shape[0]
        N2 = warped_keypoints.shape[0]
        N1s.append(N1)
        N2s.append(N2)

        true_warped_keypoints = 1.*true_warped_keypoints/np.asarray(size)*np.asarray(size_eval)
        warped_keypoints = 1.*warped_keypoints/np.asarray(size)*np.asarray(size_eval)

        true_warped_keypoints = np.expand_dims(true_warped_keypoints, 1)
        warped_keypoints = np.expand_dims(warped_keypoints, 0)
        # shapes are broadcasted to N1 x N2 x 2:
        norm = np.linalg.norm(true_warped_keypoints - warped_keypoints, ord=None, axis=2)
        count1 = 0
        count2 = 0
        local_err1, local_err2 = None, None
        if N2 != 0:
            min1 = np.min(norm, axis=1)
            count1 = np.sum(min1 <= distance_thresh)
            local_err1 = min1[min1 <= distance_thresh]
        if N1 != 0:
            min2 = np.min(norm, axis=0)
            count2 = np.sum(min2 <= distance_thresh)
            local_err2 = min2[min2 <= distance_thresh]
        if N1 + N2 > 0:
            repeatability.append((count1 + count2) / (N1 + N2))

        diff = true_warped_keypoints - warped_keypoints
        dist = np.linalg.norm(diff, axis=-1)
        dist = np.min(dist, axis=1)
        correct_dist = dist[np.less_equal(dist, distance_thresh)]
        # error.append(correct_dist)

        # warped_des = np.squeeze(warped_des)
        # des = np.squeeze(des)

        l1 = warped_des.dot(warped_des.T) / np.outer(np.linalg.norm(warped_des, axis=1), np.linalg.norm(warped_des, axis=1))
        l1 = l1 - np.eye(l1.shape[0])

        l2 = des.dot(des.T) / np.outer(np.linalg.norm(des, axis=1), np.linalg.norm(des, axis=1))
        l2 = l2 - np.eye(l2.shape[0])

        sim.append((np.mean(l1) + np.mean(l2)) * 0.5)

        if count1 + count2 > 0:
            localization_err = 0
            if local_err1 is not None:
                localization_err += (local_err1.sum())/ (count1 + count2)
            if local_err2 is not None:
                localization_err += (local_err2.sum())/ (count1 + count2)
            error.append(localization_err)
    if verbose:
        print("eval image size by W x H: ", size_eval)
        print("Average number of points in the first image: " + str(np.mean(N1s)))
        print("Average number of points in the second image: " + str(np.mean(N2s)))
    # return np.mean(repeatability), np.mean(np.concatenate(error)), np.mean(sim)
    return np.mean(repeatability), np.mean(error), np.mean(sim)

if __name__ == '__main__':

    start_epoch = 0
    end_epoch = 0
    print('start process epoch:%d -> %d' % (start_epoch, end_epoch))

    for i in range(start_epoch, end_epoch + 1):
        eval_epoch = i
        eval_topk = 300
        eval_dis = 3

        eval_out = eval_root_path + ('epoch_%d' % eval_epoch) + '/HPatch/'
        eval_data = data_root_path + 'hpatches-sequences-release/'
        eval_out = '/home/bodong/hpatches_result/'
        if not os.path.exists(eval_out):
            continue

        print(('\neval_epoch:%d, eval_topk:%d, eval_dis:%d' % (eval_epoch, eval_topk, eval_dis)))

        repeatability, loc_error, sim = compute_repeatability(eval_out=eval_out, eval_data='v', keep_k_points=eval_topk, distance_thresh=eval_dis, verbose=True)
        print('repeatability:', repeatability)
        print('loc l1 error pixel:', loc_error)
        print('cosine similarity rang[-1, 1]:', sim)

    print('\nend all process!!')
