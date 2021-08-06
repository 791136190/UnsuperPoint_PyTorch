import os
import torch
import cv2
import numpy as np
import argparse
import time
from Unsuper.utils import common_utils, utils
from Unsuper.configs.config import cfg, cfg_from_list, cfg_from_yaml_file
from symbols.get_model import get_sym


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='../Unsuper/configs/UnsuperPoint_coco.yaml',
                        help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=1, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    # parser.add_argument('--extra_tag', type=str, default='no_score', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    # parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--eval_all', action='store_true', default=True, help='whether to evaluate all checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    # cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg

def main():
    args, cfg = parse_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = '../output/ckpt/checkpoint_epoch_14.pth'

    model = get_sym(model_config=cfg['MODEL'], image_shape=cfg['data']['IMAGE_SHAPE'], is_training=False)

    eval_out_dir = './test_image_log/'
    if os.path.exists(eval_out_dir):
        print('dir exists')
    else:
        os.mkdir(eval_out_dir, 777)

    logger = common_utils.create_logger(eval_out_dir + 'eval_image.txt')

    video_name = 'RollerCoaster'  # car car2 daily fly robot skate under RollerCoaster rotate
    data_root = '../Data/SLAM/video/%s.flv' % video_name
    data_root = '/home/bodong/Downloads/save_img/output.avi'
    cap = cv2.VideoCapture(data_root + '')

    cur_index = 0

    save = False
    # save_name = video_path.replace('flv', 'mp4')
    save_name = './key_point_det_%s.mp4' % video_name
    if save:
        fps = 25
        size = cfg['data']['IMAGE_SHAPE']
        size = (int(size[1] * 2), int(size[0] * 2))
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        video_writer = cv2.VideoWriter(save_name, fourcc, fps, size)

    # sift = cv2.SIFT_create()
    orb = cv2.ORB_create()
    bf_hm = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
    bf = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=True)
    last_s = None
    last_p = None
    last_d = None
    last_img = None
    do_match_test = True
    do_match_num = 10

    with torch.no_grad():
        model.load_params_from_file(filename=ckpt_dir, logger=logger, to_cpu=True)
        # model.cuda()
        model.to(device)

        while cap.isOpened():

            ret, img = cap.read()
            if img is None:
                print('end read video!')
                break

            start_time = time.time()
            cur_index += 1
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            new_h, new_w = cfg['data']['IMAGE_SHAPE']

            resize_img = cv2.resize(img, (new_w, new_h))
            src_img = torch.tensor(resize_img, dtype=torch.float32)
            src_img = torch.unsqueeze(src_img, 0)
            src_img = src_img.permute(0, 3, 1, 2)
            # img0_tensor = src_img.cuda()
            img0_tensor = src_img.to(device)
            pred_dict = model.predict(img0_tensor)
            for j in pred_dict.keys():

                s1 = pred_dict[j]['s1']
                s1 = s1.reshape(-1, 1)
                p1 = pred_dict[j]['p1']
                d1 = pred_dict[j]['d1']
                input = np.concatenate((s1, p1, d1), axis=1)
                keep = utils.key_nms(input, 8)

                s1 = input[keep, 0]
                p1 = input[keep, 1:3]
                d1 = input[keep, 3:]

                loc = np.where(s1 > 0.9)
                s1 = s1[loc]
                p1 = p1[loc]
                d1 = d1[loc]

                end_time = time.time()

                # for k in range(0, min(d1.shape[0], 100)):
                #     for m in range(0, min(d1.shape[0], 100)):
                #         des_1 = d1[k]
                #         des_2 = d1[m]
                #         sim = np.dot(des_1, des_2)
                #         print(k, m, sim)

                img = cv2.cvtColor(resize_img, cv2.COLOR_RGB2BGR)

                if do_match_test and cur_index % do_match_num == 0:
                    if last_img is None:
                        last_s = s1
                        last_p = p1
                        last_d = d1
                        last_img = img.copy()
                    else:
                        # kp1, des1 = sift.detectAndCompute(last_img, None)
                        # kp2, des2 = sift.detectAndCompute(img, None)
                        # matches = bf.match(queryDescriptors=des1, trainDescriptors=des2)
                        kp1, des1 = orb.detectAndCompute(last_img, None)
                        kp2, des2 = orb.detectAndCompute(img, None)
                        matches = bf_hm.match(queryDescriptors=des1, trainDescriptors=des2)
                        img_match = cv2.drawMatches(last_img, kp1, img, kp2, matches[:50], outImg=None, flags=2)
                        img_match_cv = cv2.resize(img_match, (img_match.shape[1] * 2, img_match.shape[0] * 2))

                        # Estimate the homography between the matches using RANSAC
                        # H, inliers = cv2.findHomography(kp1, kp2, cv2.RANSAC)

                        # cv2.imshow('img_match_cv2', img_match)
                        # cv2.waitKey(1)

                        # unsuper
                        matches = bf.match(queryDescriptors=last_d, trainDescriptors=d1)
                        cv_kpts1 = [cv2.KeyPoint(last_p[i][0], last_p[i][1], 1) for i in range(last_p.shape[0])]
                        cv_kpts2 = [cv2.KeyPoint(p1[i][0], p1[i][1], 1) for i in range(p1.shape[0])]
                        img_match = cv2.drawMatches(last_img, cv_kpts1, img, cv_kpts2, matches[:50], outImg=None, flags=2)
                        img_match_usp = cv2.resize(img_match, (img_match.shape[1] * 2, img_match.shape[0] * 2))
                        # Estimate the homography between the matches using RANSAC
                        # H, inliers = cv2.findHomography(cv_kpts1, cv_kpts2, cv2.RANSAC)

                        img_match = np.concatenate((img_match_cv.copy(), img_match_usp.copy()), axis=0)
                        img_match = cv2.resize(img_match, (int(img_match.shape[1] * 0.75), int(img_match.shape[0] * 0.75)))
                        # marg = cv2.hconcat(img_match_cv, img_match_usp)
                        # print(img_match.shape, img_match_cv.shape, img_match_usp.shape)

                        # cv2.imshow('img_match_cv2', img_match_cv)
                        # cv2.imshow('img_match_usp', img_match_usp)
                        cv2.imshow('match', img_match)
                        # cv2.imwrite('match.jpg', img_match)
                        cv2.waitKey(1)

                        last_s = s1
                        last_p = p1
                        last_d = d1
                        last_img = img.copy()

                for i in range(p1.shape[0]):
                    pos = p1[i]
                    cv2.circle(img, (int(pos[0]), int(pos[1])), 1, (0, 0, 255), -1)
                    # if i < 20:
                    #  cv2.putText(img, ('%d' % i), (int(pos[0]), int(pos[1]+10)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
                cv2.putText(img, ('%d, %.1f ms' % (cur_index, (end_time - start_time) * 1000)), (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2))
                cv2.imshow('key point', img)
                key = cv2.waitKey(1)
                if key & 0xff == ord('q'):# or cur_index > 10:
                    cv2.destroyAllWindows()
                    if save:
                        video_writer.release()
                    cap.release()
                    exit(0)
                # cv2.imwrite(eval_out_dir + 'eval' + '.jpg', img)
            if save:
                video_writer.write(img)
    if save:
        video_writer.release()
    cap.release()

if __name__ == '__main__':
    main()
    print('end all process!!')

