from test_usp.detector_evaluation import compute_repeatability
from test_usp.descriptor_evaluation import homography_estimation
import glob
from PIL import Image
import numpy as np
import os
import torch
import cv2
import tqdm
from Unsuper.utils.utils import key_nms

def inference(model, image_list, eval_out, shape, border_remove):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    with torch.no_grad():
        for image_idx, image_path in enumerate(image_list):
            ls, ss, ds = [], [], []
            whs = []
            mats = [[]]
            fname = os.path.split(image_path)[1]
            for i in range(1, 7):
                img = Image.open(image_path + '/%d.ppm' % i)
                new_h, new_w = shape

                resize_img = img.resize((new_w, new_h), Image.NEAREST)
                resize_img = np.asarray(resize_img)
                # np.save('./test_img.npy',resize_img)
                src_img = torch.tensor(resize_img, dtype=torch.float32)
                src_img = torch.unsqueeze(src_img, 0)
                src_img = src_img.permute(0, 3, 1, 2)
                img0_tensor = src_img.to(device)

                pred_dict = model.predict(img0_tensor)
                j = 0
                s1_src = pred_dict[j]['s1']
                s1_src = s1_src.reshape(-1,1)
                p1_src = pred_dict[j]['p1']
                d1_src = pred_dict[j]['d1']
                loc_src = np.where(s1_src[..., 0] > 0.5)
                s1_src = s1_src[loc_src]
                p1_src = p1_src[loc_src]
                d1_src = d1_src[loc_src]
                input = np.concatenate((s1_src, p1_src, d1_src), axis=1)
                keep = key_nms(input, 4)
                s1_src = input[keep, 0]
                p1_src = input[keep, 1:3]
                d1_src = input[keep, 3:]
                # remove border point
                toremoveW = np.logical_or(p1_src[:, 0] < border_remove, p1_src[:, 0] >= (shape[1] - border_remove))
                toremoveH = np.logical_or(p1_src[:, 1] < border_remove, p1_src[:, 1] >= (shape[0] - border_remove))
                toremove = np.logical_or(toremoveW, toremoveH)
                s1_src = s1_src[~toremove]
                p1_src = p1_src[~toremove]
                d1_src = d1_src[~toremove]

                p1_src[:,1] = 1.0*p1_src[:,1] / shape[0] * img.size[1]
                p1_src[:,0] = 1.0*p1_src[:,0] / shape[1] * img.size[0]
                ls.append(p1_src)
                ss.append(s1_src)
                ds.append(d1_src)
                whs.append(img.size)

                if i > 1:
                    mats.append(np.genfromtxt(image_path + '/H_%d_%d' % (1, i)))

            new_f = os.path.join(eval_out, fname)
            if not os.path.exists(new_f):
                os.makedirs(new_f)

            for i in range(5):
                np.savez(open(os.path.join(new_f, str(i + 1) + '.ppm.usp'), 'wb'), src_score=ss[0], src_point=ls[0],
                         src_des=ds[0],
                         dst_score=ss[i + 1], dst_point=ls[i + 1], dst_des=ds[i + 1], mat=mats[i + 1], img_wh=whs[i + 1])


def evaluation(model, eval_topk=300, eval_dis=3, shape=[240,320],border_remove=4,
               val_path='/home/bodong/hpatches-sequences-release/*', eval_out='./output/hpatches-result/'):

    image_list = sorted(glob.glob(val_path))

    # output image list result
    inference(model, image_list, eval_out, shape, border_remove)

    # for illumination
    print('******illumination******')
    repeatability1, loc_error1, sim1 = compute_repeatability(eval_out=eval_out, eval_data='i',
                                                          keep_k_points=eval_topk, distance_thresh=eval_dis,
                                                          verbose=True)
    print('******view point******')
    # for view point change
    repeatability2, loc_error2, sim2 = compute_repeatability(eval_out=eval_out, eval_data='v',
                                                          keep_k_points=eval_topk, distance_thresh=eval_dis,
                                                          verbose=True)


    # calculate homo
    correctness_thresh = [1,3,5]
    correctness = []
    for thres in correctness_thresh:
        result = homography_estimation(eval_out, correctness_thresh=thres)
        correctness.append(result)
    return repeatability2, loc_error2, sim2, correctness


if __name__ == '__main__':
    from symbols.get_model import get_sym
    from Unsuper.configs.config import cfg, cfg_from_yaml_file
    from Unsuper.utils import common_utils
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_from_yaml_file('./Unsuper/configs/UnsuperPoint_coco.yaml', cfg)

    model = get_sym(model_config=cfg['MODEL'], image_shape=cfg['data']['IMAGE_SHAPE'], is_training=False)
    logger = common_utils.create_logger('./test_usp/test_image_log/eval_image.txt')
    model.load_params_from_file(filename='/home/bodong/playground/detection/UnsuperPoint_PyTorch/output/ckpt/checkpoint_epoch_3_rpt_0.616_corr_0.664.pth',
                                logger=logger,
                                to_cpu=True,
                                )

    evaluation(model, shape=cfg['data']['IMAGE_SHAPE'],
               val_path='/home/bodong/hpatches-sequences-release/*',
               eval_out='/home/bodong/hpatches_result')