import os
import torch
import glob
import tqdm
import re
import numpy as np
import datetime
import argparse
from pathlib import Path
from Unsuper.dataset import build_dataloader
from Unsuper.utils import common_utils
from Unsuper.configs.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from symbols.get_model import get_sym

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='./Unsuper/configs/UnsuperPoint_coco.yaml', help='specify the config for training')

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

def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    model.eval()
    iter_step = len(dataloader)
    dataloader_iter = iter(dataloader)
    
    progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    for i in range(iter_step):
        try:
            img0, img0_tensor, imgidx = next(dataloader_iter)
        except StopIteration:
            break

        h, w = img0[0].shape[:2]

        img0_tensor = img0_tensor.cuda()
        pred_dict = model.predict(img0_tensor)
        for j in pred_dict.keys():
            suffix = imgidx[j][imgidx[j].rfind('.'):]
            img_path = Path(imgidx[j])
            folder = img_path.parent.stem
            img_name = img_path.stem
            data_dir = result_dir / cfg['data']['export_name'] / folder
            data_dir.mkdir(parents=True, exist_ok=True)

            # for vis
            # s1 = pred_dict[j]['s1']
            # loc = np.where(s1 > 0.7)
            # p1 = pred_dict[j]['p1'][loc]
            # d1 = pred_dict[j]['d1'][loc]
            # s1 = s1[loc]
            # print(p1)
            # img0[j] = cv2.cvtColor(img0[j], cv2.COLOR_RGB2BGR)
            # for i in range(p1.shape[0]):
            #     pos = p1[i]
            #     cv2.circle(img0[j], (int(pos[0]), int(pos[1])), 1, (0, 0, 255), 1)
            # cv2.imwrite(os.path.join(str(data_dir), img_name + '.jpg'), img0[j])

            # for eval
            # cv2.imwrite(os.path.join(str(data_dir), img_name+suffix), img0[j])
            s1 = pred_dict[j]['s1']
            loc = np.where(s1 > 0.5)
            p1 = pred_dict[j]['p1'][loc]
            d1 = pred_dict[j]['d1'][loc]
            s1 = s1[loc]
            # np.savez(open(os.path.join(str(data_dir), img_name+'.ppm.Unsuper'+epoch_id), 'wb'), scores=s1, keypoints=p1, descriptors=d1, imsize=(h, w))
            np.savez(open(os.path.join(str(data_dir), img_name + '.ppm.usp'), 'wb'), scores=s1, keypoints=p1, descriptors=d1, imsize=(w, h))

        progress_bar.update()

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')


def eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=False):
    # load checkpoint
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test)
    model.cuda()

    # start evaluation
    eval_one_epoch(
        cfg, model, test_loader, epoch_id, logger, dist_test=dist_test,
        result_dir=eval_output_dir, save_to_file=args.save_to_file
    )

def repeat_eval_ckpt(cfg, model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=False):

    # tensorboard log
    # tb_log = SummaryWriter(log_dir=str(eval_output_dir / ('tensorboard_%s' % cfg.DATA_CONFIG.DATA_SPLIT['test'])))
    ckpt_list = glob.glob(os.path.join(ckpt_dir, '*checkpoint_epoch_*.pth'))
    ckpt_list.sort(key=os.path.getmtime)
    for cur_ckpt in ckpt_list:

        num_list = re.findall('checkpoint_epoch_(.*).pth', cur_ckpt)
        if num_list.__len__() == 0:
            continue
        epoch_id = num_list[-1]

        model.load_params_from_file(filename=cur_ckpt, logger=logger, to_cpu=dist_test)
        model.cuda()

        # start evaluation
        cur_result_dir = eval_output_dir / ('epoch_%s' % epoch_id)
        eval_one_epoch(
            cfg, model, test_loader, epoch_id, logger, dist_test=dist_test,
            result_dir=cur_result_dir
        )

        logger.info('Epoch %s has been evaluated' % epoch_id)


def main():
    dist_test = False
    args, cfg = parse_config()

    output_dir = cfg.ROOT_DIR / 'output' #/ cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / 'eval'

    if not args.eval_all:
        num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
        epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
        eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / 'test'
    else:
        eval_output_dir = eval_output_dir / 'eval_all'

    # if args.eval_tag is not None:
    #     eval_output_dir = eval_output_dir / args.eval_tag

    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else output_dir / 'ckpt'

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg['data'],
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )

    model = get_sym(model_config=cfg['MODEL'], image_shape=cfg['data']['IMAGE_SHAPE'], is_training=False)# build_network(cfg['MODEL'], cfg['data']['IMAGE_SHAPE'], False)
    with torch.no_grad():
        if args.eval_all:
            repeat_eval_ckpt(cfg, model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=dist_test)
        else:
            eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=dist_test)

if __name__ == '__main__':
    main()

