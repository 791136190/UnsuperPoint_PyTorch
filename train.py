import os
import sys
sys.path.append('./')
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from Unsuper.configs.config import cfg, log_config_to_file, cfg_from_list, cfg_from_yaml_file
from Unsuper.utils import common_utils
from Unsuper.dataset import build_dataloader
from Unsuper.utils.train_utils import build_optimizer, build_scheduler
from symbols.model_base import ModelTemplate
from Unsuper.utils.train_utils import train_model
import torch.distributed as dist
os.environ['CUDA_VISIBLE_DEVICES'] = ""
from pathlib import Path
import argparse
import datetime

from symbols.get_model import get_sym

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='./Unsuper/configs/UnsuperPoint_coco.yaml', help='specify the config for training')

    # parser.add_argument('--batch_size', type=int, default=32, required=False, help='batch size for training')
    # parser.add_argument('--epochs', type=int, default=20, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=12, help='number of workers for dataloader')
    # parser.add_argument('--extra_tag', type=str, default='no_score', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=True, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
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
    if args.launcher == 'none':
        dist_train = False
        args.batch_size = cfg['data']['batch_size']
    else:
        args.batch_size, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.batch_size, args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True
    if args.fix_random_seed:
        common_utils.set_random_seed(233)

    output_dir = cfg.ROOT_DIR / '../output'  # / cfg.TAG / args.extra_tag

    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'

    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        total_gpus = dist.get_world_size()
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    # PS D:\Program Files\tf_env\Scripts> .\tensorboard.exe --logdir=X:\project\UnsuperPoint\output\tensorboard --host=127.0.0.1 --port=8888
    # http://127.0.0.1:8888
    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None
    # tb_log = None
    # -----------------------create dataloader & network & optimizer---------------------------
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg['data'],
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True
    )

    model = get_sym(model_config=cfg['MODEL'], image_shape=cfg['data']['IMAGE_SHAPE'], is_training=True) #build_network(cfg['MODEL'], cfg['data']['IMAGE_SHAPE'], False)
    # model = build_LightningNetwork(cfg['MODEL'], cfg['data']['IMAGE_SHAPE'], False)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    optimizer = build_optimizer(model, cfg['MODEL']['OPTIMIZATION'])

    # load checkpoint if it is possible
    start_epoch = it = 0
    last_epoch = -1
    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist, logger=logger)

    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1
    else:
        # ckpt_list = glob.glob(str(ckpt_dir / '*checkpoint_epoch_*.pth'))
        # if len(ckpt_list) > 0:
        #     ckpt_list.sort(key=os.path.getmtime)
        #     it, start_epoch = model.load_params_with_optimizer(
        #         ckpt_list[-1], to_cpu=dist, optimizer=optimizer, logger=logger
        #     )
        #     last_epoch = start_epoch + 1
        # elif not dist_train:
        #     model.apply(ModelTemplate.init_weights)
        model.apply(ModelTemplate.init_weights)
        torch.nn.init.normal_(model.score[3].weight)
        torch.nn.init.normal_(model.position[3].weight)
        torch.nn.init.normal_(model.descriptor[3].weight)
    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
    logger.info(model)

    total_iters_each_epoch = len(train_loader)
    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=total_iters_each_epoch, total_epochs=cfg['MODEL']['OPTIMIZATION']['EPOCHS'],
        last_epoch=last_epoch, optim_cfg=cfg['MODEL']['OPTIMIZATION']
    )

    # -----------------------start training---------------------------
    logger.info('**********************Start training %s **********************' % (cfg.EXP_GROUP_PATH))

    train_model(
        model,
        optimizer,
        train_loader,
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg['MODEL']['OPTIMIZATION'],
        start_epoch=start_epoch,
        total_epochs=cfg['MODEL']['OPTIMIZATION']['EPOCHS'],
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        cfg=cfg
    )

    logger.info('**********************End training %s **********************\n\n\n' % (cfg.EXP_GROUP_PATH))


if __name__ == '__main__':
    main()
