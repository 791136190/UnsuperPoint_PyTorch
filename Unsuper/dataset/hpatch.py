import torch
import torch.nn
import glob
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms as tfs

from .base_dataset import BaseDataset
from ..utils.utils import resize_img, enhance

class HPatchDataset(BaseDataset):
    default_config = {}

    def init_dataset(self):
        self.name = 'hpatch'
        base_path = Path('../Data', 'HPatch')
        folders = list(base_path.iterdir())
        img_paths = []
        for folder in folders:
            imgs = glob.glob(str(folder)+'/*.ppm')
            img_paths += imgs
        data_len = len(img_paths)
        return data_len, img_paths

    # def __getitem__(self, index):
    #     if self.is_training:
    #         raise NotImplementedError
    #     else:
    #         img_file = self.train_files[index]
    #         img = cv2.imread(img_file)
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #         new_h, new_w = self.config['IMAGE_SHAPE']
    #         src_img = cv2.resize(img, (new_w, new_h))
    #         return src_img, img_file
    #
    # def test_collate_batch(*batches):
    #     img = []
    #     img_idx = []
    #     for batch in batches[1]:
    #         img.append(batch[0])
    #         img_idx.append(batch[1])
    #     img_src = torch.tensor(img, dtype=torch.float32)
    #
    #     return img, img_src.permute(0, 3, 1, 2), img_idx
    def __getitem__(self, index):
        img_file = self.train_files[index]
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        src_img = resize_img(img, self.config['IMAGE_SHAPE'])  # reshape the image
        dst_img, mat = enhance(src_img, self.config)

        # cv2 -> PIL
        src_img = Image.fromarray(src_img)  # rgb 顺序
        dst_img = Image.fromarray(dst_img)  # rgb 顺序

        src_img = tfs.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3, hue=0.2)(src_img)
        src_img = tfs.RandomGrayscale(p=0.1)(src_img)

        dst_img = tfs.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3, hue=0.2)(dst_img)
        dst_img = tfs.RandomGrayscale(p=0.1)(dst_img)

        # PIL -> cv2
        src_img = np.asarray(src_img)
        dst_img = np.asarray(dst_img)

        return src_img, dst_img, mat, img_file

    def test_collate_batch(*batches):
        src_img = []
        dst_img = []
        mat = []
        img_idx = []
        for batch in batches[1]:
            src_img.append(batch[0])
            dst_img.append(batch[1])
            mat.append(batch[2])
            img_idx.append(batch[3])
        src_img = torch.tensor(src_img, dtype=torch.float32)  # B * H * W * C
        dst_img = torch.tensor(dst_img, dtype=torch.float32)  # B * H * W * C
        mat = torch.tensor(mat, dtype=torch.float32, requires_grad=False).squeeze()  # B * 3 * 3
        return src_img.permute(0, 3, 1, 2), dst_img.permute(0, 3, 1, 2), mat, img_idx
