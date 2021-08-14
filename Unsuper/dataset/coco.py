import torch
import torch.nn
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter
import random
from torchvision import transforms as tfs

from .base_dataset import BaseDataset
from ..utils.utils import resize_img, enhance


class PilGaussianBlur(ImageFilter.Filter):
    name = "GaussianBlur"

    def __init__(self, radius=2):
        self.radius = radius

    def filter(self, image):
        return image.gaussian_blur(self.radius)

def motion_blur(image, degree=12, angle=45):
    image = np.array(image)

    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)

    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred

def gasuss_noise(image, mean=0, var=0.001):
    '''
        添加高斯噪声
        mean : 均值
        var : 方差
    '''
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    out = np.clip(out, 0.0, 1.0)
    out = np.uint8(out * 255)
    return out

class COCODataset(BaseDataset):
    default_config = {}

    def init_dataset(self):
        self.name = 'coco'
        if self.is_training:
            base_path = Path(self.config['train_path'], 'train2017/')
            image_paths = list(base_path.iterdir())
            image_paths = [str(p) for p in image_paths]
            np.random.shuffle(image_paths)
            data_len = len(image_paths)
            if self.config['truncate']:
                base = round(data_len * self.config['truncate']) // self.config['batch_size']  # 对齐N倍batch size
                base = base * self.config['batch_size']
                image_paths = image_paths[:base]
            return len(image_paths), image_paths
        else:
            base_path = Path(self.config['train_path'], 'val2017/')
            image_paths = list(base_path.iterdir())
            test_files = [str(p) for p in image_paths][:self.config['export_size']]
            return self.config['export_size'], test_files
    
    def __getitem__(self, index):
        img_file = self.train_files[index]
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = jpeg.JPEG(img_file).decode()  # jepg4py 默认就是RGB通道了

        if self.is_training:

            # *****************************//原始图像增强 翻转，旋转，运动模糊，高斯噪声//*************************
            seed = random.random()
            if seed < 0.5:
                img = cv2.flip(img, flipCode=random.randint(-1, 1))

            seed = random.random()
            if seed < 0.25:
                (h, w) = img.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, random.randint(-45, 45), random.randint(7, 13) * 0.1)
                img = cv2.warpAffine(img, M, (w, h))

            seed = random.random()
            if seed < 0.25:
                img = motion_blur(img, degree=random.randint(5, 10), angle=random.randint(-45, 45))

            seed = random.random()
            if seed < 0.25:
                img = gasuss_noise(img, mean=0, var=0.001)

            # *****************************//原始图像随机裁剪到训练大小，图像变换到新图//*************************
            src_img = resize_img(img, self.config['IMAGE_SHAPE'])  # reshape the image
            dst_img, mat = enhance(src_img, self.config)

            # cv2 -> PIL
            src_img = Image.fromarray(src_img)  # rgb 顺序
            dst_img = Image.fromarray(dst_img)  # rgb 顺序

            # *****************************//原始图，高斯模糊，随机擦除，随机颜色，随机灰度化//*******************
            seed = random.random()
            if seed < 0.25:
                src_img = src_img.filter(PilGaussianBlur(radius=random.randint(1, 2)))

            # src_img = tfs.ToTensor()(src_img)
            # src_img = tfs.RandomErasing(p=0.2, scale=(0.01, 0.1), ratio=(0.1, 10.0))(src_img)
            # src_img = tfs.ToPILImage(mode='RGB')(src_img)

            src_img = tfs.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3, hue=0.1)(src_img)
            src_img = tfs.RandomGrayscale(p=0.1)(src_img)

            # *****************************//变换图，高斯模糊，随机擦除，随机颜色，随机灰度化//*******************
            seed = random.random()
            if seed < 0.25:
                dst_img = dst_img.filter(PilGaussianBlur(radius=random.randint(1, 2)))

            # dst_img = tfs.ToTensor()(dst_img)
            # dst_img = tfs.RandomErasing(p=0.2, scale=(0.01, 0.1), ratio=(0.1, 10.0))(dst_img)
            # dst_img = tfs.ToPILImage(mode='RGB')(dst_img)

            dst_img = tfs.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3, hue=0.1)(dst_img)
            dst_img = tfs.RandomGrayscale(p=0.5)(dst_img)

            # PIL -> cv2
            src_img = np.asarray(src_img)
            dst_img = np.asarray(dst_img)

            return src_img, dst_img, mat, img_file
        else:
            src_img = cv2.resize(img, (self.config['IMAGE_SHAPE'][1], self.config['IMAGE_SHAPE'][0]))
            return src_img, img_file
        

    def collate_batch(*batches):
        src_img = []
        dst_img = []
        mat = []
        for batch in batches[1]:
            src_img.append(batch[0])
            dst_img.append(batch[1])
            mat.append(batch[2])
        src_img = torch.tensor(src_img, dtype=torch.float32)  # B * H * W * C
        dst_img = torch.tensor(dst_img, dtype=torch.float32)  # B * H * W * C
        mat = torch.tensor(mat, dtype=torch.float32, requires_grad=False).squeeze()  # B * 3 * 3
        return src_img.permute(0, 3, 1, 2), dst_img.permute(0, 3, 1, 2), mat
    
    def test_collate_batch(*batches):
        src_img = []
        img_idx = []
        for batch in batches[1]:
            src_img.append(batch[0])
            img_idx.append(batch[1])
        src_img_tensor = torch.tensor(src_img, dtype=torch.float32)  # B * H * W * C

        return src_img, src_img_tensor.permute(0, 3, 1, 2), img_idx

