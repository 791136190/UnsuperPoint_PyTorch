import numpy as np
import cv2
import collections
import random
import torch

def resize_img(img, IMAGE_SHAPE):
    h, w = img.shape[:2]
    if h < IMAGE_SHAPE[0] or w < IMAGE_SHAPE[1]:
        new_h = IMAGE_SHAPE[0]
        new_w = IMAGE_SHAPE[1]
        h = new_h
        w = new_w
        img = cv2.resize(img, (new_w, new_h), np.random.randint(3))
    new_h, new_w = IMAGE_SHAPE
    try:
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)
    except:
        print(h, new_h, w, new_w)
        raise
    if len(img.shape) == 2:
        img = img[top: top + new_h, left: left + new_w]  # crop image
    else:
        img = img[top: top + new_h, left: left + new_w, :]
    return img

def dict_update(d, u):
    """Improved update for nested dictionaries.

    Arguments:
        d: The dictionary to be updated.
        u: The update dictionary.

    Returns:
        The updated dictionary.
    """
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def get_dst_point(perspective, IMAGE_SHAPE):
    a = random.random()
    b = random.random()
    c = random.random()
    d = random.random()
    e = random.random()
    f = random.random()

    if random.random() > 0.5:
        left_top_x = perspective*a
        left_top_y = perspective*b
        right_top_x = 0.9+perspective*c
        right_top_y = perspective*d
        left_bottom_x  = perspective*a
        left_bottom_y  = 0.9 + perspective*e
        right_bottom_x = 0.9 + perspective*c
        right_bottom_y = 0.9 + perspective*f
    else:
        left_top_x = perspective*a
        left_top_y = perspective*b
        right_top_x = 0.9+perspective*c
        right_top_y = perspective*d
        left_bottom_x  = perspective*e
        left_bottom_y  = 0.9 + perspective*b
        right_bottom_x = 0.9 + perspective*f
        right_bottom_y = 0.9 + perspective*d

    dst_point = np.array([(IMAGE_SHAPE[1]*left_top_x,IMAGE_SHAPE[0]*left_top_y,1),
            (IMAGE_SHAPE[1]*right_top_x, IMAGE_SHAPE[0]*right_top_y,1),
            (IMAGE_SHAPE[1]*left_bottom_x,IMAGE_SHAPE[0]*left_bottom_y,1),
            (IMAGE_SHAPE[1]*right_bottom_x,IMAGE_SHAPE[0]*right_bottom_y,1)],dtype = 'float32')
    return dst_point

def enhance(img, config):
    IMAGE_SHAPE = config['IMAGE_SHAPE']

    src_point = np.array([[               0,                0],
                          [IMAGE_SHAPE[1]-1,                0],
                          [               0, IMAGE_SHAPE[0]-1],
                          [IMAGE_SHAPE[1]-1, IMAGE_SHAPE[0]-1]], dtype=np.float32)  # 圖片的四個頂點

    dst_point = get_dst_point(config['homographic']['perspective'], IMAGE_SHAPE)  # 透视信息

    # rot = random.randint(-2, 2) * config['homographic']['rotation'] + random.randint(0, 15)  # 旋转
    rotation = config['homographic']['rotation']
    rot = random.randint(-rotation, rotation)  # [low, high] 和numpy的随机不一样，high是可以取的

    # scale = 1.2 - config['homographic']['scale'] * random.random()  # 缩放 1.2 - 0.2 * (0,1.0) -> (1.2,1.0)
    scale = 1.0 + config['homographic']['scale'] * random.randint(-10, 20) * 0.1  # 缩放 1.2 - 0.2 * (0,1.0) -> (1.2,1.0)

    center_offset = 40
    center = (IMAGE_SHAPE[1] / 2 + random.randint(-center_offset, center_offset),
              IMAGE_SHAPE[0] / 2 + random.randint(-center_offset, center_offset))

    RS_mat = cv2.getRotationMatrix2D(center, rot, scale)
    f_point = np.matmul(dst_point, RS_mat.T).astype('float32')
    mat = cv2.getPerspectiveTransform(src_point, f_point)
    out_img = cv2.warpPerspective(img, mat, (IMAGE_SHAPE[1], IMAGE_SHAPE[0]))

    return out_img, mat

# def get_position(p_map, cell, downsample=1, flag=None, mat=None):
#     """
#         calculate the position of key points
#         transform from image A to image B
#
#         Pmap : position map (2, H, W) (X_position, Y_position)
#         flag : denote whether it's map A or map B
#         mat : transformation matrix
#     """
#     # res = torch.zeros_like(Pmap).cuda()
#     res = (cell + p_map) * downsample
#
#     if flag == 'A':
#         # print(mat.shape)
#         r = torch.zeros_like(res)
#         # https://www.geek-share.com/detail/2778133699.html  提供了src->dst的计算模式
#         denominator = res[0, :, :] * mat[2, 0] + res[1, :, :] * mat[2, 1] + mat[2, 2]
#         r[0, :, :] = (res[0, :, :] * mat[0, 0] + res[1, :, :] * mat[0, 1] + mat[0, 2]) / denominator
#         r[1, :, :] = (res[0, :, :] * mat[1, 0] + res[1, :, :] * mat[1, 1] + mat[1, 2]) / denominator
#         return r
#     else:
#         return res

def key_map_pool(map, k=3):
    for i in range(0, map.shape[0] // k, k):
        for j in range(0, map.shape[1] // k, k):
            # print(i, j)
            cur_map = map[i * k : (i+1) * k, j * k : (j+1) * k]
            # print(cur_map)
            cur_max = np.max(cur_map)
            cur_index = np.where(cur_map == cur_max)
            map[i * k : (i+1) * k, j * k : (j+1) * k] = np.zeros_like(cur_map)
            map[i * k : (i+1) * k, j * k : (j+1) * k][cur_index] = cur_max
            # print(map[i * k : (i+1) * k, j * k : (j+1) * k])
    return map

def key_nms(map, dis_threshold=3):
    scores = map[:, 0]
    x = map[:, 1]
    y = map[:, 2]

    # 获得由大到小的分数索引
    score_index = np.argsort(scores)[::-1]

    keep = []

    while score_index.shape[0] > 0:
        max_index = score_index[0]
        # 最大的肯定是需要的框
        keep.append(max_index)
        # xx = (x[max_index] - x[score_index[1:]]) ** 2
        # yy = (y[max_index] - y[score_index[1:]]) ** 2
        xx = (x[max_index] - x[score_index[:]]) ** 2
        yy = (y[max_index] - y[score_index[:]]) ** 2

        dis = np.sqrt(xx + yy)

        ids = np.where(dis < dis_threshold)
        # 以为算iou的时候没把第一个参考框索引考虑进来，所以这里都要+1
        # score_index = score_index[ids + 1]
        score_index = np.delete(score_index, ids)
    return keep

if __name__=='__main__':
    import yaml

    cfg = None
    with open('../configs/UnsuperPoint_coco.yaml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    f.close()

    # test get position
    for i in range(10):
        img = cv2.imread('../../Data/COCO/images/COCO_train2014_000000291797.jpg')
        src_img = resize_img(img, cfg['data']['IMAGE_SHAPE'])
        dist_img, mat = enhance(src_img, cfg['data'])
        point = np.array([100, 100], dtype=np.float32)
        r = np.zeros((2, ))
        cv2.circle(src_img, center=(100, 100), radius=3, color=(255,0,0), thickness=-1)
        Denominator = point[0]*mat[2,0] + point[1]*mat[2,1] + mat[2,2]
        r[0] = (point[0]*mat[0,0] +
                point[0]*mat[0,1] +mat[0,2]) / Denominator
        r[1] = (point[1]*mat[1,0] +
                point[1]*mat[1,1] +mat[1,2]) / Denominator
        r = r.astype(np.int32)
        cv2.circle(dist_img, center=(r[0], r[1]), radius=3, color=(255,0,0), thickness=-1)
        cv2.imwrite('./img/%d_scr.jpg' % i, src_img)
        cv2.imwrite('./img/%d_dist.jpg' % i, dist_img)