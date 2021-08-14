# import sys
# sys.path.append('./')
import torch
import torch.nn as nn
import numpy as np

from symbols.model_factory import ResNet, UnsuperVggTiny
from symbols.model_base import ModelTemplate

class UnSuperPoint(ModelTemplate):
    def __init__(self, base_model, model_config, IMAGE_SHAPE, training=True):
        super(UnSuperPoint, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.config = model_config
        self.downsample = model_config['downsample']
        # self.L2_norm = model_config['L2_norm']
        self.image_shape = IMAGE_SHAPE
        self.feature_hw = [self.image_shape[0] // self.downsample, self.image_shape[1] // self.downsample]

        # export threshold
        # self.score_th = model_config['score_th']
        self.correspond = model_config['correspond']
        self.position_weight = model_config['position_weight']
        self.score_weight = model_config['score_weight']
        self.rep_weight = model_config['rep_weight']

        # LOSS
        self.usp = model_config['LOSS']['usp']
        self.uni_xy = model_config['LOSS']['uni_xy']
        self.desc = model_config['LOSS']['desc']
        self.decorr = model_config['LOSS']['decorr']

        self.d = model_config['d']
        self.m_p = model_config['m_p']
        self.m_n = model_config['m_n']
        # self.dis_th = model_config['dis_th']

        self.eps = 1e-12

        # create mesh grid
        x = torch.arange(self.image_shape[1] // self.downsample, requires_grad=False, device='cuda'if torch.cuda.is_available() else 'cpu')
        y = torch.arange(self.image_shape[0] // self.downsample, requires_grad=False, device='cuda'if torch.cuda.is_available() else 'cpu')
        y, x = torch.meshgrid([y, x])
        self.cell = torch.stack([x, y], dim=0)

        self.base_model = base_model
        self.input_ch = 128
        self.des_ch = 128

        self.score = nn.Sequential(
            nn.Conv2d(self.input_ch, self.input_ch, 3, 1, padding=1),
            nn.BatchNorm2d(self.input_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.input_ch, 1, 1, 1, padding=0),
            nn.Sigmoid()
        )
        self.position = nn.Sequential(
            nn.Conv2d(self.input_ch, self.input_ch, 3, 1, padding=1),
            nn.BatchNorm2d(self.input_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.input_ch, 2, 1, 1, padding=0),
            nn.Sigmoid()
        )
        self.descriptor = nn.Sequential(
            nn.Conv2d(self.input_ch, self.input_ch*2, 3, 1, padding=1),
            nn.BatchNorm2d(self.input_ch*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.input_ch*2, self.input_ch*2, 3, 1, padding=1),
            nn.BatchNorm2d(self.input_ch*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.input_ch*2, self.des_ch, 1, 1, padding=0),
            nn.BatchNorm2d(self.des_ch)
        )

    def forward(self, x):
        x = x /256. - 0.5  # /127.5
        feature = self.base_model(x)

        s = self.score(feature)
        p = self.position(feature)
        d = self.descriptor(feature)
        # desc = self.interpolate(p, d, self.feature_hw[0], self.feature_hw[1])  # (B, C, H, W)
        # d = torch.nn.functional.normalize(input=d, p=2, dim=1, eps=self.eps)
        return s, p, d

    def loss(self, batch_as, batch_ap, batch_ad, batch_bs, batch_bp, batch_bd, batch_mat):
        loss = 0
        batch = batch_as.shape[0]
        loss_batch_array = np.zeros((7,))

        for i in range(batch):
            loss_batch, loss_item = self.UnsuperPointLoss(batch_as[i], batch_ap[i], batch_ad[i],
                                                          batch_bs[i], batch_bp[i], batch_bd[i], batch_mat[i])
            loss += loss_batch
            loss_batch_array += loss_item

        return loss / batch, loss_batch_array / batch

    # def interpolate(self, p, d, h, w):
    #     # b, c, h, w
    #     # h, w = p.shape[2:]
    #     samp_pts = self.get_batch_position(p)
    #     samp_pts[:, 0, :, :] = (samp_pts[:, 0, :, :] / (float(w)/2.)) - 1.
    #     samp_pts[:, 1, :, :] = (samp_pts[:, 1, :, :] / (float(h)/2.)) - 1.
    #     samp_pts = samp_pts.permute(0, 2, 3, 1)
    #     desc = torch.nn.functional.grid_sample(d, samp_pts, padding_mode='zeros', align_corners=False)  # 这里的pad模式可能要调整 border， mx.nd.GridGenerator()
    #     return desc

    def UnsuperPointLoss(self, a_s, a_p, a_d, b_s, b_p, b_d, mat):
        position_a = self.get_position(a_p, self.cell, self.downsample, flag='A', mat=mat)  # c h w, where c==2
        position_b = self.get_position(b_p, self.cell, self.downsample, flag='B', mat=None)

        key_dist = self.get_dis(position_a, position_b)  # c h w -> p p

        batch_loss = 0
        loss_item = []

        if self.usp > 0:
            position_k_loss, score_k_loss, usp_k_loss = self.usp_loss(a_s, b_s, key_dist)
            usp_loss = self.usp * (position_k_loss + score_k_loss + usp_k_loss)
            batch_loss += usp_loss
            loss_item.extend([usp_loss.item(), position_k_loss.item(), score_k_loss.item(), usp_k_loss.item()])
        else:
            loss_item.append(0.)

        if self.uni_xy > 0:
            uni_xy_loss = self.uni_xy * self.uni_xy_loss(a_p, b_p)
            batch_loss += uni_xy_loss
            loss_item.append(uni_xy_loss.item())
        else:
            loss_item.append(0.)

        if self.desc > 0:
            desc_loss = self.desc * self.desc_loss(a_d, b_d, key_dist)
            batch_loss += desc_loss
            loss_item.append(desc_loss.item())
        else:
            loss_item.append(0.)

        if self.decorr > 0:
            decorr_loss = self.decorr * self.decorr_loss(a_d, b_d)
            batch_loss += decorr_loss
            loss_item.append(decorr_loss.item())
        else:
            loss_item.append(0.)

        return batch_loss, np.array(loss_item)

    def get_position(self, p_map, cell, downsample, flag=None, mat=None):
        res = (cell + p_map) * downsample

        if flag == 'A':
            # https://www.geek-share.com/detail/2778133699.html  提供了src->dst的计算模式
            r = torch.zeros_like(res)
            denominator = res[0, :, :] * mat[2, 0] + res[1, :, :] * mat[2, 1] + mat[2, 2]
            r[0, :, :] = (res[0, :, :] * mat[0, 0] + res[1, :, :] * mat[0, 1] + mat[0, 2]) / denominator
            r[1, :, :] = (res[0, :, :] * mat[1, 0] + res[1, :, :] * mat[1, 1] + mat[1, 2]) / denominator
            return r
        else:
            return res

    def get_dis(self, p_a, p_b):
        c = p_a.shape[0] # 2
        reshape_pa = p_a.reshape((c, -1)).permute(1, 0)  # c h w -> c p -> p c
        reshape_pb = p_b.reshape((c, -1)).permute(1, 0)

        x = torch.unsqueeze(reshape_pa[:, 0], 1) - torch.unsqueeze(reshape_pb[:, 0], 0)  # p c -> p 1 - 1 p -> p p
        y = torch.unsqueeze(reshape_pa[:, 1], 1) - torch.unsqueeze(reshape_pb[:, 1], 0)
        dis = torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2) + self.eps) # p p
        return dis

    def usp_loss(self, a_s, b_s, dis):
        reshape_as_k, reshape_bs_k, d_k = self.get_point_pair(a_s, b_s, dis)  # p -> k

        position_k_loss = torch.sum(d_k)  # 最小化距离函数，监督offset

        score_k_loss = torch.sum(torch.pow(reshape_as_k - reshape_bs_k, 2))  # 监督分数一致性

        sk_ = (reshape_as_k + reshape_bs_k) / 2
        d_ = torch.mean(d_k)
        # 可重复性监督，分数高的地方->距离就小。分数低的地方->距离就大, 因为有正有负，这样就会让负数的score变大
        # 负数的存在是关键，是让score提高的关键变量
        usp_k_loss = torch.sum(sk_ * torch.detach(d_k - d_))

        # 按文章的思路，距离小的地方->分数高，距离大的地方->分数低
        # high = (d_k > d_) sk_[high] = sk_[high]
        # low = (d_k <= d_)
        # sk_[low] = 1. - sk_[low]
        # usp_k_loss = torch.mean(sk_)

        position_k_loss = position_k_loss * self.position_weight
        score_k_loss = score_k_loss * self.score_weight
        usp_k_loss = usp_k_loss * self.rep_weight

        # 在趋于平稳后，分布是 -0.08, 1.0, 0.03
        # print(usp_k_loss, position_k_loss, score_k_loss)

        total_usp = position_k_loss + score_k_loss + usp_k_loss
        # return total_usp
        return position_k_loss, score_k_loss, usp_k_loss

    def get_point_pair(self, a_s, b_s, dis):
        a2b_min_id = torch.argmin(dis, dim=1)
        len_p = len(a2b_min_id)
        ch = dis[list(range(len_p)), a2b_min_id] < self.correspond
        reshape_as = a_s.reshape(-1)
        reshape_bs = b_s.reshape(-1)

        a_s = reshape_as[ch]
        b_s = reshape_bs[a2b_min_id[ch]]
        d_k = dis[ch, a2b_min_id[ch]]

        return a_s, b_s, d_k

    def uni_xy_loss(self, a_p, b_p):
        c = a_p.shape[0]
        reshape_pa = a_p.reshape((c, -1)).permute(1, 0)  # c h w -> c p -> p c where c=2
        reshape_pb = b_p.reshape((c, -1)).permute(1, 0)

        loss = (self.get_uni_xy(reshape_pa[:, 0]) + self.get_uni_xy(reshape_pa[:, 1]))
        loss += (self.get_uni_xy(reshape_pb[:, 0]) + self.get_uni_xy(reshape_pb[:, 1]))

        return loss

    def get_uni_xy(self, position):
        # i = torch.argsort(position) + 1  # 返回的索引是0开始的
        # i = i.float()
        # p = len(position)
        # uni_l1 = torch.mean(torch.pow(position - (i - 1) / (p - 1), 2))

        idx = torch.argsort(position).detach()  # 返回的索引是0开始的 上面的方式loss会略大0.000x级别
        p = position.shape[0]
        idx_f = torch.arange(p).float().to(self.device).detach()
        uni_l2 = torch.sum(torch.pow(position[idx] - (idx_f / p), 2))


        return uni_l2

    def desc_loss(self, d_a, d_b, dis):
        c = d_a.shape[0] # num of feature 128 or 256
        reshape_da = d_a.reshape((c, -1)).permute(1, 0)  # c h w -> c p -> p c
        reshape_db = d_b.reshape((c, -1))  # c h w -> c p
        pos = (dis.detach() <= 5)
        neg = (dis.detach() > 5)
        ab = torch.mm(reshape_da, reshape_db)  # p c * c p -> p p

        # 监督图a和图b的相同位置生成相似的描述子
        # margin loss
        # pos = min(ab[pos]) neg = max(ab[neg])
        # loss = max(0, m + (neg - pos))
        margin_loss = (self.m_p - self.m_n) + torch.max(ab[neg]) - torch.min(ab[pos])
        margin_loss = torch.clamp(margin_loss, min=0.0)
        # print(torch.max(ab[neg]), torch.min(ab[pos]))
        # print(torch.max(ab[pos]), torch.min(ab[neg]))
        # print(torch.max(ab), torch.min(ab))

        ab[pos] = self.d * (self.m_p - ab[pos])
        ab[neg] = ab[neg] - self.m_n
        ab = torch.clamp(ab, min=0.0)
        # print(torch.mean(ab[pos]), torch.mean(ab[neg]))
        # print(torch.sum(ab[pos]), torch.sum(ab[neg]))
        # print(torch.mean(ab), loss)
        # print('dddddddddd')
        loss = torch.sum(ab)
        return loss

    def decorr_loss(self, d_a, d_b):
        c, h, w = d_a.shape
        reshape_da = d_a.reshape((c, -1))  # .permute(1, 0)  # c h w -> c p
        reshape_db = d_b.reshape((c, -1))  # .permute(1, 0)
        loss = self.get_r_b(reshape_da)
        loss += self.get_r_b(reshape_db)
        return loss

    def get_r_b(self, reshape_d):
        # F = reshape_D.shape[0]
        # v_ = torch.mean(reshape_D, dim=1, keepdim=True)
        # V_v = reshape_D - v_
        # molecular = torch.matmul(V_v, V_v.transpose(1, 0))
        # V_v_2 = torch.sum(torch.pow(V_v, 2), dim=1, keepdim=True)
        # denominator = torch.sqrt(torch.matmul(V_v_2, V_v_2.transpose(1, 0)))
        # one = torch.eye(F).cuda()
        # # l1 = torch.sum(molecular / denominator - one) / (F * (F-1))
        # l1 = (molecular / denominator) * (1 - one)
        # # return torch.sum(molecular / denominator - one) / (F * (F-1))

        # F = reshape_D.shape[0]
        # Ys_mean = torch.mean(reshape_D, dim=1, keepdim=True)
        # Ys_sd = torch.sqrt(torch.var(reshape_D, dim=1, keepdim=True))
        #
        # V = (reshape_D - Ys_mean) / Ys_sd
        #
        # Rs = torch.matmul(V, V.transpose(1, 0))
        # lo = (1 - torch.eye(F).cuda()) * Rs
        # return torch.mean(lo)

        f, p = reshape_d.shape

        # 监督不同bit位的相关性
        # x_mean = torch.mean(reshape_d, dim=1, keepdim=True)  # c p -> c 1
        # x_var = torch.mean((reshape_d - x_mean) ** 2, dim=1, keepdim=True)
        # x_hat = (reshape_d - x_mean) / torch.sqrt(x_var + eps)
        # rs = torch.mm(x_hat, x_hat.transpose(1, 0)) / p  # c p * p c -> c c
        # # ys = (1 - torch.eye(f).cuda()) * rs
        # ys = rs - torch.eye(f, device=reshape_d.device)
        # loss = torch.mean(torch.pow(ys, 2))

        # 监督不同位置描数子整体相关性, -1~1 -> 0~2,
        rs = torch.mm(reshape_d.transpose(1, 0), reshape_d)
        ys = rs - torch.eye(p, device=reshape_d.device)
        loss = torch.sum(ys)

        return loss

    def predict(self, img):
        s1, p1, d1 = self.forward(img)

        # 统计offset的偏移范围
        # array = np.zeros(10)
        # for i in range(10):
        #     pa_re = p1[:, 0, :, :].reshape(1200)
        #     da1 = 0.1 * i <= pa_re
        #     da2 = pa_re <= 0.1 * (1 + i)
        #     da = da1 * da2
        #     array[i] = torch.sum(da)
        # al = np.sum(array)
        # print('offset x', array)
        # for i in range(10):
        #     pa_re = p1[:, 1, :, :].reshape(1200)
        #     da1 = 0.1 * i <= pa_re
        #     da2 = pa_re <= 0.1 * (1 + i)
        #     da = da1 * da2
        #     array[i] = torch.sum(da)
        # al = np.sum(array)
        # print('offset y', array)

        batch_size = s1.shape[0]
        # position1 = self.get_batch_position(p1)
        position1 = self.get_position(p1, self.cell, self.downsample)
        position1 = position1.reshape((batch_size, 2, -1)).permute(0, 2, 1)  # B * (HW) * 2
        s1 = s1.reshape((batch_size, -1))
        c = d1.shape[1]
        d1 = d1.reshape((batch_size, c, -1)).permute(0, 2, 1)  # B * (HW) * c

        output_dict = {}
        for i in range(batch_size):
            s1_ = s1[i, ...].cpu().numpy()
            p1_ = position1[i, ...].cpu().numpy()
            d1_ = d1[i, ...].cpu().numpy()
            output_dict[i] = {'s1': s1_, 'p1': p1_, 'd1': d1_}
        return output_dict

def get_sym(model_config, image_shape, is_training):
    # base_model = UnsuperVggTiny()# UnsuperShortcut()
    base_model = ResNet()
    model = UnSuperPoint(base_model=base_model, model_config=model_config, IMAGE_SHAPE=image_shape, training=is_training)
    return model

if __name__ == '__main__':
    import yaml

    cfg = None
    with open('../Unsuper/configs/UnsuperPoint_coco.yaml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    f.close()

    model = get_sym(model_config=cfg['MODEL'], image_shape=cfg['data']['IMAGE_SHAPE'], is_training=True)
    model.eval()

    from thop import profile
    # 增加可读性
    from thop import clever_format

    input = torch.randn((1, 3, 240, 320))

    # # to onnx
    input_names = ['input']
    output_names = ['score', 'position', 'descriptor']
    torch.onnx.export(model, input, '../output/usp_test.onnx', input_names=input_names, output_names=output_names,
                      verbose=True)  # , keep_initializers_as_inputs=True, opset_version=11
    # flops
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print('flops:', flops, 'params:', params)

    print('end process!!!')
