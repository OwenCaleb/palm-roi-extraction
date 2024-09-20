
import os
import cv2
import math
import random
import numpy as np
import torch.utils.data as data
import pycocotools.coco as coco
import re
from PIL import Image
class ctDataset(data.Dataset):
    num_classes = 1
    default_resolution = [512, 512]
    mean = np.array([0.5194416012442385, 0.5378052387430711, 0.533462090585746], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.3001546018824507, 0.28620901391179554, 0.3014112676161966], dtype=np.float32).reshape(1, 1, 3)

    def __init__(self,data_name='Tongji', data_dir='data', split='palmprint'):
        self.data_dir = os.path.join('./'+data_dir, 'palmprint')
        self.img_dir = os.path.join(self.data_dir, 'images')
        self.trainjsonname = '{}_train.json'.format(data_name)
        self.valjsonname='{}_val.json'.format(data_name)
        try:
            if split == 'palmprint':
                self.annot_path = os.path.join(self.data_dir, 'annotations', self.trainjsonname)
            elif split == 'val':
                self.annot_path = os.path.join(self.data_dir, 'annotations', self.valjsonname)
        except:
            print('No any data!')

        self.max_objs = 128
        self.class_name = ['obj']
        self._valid_ids = [1] # 只有一种检测目标：palmprint
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)} # 1:0
        self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) for v in range(1, self.num_classes + 1)]
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]], dtype=np.float32)

        self.split = split
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        dataset=''
        MPD_pattern = r'^(\d{3})_(\d)_(\w)_(\w)_(\d{2})\.jpg$$'
        MPD_match = re.match(MPD_pattern, file_name)
        if MPD_match:
            dataset ="PalmSet/MPD"
            img_path = os.path.join(self.img_dir, 'MPD','PalmSet', file_name)
        elif file_name.startswith("Tongji_"):
            dataset= "Tongji"
            img_path = os.path.join(self.img_dir, dataset, file_name)
        elif file_name.startswith("casia_"):
            dataset = "CASIA"
            img_path = os.path.join(self.img_dir, dataset, file_name)
        else:
            dataset=dataset
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs) #1
        img = cv2.imread(img_path)# 顺序 高宽通道 假设 600 * 400
        # height, width, channels = img.shape MPD 数据集为例
        # print(height) 4160
        # print(width)  3120
        # print(channels)  3
        # image = Image.open(img_path)
        # width, height = image.size 3120 4160
        # print(width)
        # print(height)
        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)  # 中心点 200 300
        keep_res = False  #
        if keep_res:
            # 不是32的倍数
            input_h = (height | 31) + 1
            input_w = (width | 31) + 1
            # size 416 618
            s = np.array([input_w, input_h], dtype=np.float32)
            print(s.shape)
        else:
            # 取最大的
            s = max(img.shape[0], img.shape[1]) * 1.0
            input_h, input_w = 512, 512

        # 没有用到rot
        trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)#线性插值方法
        inp = (inp.astype(np.float32) / 255.)

        # 归一化
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        down_ratio = 4
        output_h = input_h // down_ratio
        output_w = input_w // down_ratio
        num_classes = self.num_classes
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])
        # heatmap 1 128 128 h w
        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        # where  128 2
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        # 128 1
        ang = np.zeros((self.max_objs, 1), dtype=np.float32)
        # 用于存储回归值（目标位置的偏移量） 128 2
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        # 用于存储目标中心的索引 128
        ind = np.zeros((self.max_objs), dtype=np.int64)
        # 用于指示哪些目标有回归值 128
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        draw_gaussian = draw_umich_gaussian
        for k in range(num_objs):  # num_objs图中标记物数目
            ann = anns[k]  # 第几个标记物的标签
            # 将COCO格式的bbox转换为[x1, y1, x2, y2]格式
            bbox, an = coco_box_to_bbox(ann['bbox'])
            cls_id = int(self.cat_ids[ann['category_id']])
            # print(cls_id) 0
            bbox[:2] = affine_transform(bbox[:2], trans_output)  # 将box坐标转换到 128*128内的坐标
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
            # 上面几行都是做数据扩充和resize之后的变换，不重要
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                # 计算目标中心点坐标 在这里精度也产生了损失，因此要offeset回归
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32) #[width/2,height/2]
                # 在heatmap上绘制高斯分布 hm[0](h,w) c r
                draw_gaussian(hm[cls_id], ct_int, radius)
                # 记录目标的宽、高
                wh[k] = 1. * w, 1. * h
                # 记录目标的角度
                ang[k] = 1. * an
                # 计算heatmap中目标中心点的位置索引
                ind[k] = ct_int[1] * output_w + ct_int[0]
                # 计算目标中心点与整数坐标点的偏差
                reg[k] = ct - ct_int
                # 更新reg_mask以指示目标的存在
                reg_mask[k] = 1
        # 到此为止，包装完成。后面就是让神经网络训练出这几个参数，接近GroundTruth
        # inp RGB 512  hm 1 128 128
        ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'ang': ang}
        reg_offset_flag = True  # 是否添加 'reg' 键
        if reg_offset_flag:
            ret.update({'reg': reg})
        return ret


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def lighting_(data_rng, image, alphastd, eigval, eigvec):
    alpha = data_rng.normal(scale=alphastd, size=(3,))
    image += np.dot(eigvec, eigval * alpha)


def blend_(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2


def saturation_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs[:, :, None])


def brightness_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    image *= alpha


def contrast_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs_mean)


# 颜色扩充
def color_aug(data_rng, image, eig_val, eig_vec):
    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)
    gs = grayscale(image)
    gs_mean = gs.mean()
    for f in functions:
        f(data_rng, image, gs, gs_mean, 0.4)
    lighting_(data_rng, image, 0.1, eig_val, eig_vec)


def get_3rd_point(a, b): #获得直角三角形第三个点
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return src_result


def get_affine_transform(center, scale, rot, output_size,
                         shift=np.array([0, 0], dtype=np.float32), inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]  #原始图像的width 416 608
    dst_w = output_size[0] #、要是保持清晰度就裁剪，不进行缩放 scale就是原始 否则scale就是目标大小
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    # 这里在标签中应该处理了 角度问题
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    # 锚点，中心点不变
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def coco_box_to_bbox(box):
    bbox = np.array([box[0] - box[2] / 2, box[1] - box[3] / 2, box[0] + box[2] / 2, box[1] + box[3] / 2],
                    dtype=np.float32)
    ang = float(box[4])
    return bbox, ang


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2
    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def draw_dense_reg(regmap, heatmap, center, value, radius, is_offset=False):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    value = np.array(value, dtype=np.float32).reshape(-1, 1, 1)
    dim = value.shape[0]
    reg = np.ones((dim, diameter * 2 + 1, diameter * 2 + 1), dtype=np.float32) * value
    if is_offset and dim == 2:
        delta = np.arange(diameter * 2 + 1) - radius
        reg[0] = reg[0] - delta.reshape(1, -1)
        reg[1] = reg[1] - delta.reshape(-1, 1)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_regmap = regmap[:, y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom,
                      radius - left:radius + right]
    masked_reg = reg[:, radius - top:radius + bottom,
                 radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        idx = (masked_gaussian >= masked_heatmap).reshape(1, masked_gaussian.shape[0], masked_gaussian.shape[1])
        masked_regmap = (1 - idx) * masked_regmap + idx * masked_reg
    regmap[:, y - top:y + bottom, x - left:x + right] = masked_regmap
    return regmap


def get_border(border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i



