import os
import sys
from collections import OrderedDict
import cv2
import math
import time
import torch
import numpy as np
import torch.nn as nn
from Loss import _gather_feat
from dataset import get_affine_transform
from Loss import _transpose_and_gather_feat
from PIL import Image
from torchvision import transforms
from backbone.dlanet_dcn import DlaNet
def mk_file(file_path: str):
    if os.path.exists(file_path):
        return
    os.makedirs(file_path)
def draw(res,image,filename,filepath2):
    # 读取图像
    if not res:
        print("cropping and saving fail!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!: " + filename.split('.')[0] + '.png')
        image.save(os.path.join(path_name_all+'/fail_images', filename.split('.')[0] + '.png'))
    for class_name,lx,ly,rx,ry,ang, prob in res:
        result = [int((rx+lx)/2),int((ry+ly)/2),int(rx-lx),int(ry-ly),ang]
        result=np.array(result)
        x=int(result[0])
        y=int(result[1])
        height=int(result[2])
        width=int(result[3])
        angle = result[4]

        # 指定新的尺寸
        lenge = max(image.width, image.height)
        new_width = lenge * 3
        new_height = lenge * 3

        # 计算左上角的坐标，以在新尺寸内居中显示图像
        left = (new_width - image.width) // 2
        top = (new_height - image.height) // 2

        # 创建新的画布，填充为黑色
        padded_image = Image.new('L', (new_width, new_height), color='black')
        image = image.convert('L')
        # 将原始图像粘贴到新的画布中
        padded_image.paste(image, (left, top))
        center_x = x + (new_width - image.width) // 2
        center_y = y + (new_height - image.height) // 2
        # 将角度转换为弧度
        angle_rad = math.radians(angle)

        # 计算旋转框的四个角点坐标
        top_left = (center_x - width / 2, center_y - height / 2)
        top_right = (center_x + width / 2, center_y - height / 2)
        bottom_left = (center_x - width / 2, center_y + height / 2)
        bottom_right = (center_x + width / 2, center_y + height / 2)

        # 得到新的旋转框的四个顶点坐标
        new_rect_points = [top_left, top_right, bottom_right, bottom_left]

        rotated_image = padded_image.rotate(-angle + 90, center=(center_x, center_y), resample=Image.BICUBIC, expand=False)

        # 得到新的旋转框的坐标范围
        min_x = min(point[0] for point in new_rect_points)
        max_x = max(point[0] for point in new_rect_points)
        min_y = min(point[1] for point in new_rect_points)
        max_y = max(point[1] for point in new_rect_points)

        # 裁剪旋转后的图像
        cropped_image = rotated_image.crop((min_x, min_y, max_x, max_y))

        # 保存结果 all_res34_img_output_all   Tongji
        images_save_path = os.path.join(path_name_all, filepath2)
        cropped_image.save(os.path.join(images_save_path, filename.split('.')[0] + '.png'))
        print("cropping and saving successfully: " + filename.split('.')[0] + '.png')
        break
# Non-Maximum Suppression keep max-value within margin in 3*3
def _nms(heat, kernel=3):
    # keep size
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep
# 取出前K个最大相应
def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
    # (batch, cat, K)
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)
    # 确保有效范围
    topk_inds = topk_inds % (height * width)
    # 注意 图像坐标 PIL not numpy 坐标
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()
    # (batch, K) 前c个最有可能的
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    # 最可能class
    topk_clses = (topk_ind / K).int()
    # 前40个 的图像index value:0-wh-1
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    # 前40个的 y
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)
    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs
def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]
def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords
def ctdet_post_process(dets, c, s, h, w, num_classes):
  # dets: batch x max_dets x dim
  # return 1-based class det dict
  ret = []
  for i in range(dets.shape[0]):
    top_preds = {}
    dets[i, :, :2] = transform_preds(dets[i, :, 0:2], c[i], s[i], (w, h))
    dets[i, :, 2:4] = transform_preds(dets[i, :, 2:4], c[i], s[i], (w, h))
    classes = dets[i, :, -1]
    for j in range(num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :4].astype(np.float32),
        dets[i, inds, 4:6].astype(np.float32)], axis=1).tolist()
    ret.append(top_preds)
  return ret

# 读取numpy图像
def read_numpy_image(path):
    original_img = Image.open(path)
    # Pillow 中的 split() 用于分离图像的通道，比如将 RGB 分成 R、G、B 三个单通道。
    if len(original_img.split()) < 3:
        img1 = cv2.imread(path)
        gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = np.zeros_like(img1)
        img2[:, :, 0] = gray
        img2[:, :, 1] = gray
        img2[:, :, 2] = gray
        img = img2
    elif len(original_img.split()) > 3:
        img1 = cv2.imread(path)
        img2 = img1.convert('RGB')
        img = img2
    else:
        img = cv2.imread(path)
    return img
#原始图像缩放到512 512 并转为（1，3，512，512）
def pre_process(image):
    # 高在前
    height, width = image.shape[0:2]
    inp_height, inp_width = 512, 512
    c = np.array([width / 2., height / 2.], dtype=np.float32)
    s = max(height, width) * 1.0
    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    inp_image = cv2.warpAffine(image, trans_input, (inp_width, inp_height), flags=cv2.INTER_LINEAR)

    mean = np.array([0.5194416012442385, 0.5378052387430711, 0.533462090585746], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.3001546018824507, 0.28620901391179554, 0.3014112676161966], dtype=np.float32).reshape(1, 1, 3)

    inp_image = ((inp_image / 255. - mean) / std).astype(np.float32)

    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)  # 三维reshape到4维，（1，3，512，512）

    images = torch.from_numpy(images)
    meta = {'c': c, 's': s,
            'out_height': inp_height // 4,
            'out_width': inp_width // 4}
    return images, meta
# 模型预测
def process(images):
    with torch.no_grad():
        output = model(images)
        hm = output['hm'].sigmoid_()
        ang = output['ang'].relu_()
        wh = output['wh']
        reg = output['reg']
        # CUDA 操作同步 GPU 上的所有计算任务都已完成
        # torch.cuda.synchronize()
        # forward_time = time.time()
        dets = ctdet_decode(hm, wh, ang, reg=reg, K=100)
        return output, dets
# 目标检测的解码函数 解码得到前100个最优解
def ctdet_decode(heat, wh, ang, reg=None, K=100):
    batch, cat, height, width = heat.size()
    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps 非最大值抑制
    # 经过3x3的max_pooling之后，确实消除了一些低响应区域，但由于3x3的核太小，只进行一次池化操作，无法消除所有底响应区域。
    # 这样的结果是不可用的
    # 所以最方便的方法还是用nms进行后处理
    heat = _nms(heat)
    # 取出前K个最大响应
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    # get Top-K reg
    reg = _transpose_and_gather_feat(reg, inds)
    reg = reg.view(batch, K, 2)
    # 中心坐标+偏移量
    xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
    ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    wh = _transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, 2)
    ang = _transpose_and_gather_feat(ang, inds)
    ang = ang.view(batch, K, 1)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    # 获得最终的bboxes
    bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2,
                        ang], dim=2)
    # (batch_size, K, 7)
    detections = torch.cat([bboxes, scores, clses], dim=2)
    return detections
#后处理，从512 512下坐标 变换回原始图像 下坐标
def post_process(dets, meta):
    # bbox meta
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    num_classes = 1 #目标只有1个
    # 后处理 仿射回原始图像的坐标 512->原始坐标
    dets = ctdet_post_process(dets.copy(), [meta['c']], [meta['s']], meta['out_height'], meta['out_width'],num_classes)
    for j in range(1, num_classes + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 6)
        dets[0][j][:, :5] /= 1#看起来多余
    return dets[0]
'''
对候选框进行过滤，保证每张图片的检测目标数量不超过 max_obj_per_img（100个目标）。
提取所有检测框的分数，找出前 100 个目标的最低分作为阈值，筛选掉得分较低的候选框。
'''
def merge_outputs(detections):
    num_classes = 1
    max_obj_per_img = 100
    scores = np.hstack([detections[j][:, 5] for j in range(1, num_classes + 1)])
    if len(scores) > max_obj_per_img:
      kth = len(scores) - max_obj_per_img
      thresh = np.partition(scores, kth)[kth]
      for j in range(1, 2 + 1):
        keep_inds = (detections[j][:, 5] >= thresh)
        detections[j] = detections[j][keep_inds]
    return detections
if __name__ == '__main__':
    device = torch.device("cuda")
    gpus = [0]
    model = DlaNet(34)
    model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])
    checkpoint_path = './best_roi_model.pth'
    if os.path.exists(checkpoint_path):
        print("model '{}' exists.".format(checkpoint_path))
        # 注意，以相同的顺序加载保证环境一致
        checkpoint = torch.load(checkpoint_path)
        loaded_keys = set(model.state_dict().keys())
        checkpoint_keys = set(checkpoint.keys())
        if loaded_keys == checkpoint_keys:
            print("Model loaded successfully.")
            model.load_state_dict(checkpoint)
        else:
            print("Warning: Model keys do not match exactly.")
    else:
        print("model '{}' does not exist. Exit.".format(checkpoint_path))
        sys.exit(1)
    # 结果集路径
    pth_name='best_roi_model_'
    # 成功结果路径
    path_name_all = pth_name + 'img_output'
    # 失败结果路径
    mk_file(path_name_all+'/fail_images/')
    # img_path/images_path_all 构成测试集路径
    img_path = './test_data_base/'
    images_path_all = [
        "MPD"
    ]
    for path in images_path_all:
        # best_roi_model_img_output/Tongji 成功输出路径
        path=os.path.join(path,'')
        images_save_path = os.path.join(path_name_all, path)
        mk_file(images_save_path)
        for original_img in os.listdir(img_path+path):
            # ./test_data_base/Tongji
            img=read_numpy_image(os.path.join(img_path,path,original_img))
            images, meta = pre_process(img) # -> 512 512
            images = images.to(device)
            output, dets = process(images)
            # 后处理
            dets = post_process(dets, meta)
            ret = merge_outputs(dets)

            # 用于存放检测结果。每一行存放一个检测框的信息，包含了 7 个数值
            res = np.empty([1,7])
            # print(ret) # dict   {1:(100,6)}掌纹（1）的类别中前100个候选框的数据
            # for i, c in ret.items():
            #     print(str(i)+' '+str(c.shape))

            for i, c in ret.items():
                # 提取了类别 i 中所有检测框的置信度 也就是heatmap的值
                tmp_s = ret[i][ret[i][:, 5] > 0.3]
                if len(tmp_s) == 0:
                    # 如果没有置信度大于0.3的结果，则提取最大置信度的结果
                    max_conf_idx = np.argmax(ret[i][:, 5])  # 获取最大置信度的索引
                    max_conf_detection = ret[i][max_conf_idx:max_conf_idx + 1]  # 提取最大置信度的检测结果
                    tmp_s = max_conf_detection
                # print(tmp_s) #[[247.91637   224.02309   520.4284    481.27222    74.17106     0.6661045]]
                tmp_c = np.ones(len(tmp_s))
                # print(tmp_c) # [2]
                # print(tmp_c.shape)
                # tmp第一列是类别标签，后面的列是检测框坐标和置信度。
                tmp = np.c_[tmp_c, tmp_s]
                res = np.append(res, tmp, axis=0)
            res = np.delete(res, 0, 0)
            # print(res.size) #7
            # print(res) #[[  1. 200.00852966 170.22557068 480.51239014 434.8507385 ,76.31229401   0.53650278]]
            res = res.tolist()
            # print(images.shape) torch.Size([1, 3, 512, 512])
            # images = img.squeeze(0).permute(1, 2, 0).cpu().numpy()  # 去掉多余的维度
            # 如果你的数据是浮点数，你需要将它转换为 uint8
            # 假设数据在 [0, 1] 范围内，将其缩放到 [0, 255]
            # images = (images * 255).astype(np.uint8)
            # numpy() 方法将 Tensor 转换为 NumPy 数组，但在此之前需要确保 Tensor 在 CPU 上
            images = Image.fromarray(img)
            # ./test_data_base/Tongji
            draw(res,images,original_img,path)