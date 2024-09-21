# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 15:19:02 2020

@author: Lim
"""
import os
import cv2
import math
import time
import torch
import evaluation
import numpy as np
# from resnet_dcn import ResNet
from backbone.resnet import ResNet
from backbone.dlanet_dcn import DlaNet
import sys
# from dlanet_dcn import DlaNet
import matplotlib.pyplot as plt
from predict import pre_process, ctdet_decode, post_process, merge_outputs
import json

# =============================================================================
# 推断
# =============================================================================
def process(images, return_time=False):
    with torch.no_grad():
        output = model(images)
        hm = output['hm'].sigmoid_()
        ang = output['ang'].relu_()
        wh = output['wh']
        reg = output['reg']

        torch.cuda.synchronize()
        forward_time = time.time()
        dets = ctdet_decode(hm, wh, ang, reg=reg, K=100)  # K 是最多保留几个目标

    if return_time:
        return output, dets, forward_time
    else:
        return output, dets


# =============================================================================
# 常规 IOU
# =============================================================================
def iou(bbox1, bbox2, center=False):
    """Compute the iou of two boxes.
    Parameters
    ----------
    bbox1, bbox2: list.
        The bounding box coordinates: [xmin, ymin, xmax, ymax] or [xcenter, ycenter, w, h].
    center: str, default is 'False'.
        The format of coordinate.
        center=False: [xmin, ymin, xmax, ymax]
        center=True: [xcenter, ycenter, w, h]
    Returns
    -------
    iou: float.
        The iou of bbox1 and bbox2.
    """
    if center == False:
        xmin1, ymin1, xmax1, ymax1 = bbox1
        xmin2, ymin2, xmax2, ymax2 = bbox2
    else:
        xmin1, ymin1 = bbox1[0] - bbox1[2] / 2.0, bbox1[1] - bbox1[3] / 2.0
        xmax1, ymax1 = bbox1[0] + bbox1[2] / 2.0, bbox1[1] + bbox1[3] / 2.0
        xmin2, ymin2 = bbox2[0] - bbox2[2] / 2.0, bbox2[1] - bbox2[3] / 2.0
        xmax2, ymax2 = bbox2[0] + bbox2[2] / 2.0, bbox2[1] + bbox2[3] / 2.0

    # 获取矩形框交集对应的顶点坐标(intersection)
    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])

    # 计算两个矩形框面积
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    # 计算交集面积
    inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))
    # 计算交并比
    iou = inter_area / (area1 + area2 - inter_area + 1e-6)
    return iou


# bbox1 = [1,1,2,2]
# bbox2 = [2,2,2,2]
# ret = iou(bbox1,bbox2,True)


# =============================================================================
# 旋转 IOU
# =============================================================================
def iou_rotate_calculate(boxes1, boxes2):
    #    print("####boxes2:", boxes1.shape)
    #    print("####boxes2:", boxes2.shape)
    area1 = boxes1[2] * boxes1[3]
    area2 = boxes2[2] * boxes2[3]
    r1 = ((boxes1[0], boxes1[1]), (boxes1[2], boxes1[3]), boxes1[4])
    r2 = ((boxes2[0], boxes2[1]), (boxes2[2], boxes2[3]), boxes2[4])
    int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
    if int_pts is not None:
        order_pts = cv2.convexHull(int_pts, returnPoints=True)
        int_area = cv2.contourArea(order_pts)
        # 计算出iou
        ious = int_area * 1.0 / (area1 + area2 - int_area)
    #        print(int_area)
    else:
        ious = 0
    return ious


# 用中心点坐标、长宽、旋转角
# boxes1 = np.array([1,1,2,2,0],dtype='float32')
# boxes2 = np.array([2,2,2,2,0],dtype='float32')
# ret = iou_rotate_calculate(boxes1,boxes2)


# =============================================================================
# 获得标签信息
# =============================================================================
def get_lab_ret_from_xml(xml_path):
    ret = []
    # print(xml_path)
    with open(xml_path, 'r', encoding='UTF-8') as fp:
        ob = []
        flag = 0
        for p in fp:
            key = p.split('>')[0].split('<')[1]
            if key == 'cx':
                ob.append(p.split('>')[1].split('<')[0])
            if key == 'cy':
                ob.append(p.split('>')[1].split('<')[0])
            if key == 'w':
                ob.append(p.split('>')[1].split('<')[0])
            if key == 'h':
                ob.append(p.split('>')[1].split('<')[0])
            if key == 'angle':
                ob.append(p.split('>')[1].split('<')[0])
                flag = 1
            if flag == 1:
                x1 = float(ob[0])
                y1 = float(ob[1])
                w = float(ob[2])
                h = float(ob[3])
                angle = float(ob[4]) * 180 / math.pi
                angle = angle if angle < 180 else angle - 180
                bbox = [x1, y1, w, h, angle]  # COCO 对应格式[x,y,w,h]
                ret.append(bbox)
                ob = []
                flag = 0
    return ret


def get_lab_ret_from_json(json_path, filename):
    # 读取JSON文件
    with open(json_path, 'r', encoding='UTF-8') as fp:
        data = json.load(fp)

    ret = []
    image_id = None

    # 在images字段中找到对应的filename，并获取其image_id
    for image in data['images']:
        if image['file_name'] == filename:
            image_id = image['id']
            break

    # 如果未找到对应的image_id，返回空结果
    if image_id is None:
        return []  # 找不到对应的图片，返回空列表

    # 在annotations字段中根据image_id查找对应的bbox信息
    for annotation in data['annotations']:
        if annotation['image_id'] == image_id:
            bbox = annotation['bbox']

            # bbox已经是 [cx, cy, w, h, angle] 格式，直接转换
            cx = bbox[0]
            cy = bbox[1]
            w = bbox[2]
            h = bbox[3]
            angle = bbox[4]

            # 角度转换为度数，保持一致性
            angle_in_degrees = angle * 180 / math.pi
            angle_in_degrees = angle_in_degrees if angle_in_degrees < 180 else angle_in_degrees - 180

            # 构建输出格式为[x, y, w, h, angle]
            ret.append([cx, cy, w, h, angle_in_degrees])

    return ret

def get_pre_ret(img_path, device):
    image = cv2.imread(img_path)
    images, meta = pre_process(image)
    images = images.to(device)
    output, dets, forward_time = process(images, return_time=True)

    dets = post_process(dets, meta)
    ret = merge_outputs(dets)

    res = np.empty([1, 7])
    for i, c in ret.items():
        tmp_s = ret[i][ret[i][:, 5] > 0.3]
        tmp_c = np.ones(len(tmp_s)) * (i + 1)
        tmp = np.c_[tmp_c, tmp_s]
        res = np.append(res, tmp, axis=0)
    res = np.delete(res, 0, 0)
    res = res.tolist()
    return res


def pre_recall(root_path, device, path='',iou=0.5):
    imgs = os.listdir(root_path)
    num = 0
    all_pre_num = 0
    all_lab_num = 0
    miou = 0
    mang = 0
    a=path
    for img in imgs:
        if img.split('.')[-1] == 'jpg':
            img_path = os.path.join(root_path, img)
            # xml_path = os.path.join(a, img.split('.')[0] + '.xml')
            pre_ret = get_pre_ret(img_path, device)
            # lab_ret = get_lab_ret_from_xml(xml_path)
            lab_ret = get_lab_ret_from_json(path,img)
            if(len(lab_ret)==0):
                print("continue")
                continue
            all_pre_num += len(pre_ret)
            all_lab_num += len(lab_ret)
            for class_name, lx, ly, rx, ry, ang, prob in pre_ret:
                pre_one = np.array([(rx + lx) / 2, (ry + ly) / 2, rx - lx, ry - ly, ang])
                for cx, cy, w, h, ang_l in lab_ret:
                    lab_one = np.array([cx, cy, w, h, ang_l])
                    iou = iou_rotate_calculate(pre_one, lab_one)
                    ang_err = abs(ang - ang_l) / 180
                    if iou > 0.5:
                        num += 1
                        miou += iou
                        mang += ang_err
        else:
            continue
    print(num)
    print(all_pre_num)
    print(all_lab_num)
    print(mang)
    print(miou)
    return num / all_pre_num, num / all_lab_num, mang / num, miou / num


if __name__ == '__main__':
    import torch.nn as nn

    device = torch.device("cuda")
    gpus = [0,1,2,3]
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
    model.eval()
    # p, r, mang, miou = pre_recall('./data/palmprint/images/Tongji', device,'./labelGenerator/label')
    p, r, mang, miou = pre_recall('./data/palmprint/images/MPD/PalmSet', device,'./data/palmprint/annotations/MPD_train.json')
    F1 = (2 * p * r) / (p + r)
    print('miou: ', miou)
    # print('F1: ', F1)

    #  miou:  0.7513119339954271

    '''
    DLA-DCN
    12032
    12267
    12806
    152168.5494255132
    8891.84459260831
    miou:  0.7390163391463023
    '''
    '''
    DLA-DCN
    12818
    12818
    12806
    165347.78109112574
    10080.705813738794
    miou:  0.7864491975143387
    '''
    '''
    DLA-DCN
    12831
    12831
    12806
    165725.43430258168
    10100.24127978022
    miou:  0.7871749107458671
    '''