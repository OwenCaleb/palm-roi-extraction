import torch
import matplotlib.pyplot as plt
import scipy.io
import os
from PIL import Image
import json


def get_label_position(slots, marks):
    # slots: [4]
    # marks: [k, 2] or [3, 2]
    # pointx: [2]
    point1 = marks[slots[0] - 1, :]
    point2 = marks[slots[1] - 1, :]
    point3 = marks[slots[2] - 1, :]
    point4 = marks[slots[3] - 1, :]

    point5 = (point1 + point2) / 2
    point6 = (point3 + point4) / 2

    rotate_vec = torch.tensor([[0., -1.], [1., 0.]], dtype=point1.dtype)
    vec1 = torch.matmul(point2 - point1, rotate_vec)
    sideLength1 = torch.linalg.norm(vec1)
    vec1 = vec1 / sideLength1

    vec2 = torch.matmul(point4 - point3, rotate_vec)
    sideLength2 = torch.linalg.norm(vec2)
    vec2 = vec2 / sideLength2

    vec = torch.matmul(point6 - point5, rotate_vec)
    sideLength = torch.linalg.norm(vec)
    vec = vec / sideLength

    slot_point1 = point5 * 7 / 4 - point6 * 3 / 4 - vec * sideLength / 2
    slot_point2 = point5 * 7 / 4 - point6 * 3 / 4 - vec * sideLength * 3
    slot_point3 = point6 * 7 / 4 - point5 * 3 / 4 - vec * sideLength / 2
    slot_point4 = point6 * 7 / 4 - point5 * 3 / 4 - vec * sideLength * 3

    return slot_point1, slot_point2, slot_point3, slot_point4


def read_mat_file(file_path, file_name):
    mat_data = scipy.io.loadmat(os.path.join(file_path, file_name))

    return torch.tensor(mat_data['slots'], dtype=torch.int32).squeeze(0), torch.tensor(mat_data['marks'], dtype=torch.float32)


def get_cocoformat_data(slot_points):
    # slot_points: [4, 2]
    center_point = torch.sum(slot_points, dim=0) / 4
    _, max_y_index = torch.max(slot_points[0:2, 1], dim=0)
    _, min_y_index = torch.min(slot_points[0:2, 1], dim=0)

    if slot_points[min_y_index, 0] >= slot_points[max_y_index, 0]:
        h = torch.norm(slot_points[max_y_index, :] - slot_points[min_y_index, :])
        w = torch.norm(slot_points[0, :] - slot_points[2, :])
        v1 = torch.tensor([0, -1], dtype=slot_points.dtype)
        v2 = slot_points[min_y_index, :] - slot_points[max_y_index, :]
        angle = angel_between(v1, v2)
        # (2 * math.pi - float(self.ob[5]) + math.pi / 2) * 180/math.pi
    else:
        w = torch.norm(slot_points[max_y_index, :] - slot_points[min_y_index, :])
        h = torch.norm(slot_points[0, :] - slot_points[2, :])
        v1 = torch.tensor([-1, 0], dtype=slot_points.dtype)
        v2 = slot_points[min_y_index, :] - slot_points[max_y_index, :]
        angle = angel_between(v1, v2)

    return center_point, w, h, angle


def generate_mpd_json_file(image_mat_path, store_json_path, store_json_file, split_test=False):
    total_file_name = os.listdir(image_mat_path)
    image_file_name_list = []
    mat_file_name_list = []
    for file_name in total_file_name:
        if file_name.endswith('.jpg'):
            image_file_name_list.append(file_name)
        elif file_name.endswith('.mat'):
            mat_file_name_list.append(file_name)

    image_mat_tuple_list = []
    for image_file_name in image_file_name_list:
        correspond_mat_file_name = image_file_name[:-3] + 'mat'
        if correspond_mat_file_name in mat_file_name_list:
            image_mat_tuple_list.append((image_file_name, correspond_mat_file_name))
    print(f"The number of image and mat config all exists: {len(image_mat_tuple_list)}")

    # create json file
    train_json_content = {}
    train_json_content['images'] = []
    train_json_content["categories"] = [{"supercategory": "palmprint", "id": 1, "name": "palmprint"}]
    train_json_content["annotations"] = []

    if split_test:
        test_json_content = {}
        test_json_content['images'] = []
        test_json_content["categories"] = [{"supercategory": "palmprint", "id": 1, "name": "palmprint"}]
        test_json_content["annotations"] = []
        test_image_id = 1
        test_annotation_id = 1

    # define vari

    train_image_id = 1
    train_annotation_id = 1

    for (image_file_name, mat_file_name) in image_mat_tuple_list:
        if split_test:
            rand_num = torch.rand(1).item()
        # images
        image_message = {}
        image = Image.open(os.path.join(image_mat_path, image_file_name))
        image_message["weight"], image_message['height'] = image.size

        image_message["file_name"] = image_file_name
        if split_test and rand_num >= 0.2:
            image_message["id"] = train_image_id
            train_json_content['images'].append(image_message)
        else:
            image_message["id"] = test_image_id
            test_json_content['images'].append(image_message)

        # annotations
        annotation = {}
        annotation["segmentation"] = [[0.0]]
        annotation["iscrowd"] = 0

        slots, marks = read_mat_file(image_mat_path, mat_file_name)
        slot_point1, slot_point2, slot_point3, slot_point4 = get_label_position(slots=slots, marks=marks)
        slot_points = torch.cat((slot_point1.unsqueeze(0), slot_point2.unsqueeze(0), slot_point3.unsqueeze(0), slot_point4.unsqueeze(0)), dim=0)
        center_point, w, h, angle = get_cocoformat_data(slot_points)
        annotation["bbox"] = [center_point[0].item(), center_point[1].item(), w.item(), h.item(), angle.item()]
        annotation["category_id"] = 1

        if split_test and rand_num >= 0.2:
            annotation["image_id"] = train_image_id
            annotation["id"] = train_annotation_id
            train_json_content["annotations"].append(annotation)
            train_image_id = train_image_id + 1
            train_annotation_id = train_annotation_id + 1
        else:
            annotation["image_id"] = test_image_id
            annotation["id"] = test_annotation_id
            test_json_content["annotations"].append(annotation)
            test_image_id = test_image_id + 1
            test_annotation_id = test_annotation_id + 1

    print(f"Train images number: {train_image_id - 1}")
    print(f"Test iamges number: {test_image_id - 1}")
    with open(os.path.join(store_json_path, store_json_file), 'w') as json_file:
        json.dump(train_json_content, json_file, indent=4)

    if split_test:
        with open(os.path.join(store_json_path, "../data/palmprint/annotations/MPD_val.json"), 'w') as test_json_file:
            json.dump(test_json_content, test_json_file, indent=4)

    print("Completed!")


def angel_between(v1, v2):
    # 计算向量的点积
    dot_product = torch.dot(v1, v2)

    # 计算向量的模
    norm_v1 = torch.norm(v1)
    norm_v2 = torch.norm(v2)

    # 计算夹角的余弦值
    cos_theta = dot_product / (norm_v1 * norm_v2)

    # 通过反余弦函数获取角度，并将其转换为度数
    angle_rad = torch.acos(cos_theta)  # 弧度制
    angle_deg = torch.rad2deg(angle_rad)  # 将弧度制转换为角度制

    return angle_deg


if __name__ == '__main__':
    image_mat_path = '../data/palmprint/images/MPD/PalmSet'
    store_json_path = './'
    store_json_file = '../data/palmprint/annotations/MPD_train.json'
    generate_mpd_json_file(image_mat_path, store_json_path, store_json_file, split_test=True)
