import os
import random
import shutil
from shutil import copy, rmtree

def mk_file(file_path: str):
    if os.path.exists(file_path):
        # return
        # 如果文件夹存在，则先删除原文件夹在重新创建
        rmtree(file_path)
    os.makedirs(file_path)

source_folder = "./label"  # 源文件夹路径
val_folder = "./images_val"  # 目标文件夹路径
train_folder = "./images_train"  # 目标文件夹路径
mk_file(val_folder)
mk_file(train_folder)
num_images_to_copy = 500  # 要复制的图片数量

# 获取源文件夹中的所有图片文件
image_files = [f for f in os.listdir(source_folder) ]

# 随机选择要复制的XML文件 五分之一测试复制到val文件夹里
selected_images = random.sample(image_files, len(image_files)//5)

# 复制选定的图片文件到目标文件夹
for image_file in image_files:
    source_path = os.path.join(source_folder, image_file)
    if image_file in selected_images:
        val_path = os.path.join(val_folder, image_file)
        shutil.copyfile(source_path, val_path)
    else:
        train_path = os.path.join(train_folder, image_file)
        shutil.copyfile(source_path, train_path)
print(f"{len(image_files)-len(selected_images)}份XML已复制到train文件夹。")
print(f"{len(selected_images)}份XML已复制到val文件夹。")



