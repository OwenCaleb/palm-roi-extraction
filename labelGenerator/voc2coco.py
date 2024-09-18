import json
import cv2
import numpy as np
import glob
import PIL.Image
import os,sys
import math
 
class PascalVOC2coco(object):
    def __init__(self, xml=[], save_json_path='./new.json'):
        '''
        :param xml: 所有Pascal VOC的xml文件路径组成的列表
        :param save_json_path: json保存位置
        '''
        self.xml = xml
        self.save_json_path = save_json_path
        self.images = []
        self.categories = []
        self.annotations = []
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0
        self.ob = []
        self.save_json()
 
    def data_transfer(self):
        for num, json_file in enumerate(self.xml):
            # 进度输出
            sys.stdout.write('\rConverting image progress : {%d/%d} current file : {%s}\n' % (num + 1, len(self.xml),json_file))
            sys.stdout.flush()
            self.json_file = json_file
            self.num = num
            # ./ label
            path = os.path.dirname(self.json_file)
            # .
            path = os.path.dirname(path)
            self.ob=[]
            # 打开文件
            with open(json_file, 'r', encoding='UTF-8') as fp:
                flag = 0
                for p in fp:
                    f_name = 1
                    if 'filename' in p:
                        # self.filen_ame = p.split('>')[1].split('<')[0]
                        self.filen_ame = json_file.split('/')[-1].split('.')[0] + '.' + p.split('>')[1].split('<')[0].split('.')[-1]
                        f_name = 0
                        # ./SegmentationObject/BJTU_004_F_L2.jpg
                        self.path = os.path.join(path, 'SegmentationObject', self.filen_ame)
                    if 'width' in p:
                        self.width = int(p.split('>')[1].split('<')[0])
                    if 'height' in p:
                        self.height = int(p.split('>')[1].split('<')[0])
                        self.images.append(self.image())
                    if flag == 1:
                        # name 'palmprint'
                        self.supercategory = self.ob[0]
                        if self.supercategory not in self.label:
                            self.categories.append(self.categorie())
                            self.label.append(self.supercategory)
                        # print(self.ob)
                        x1 = float(self.ob[1])
                        y1 = float(self.ob[2])
                        w = float(self.ob[3])
                        h = float(self.ob[4])
                        angle = (2 * math.pi - float(self.ob[5]) + math.pi / 2) * 180/math.pi
                        angle = angle if angle < 360 else angle-360

                        # boundind box 左上右下->左上宽高 后面dataset中处理这里不用管
                        # rectangle只在本文件分割中使用
                        self.rectangle = [x1 - w / 2, y1 - h / 2, x1 + w / 2, y1 + h / 2]  # VOC 格式
                        self.bbox =[x1, y1, w, h, angle]  # COCO 对应格式[x,y,w,h]
 
                        self.annotations.append(self.annotation())
                        self.annID += 1
                        self.ob = []
                        flag = 0
                    elif f_name == 1:
                        key = p.split('>')[0].split('<')[1]
                        # print(key)
                        # print(self.ob)
                        # print(key)
                        if key == 'name':
                            self.ob.append(p.split('>')[1].split('<')[0])
                        if key == 'cx':
                            self.ob.append(p.split('>')[1].split('<')[0])
                        if key == 'cy':
                            self.ob.append(p.split('>')[1].split('<')[0])
                        if key == 'w':
                            self.ob.append(p.split('>')[1].split('<')[0])
                        if key == 'h':
                            self.ob.append(p.split('>')[1].split('<')[0])
                        if key == 'angle':
                            self.ob.append(p.split('>')[1].split('<')[0])
                            flag = 1
 
 
        sys.stdout.write('\n')
        sys.stdout.flush()
 
    def image(self):
        image = {}
        image['height'] = self.height
        image['width'] = self.width
        image['id'] = self.num + 1
        image['file_name'] = self.filen_ame
        return image
 
    def categorie(self):
        categorie = {}
        # 'palmprint'
        categorie['supercategory'] = self.supercategory
        categorie['id'] = len(self.label) + 1  # 0 默认为背景
        categorie['name'] = self.supercategory
        return categorie
 
    def annotation(self):
        annotation = {}

        annotation['segmentation'] = [list(map(float, self.getsegmentation()))]
        annotation['iscrowd'] = 0
        annotation['image_id'] = self.num + 1

        annotation['bbox'] = self.bbox
        annotation['category_id'] = self.getcatid(self.supercategory)
        annotation['id'] = self.annID
        return annotation
 
    def getcatid(self, label):
        for categorie in self.categories:
            if label == categorie['name']:
                return categorie['id']
        return -1
 
    def getsegmentation(self):
 
        try:
            mask_1 = cv2.imread(self.path, 0)
            mask = np.zeros_like(mask_1, np.uint8)
            rectangle = self.rectangle
            rectangle = [int(x) for x in rectangle]
            print(rectangle)

            # print(type(rectangle[0]), type(rectangle[1]), type(rectangle[2]), type(rectangle[3]))
            # 提取灰度图框中部分,切片操作时，索引值必须是整数类型
            mask[rectangle[1]:rectangle[3], rectangle[0]:rectangle[2]] = mask_1[rectangle[1]:rectangle[3],
                                                                         rectangle[0]:rectangle[2]]

            save_dir = './mask_image_original'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                print('Creating folder sucess!')
            # 构建完整的文件路径
            save_path = os.path.join(save_dir, self.filen_ame)
            cv2.imwrite(save_path, mask)
            print('Save mask_original success!')

            # 计算矩形中点像素值
            mean_x = (rectangle[0] + rectangle[2]) // 2
            mean_y = (rectangle[1] + rectangle[3]) // 2

            # end是矩形区域右边界或图像宽度的最小值，start是矩形区域左边界或0的最大值。
            end = min((mask.shape[1], int(rectangle[2]) + 1))
            start = max((0, int(rectangle[0]) - 1))
 
            # flag = True
            # for i in range(mean_x, end):
            #     x_ = i;
            #     y_ = mean_y
            #     pixels = mask_1[y_, x_]
            #     if pixels != 0 and pixels != 220:  # 0 对应背景 220对应边界线
            #         mask = (mask == pixels).astype(np.uint8)
            #         flag = False
            #         break
            # if flag:
            #     for i in range(mean_x, start, -1):
            #         x_ = i;
            #         y_ = mean_y
            #         pixels = mask_1[y_, x_]
            #         if pixels != 0 and pixels != 220:
            #             mask = (mask == pixels).astype(np.uint8)
            #             break
            self.mask = mask

            save_dir = './mask_image'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                print('Creating folder sucess!')
            # 构建完整的文件路径
            save_path = os.path.join(save_dir, self.filen_ame)
            cv2.imwrite(save_path, mask)
            print('Save mask success!')

            return self.mask2polygons()


        except Exception as e:

            print(f"An error occurred: {e}")

            return [0]
 
    def mask2polygons(self):
        # 在 OpenCV3.x版本中，它返回三个值：修改后的图像、轮廓和它们的层次结构。
        # 在 OpenCV4.x及更高版本中，它只返回两个值：轮廓列表和它们的层次结构
        contours = cv2.findContours(self.mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找到轮廓线
        bbox=[]
        for cont in contours[0]:
            [bbox.append(i) for i in list(cont.flatten())]
            # map(bbox.append,list(cont.flatten()))

        # 创建一个空白图像用于绘制轮廓
        contour_image = np.zeros_like(self.mask)

        # 绘制轮廓
        cv2.drawContours(contour_image, contours[0], -1, (255, 255, 255), 1)  # 轮廓颜色为白色，线宽为1

        contour_save_dir = './contour_image'
        if not os.path.exists(contour_save_dir):
            os.makedirs(contour_save_dir)
            print('Creating folder for contours success!')

        contour_save_path = os.path.join(contour_save_dir, 'contours_' + self.filen_ame)
        cv2.imwrite(contour_save_path, contour_image)
        print('Save contour image success!')


        return bbox # list(contours[1][0].flatten())

  
    def data2coco(self):
        data_coco = {}
        data_coco['images'] = self.images
        data_coco['categories'] = self.categories
        data_coco['annotations'] = self.annotations
        return data_coco
 
    def save_json(self):
        self.data_transfer()
        self.data_coco = self.data2coco()
        print('success')
        # 保存json文件
        json.dump(self.data_coco, open(self.save_json_path, 'w'), indent=4)  # indent=4 更加美观显示
 
 
xml_file = glob.glob('./images_train/*.xml')
PascalVOC2coco(xml_file, '../data/palmprint/annotations/Tongji_train.json')

xml_file = glob.glob('./images_val/*.xml')
PascalVOC2coco(xml_file, '../data/palmprint/annotations/Tongji_val.json')


