

# Palm-ROI-Extraction

Palm-ROI-Extraction
<br />

</p>

## 目录

- [上手指南](#上手指南)
  - [开发前的配置要求](#开发前的配置要求)
  - [安装步骤](#安装步骤)
- [文件目录说明](#文件目录说明)
- [其它说明](#其它说明)

### 上手指南

###### 开发前的配置要求

1. 虚拟环境创建python3.8cuda12.1
2. 其它需要的包

###### **安装步骤**

1. Clone the repo

```sh
git clone https://github.com/OwenCaleb/palm-roi-extraction
```

### 文件目录说明

eg:请按以下结构进行项目组织

```
filetree 
│  ├── /backbone/
│  │  ├── dlanet.py
│  │  ├── dlanet_dcn.py
│  │  ├── resnet.py
│  │  └── resnet_dcn.py
│  ├── /best_roi_model_img_output/
│  │  ├── /fail_images/
│  │  ├── /MPD/
│  │  │  ├── 001_1_h_l_04.png
│  │  │  ├── ...
│  │  │  └── *.png
│  │  └── ...(Other DataSet)
│  ├── /data/
│  │  └── /palmprint/
│  │  │  ├── /annotations/
│  │  │  │  ├── MPD_train.json
│  │  │  │  ├── MPD_val.json
│  │  │  │  ├── *_train.json(Other DataSet)
│  │  │  │  └── *_val.json(Other DataSet)
│  │  │  └── /images/
│  │  │  │  ├── /MPD/
│  │  │  │  │  ├── 001_1_h_l_04.png
│  │  │  │  │  ├── ...
│  │  │  │  │  └── *.png
│  │  │  │  └── ...(Other DataSet)
│  ├── /dcn/
│  │  ├── /functions/
│  │  │  ├── __init__.py
│  │  │  ├── deform_conv.py
│  │  │  └── deform_pool.py
│  │  ├── /modules/
│  │  │  ├── __init__.py
│  │  │  ├── deform_conv.py
│  │  │  └── deform_pool.py
│  │  ├── deform_conv_cuda.cpython-38-x86_64-linux-gnu.so
│  │  └── deform_pool_cuda.cpython-38-x86_64-linux-gnu.so
│  ├── /labelGenerator/
│  │  ├── /images_train/
│  │  ├── /images_val/
│  │  ├── /label/
│  │  │  ├── ...
│  │  │  └── *.xml
│  │  ├── annotation_trans.py
│  │  ├── trans.py
│  │  └── voc2coco.py
│  ├── /test_data_base/
│  │  ├── /MPD/
│  │  │  ├── 001_1_h_l_04.png
│  │  │  ├── ...
│  │  │  └── *.png
│  │  └── ...(Other DataSet)
├── dataset.py
├── Loss.py
├── predict.py
├── /templates/
├── train.py
└── train-o.py。
```

### 其它说明

待补充
