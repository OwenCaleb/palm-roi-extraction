from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import math
import logging
import numpy as np
from os.path import join

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from dcn.modules.deform_conv import ModulatedDeformConvPack as DCN
#from DCNv2.dcn_v2 import DCN

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
        # bottle_planes = dim * cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation, bias=False,
                               dilation=dilation, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            # 16 16 512 512
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            # 16 32 512 256
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, linear_root=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,
                           root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)
        # 经过5个stage后从  512 16  ->   16  512
        '''
        初始化方法是 He 初始化（也称为 Kaiming 初始化），它是 Xavier 初始化的一种改进，专门用于 ReLU 激活函数>
        He 初始化通过选择合适的方差来保持前向传播中信号的稳定性，防止梯度消失或爆炸。
        我们直接使用预训练模型。
        '''
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))
        # 被封装在 modules 列表中，并最终用 nn.Sequential(*modules) 封装成一个顺序容器，返回作为一个整体的卷积层。
        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                # 批量归一化用于对卷积层的输出进行标准化，以加速训练和提高模型的稳定性。在训练过程中以BN_MOMENTUM的速率更新其运行均值和方差
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)])
            inplanes = planes
        # 被封装在 modules 列表中，并最终用 nn.Sequential(*modules) 封装成一个顺序容器，返回作为一个整体的卷积层。
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        '''
        getattr 是 Python 内置的函数，允许动态获取对象的属性。
        'level{}'.format(i) 会生成像 'level0'、'level1' 这样的层级名称，
        getattr(self, 'level0')(x) 等价于调用 self.level0(x)
        '''
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        '''
        torch.Size([4, 16, 512, 512])
        torch.Size([4, 32, 256, 256])
        torch.Size([4, 64, 128, 128])
        torch.Size([4, 128, 64, 64])
        torch.Size([4, 256, 32, 32])
        torch.Size([4, 512, 16, 16])
        '''
        # for i, layer in enumerate(y):
        #     print(i)
        #     print(layer.shape)
        return y

    def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):
        # fc = self.fc
        if name.endswith('.pth'):
            model_weights = torch.load(data + name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)
        num_classes = len(model_weights[list(model_weights.keys())[-1]])
        self.fc = nn.Conv2d(
            self.channels[-1], num_classes,
            kernel_size=1, stride=1, padding=0, bias=True)
        self.load_state_dict(model_weights)
        # self.fc = fc


def dla34(pretrained=True, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    if pretrained:
        # 学到了基本的特征，比如边缘、纹理、形状等，在迁移学习的任务中可以作为网络的初始权重，从而加快训练速度，或者提高模型的泛化能力。
        model.load_pretrained_model(data='imagenet', name='dla34', hash='ba72cf86')
    return model

class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

# 初始化卷积层的偏置（bias）为零。
def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

# 初始化上采样（通常是转置卷积或上卷积）层的权重 类似于高斯的权重分布
def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.conv = DCN(chi, cho, kernel_size=(3,3), stride=1, padding=1, dilation=1, deformable_groups=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x


class IDAUp(nn.Module):
    '''
    ida_0(256,[256,512],[1,2])
    ida_1(128,[128,256,256],[1,2,2])
    ida_2(64,[64,128,128,128],[1,2,2,2])
    另外，又创建了一个 64 [64,128,256] [1,2,4]
    '''
    def __init__(self, o, channels, up_f):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i] #512  /  256  256/.../ 128 256
            f = int(up_f[i])  #2   / 2  2 /.../ 2 4
            proj = DeformConv(c, o) # 512->256  /  256->128 256->128/.../128->64 256->64
            node = DeformConv(o, o) # 256->256  /  128->128 128->128/.../64->64 64->64
            # print( f * 2) #4
            # print( f) #2
            # 16->32变大一倍 /32->64 变大一倍 32->64 变大一倍
            up = nn.ConvTranspose2d(o, o, f * 2, stride=f,
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)
                 
        
    def forward(self, layers, startp, endp):
        #  # ida_0(layers, 4, 6) ida_1(layers, 3, 6) ida_2(layers, 2, 6)
        '''
        单独创建的IDAUP 是 out0 out1 out2 out3
            0torch.Size([4, 64, 128, 128])
            1torch.Size([4, 128, 64, 64])
            2torch.Size([4, 256, 32, 32])
            startp=0
            endp=3
        '''
        for i in range(startp + 1, endp): #5 /4 5/.../1 2
            upsample = getattr(self, 'up_' + str(i - startp)) #1 /1  2
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])



class DLAUp(nn.Module):
    # 2  /  64 128 256 512 / 1 2 4 8
    def __init__(self, startp, channels, scales, in_channels=None):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        '''
        ida_0(256,[256,512],[1,2])
        ida_1(128,[128,256,256],[1,2,2])
        ida_2(64,[64,128,128,128],[1,2,2,2])
        '''
        for i in range(len(channels) - 1):# 0  1  2
            j = -i - 2 # -2   -3   -4
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(channels[j], in_channels[j:],
                          scales[j:] // scales[j]))
            # print(channels[j])
            # print( in_channels[j:])
            # print(scales[j:] // scales[j])
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        # torch.Size([4, 16, 512, 512])
        # torch.Size([4, 32, 256, 256])
        # torch.Size([4, 64, 128, 128])
        # torch.Size([4, 128, 64, 64])
        # torch.Size([4, 256, 32, 32])
        # torch.Size([4, 16, 512, 512])
        # for layer in layers:
        #     print(layer.shape)
        out = [layers[-1]]
        for i in range(len(layers) - self.startp - 1):# 0 1 2
            # print(i)
            ida = getattr(self, 'ida_{}'.format(i)) #ida_0 ida_1 ida_2
            '''
            ida_0(256,[256,512],[1,2])
            ida_1(128,[128,256,256],[1,2,2])
            ida_2(64,[64,128,128,128],[1,2,2,2])
            '''
            ida(layers, len(layers) -i - 2, len(layers)) # ida_0(layers, 4, 6) ida_1(layers, 3, 6) ida_2(layers, 2, 6)
            out.insert(0, layers[-1])
        return out


class Interpolate(nn.Module):
    def __init__(self, scale, mode):
        super(Interpolate, self).__init__()
        self.scale = scale
        self.mode = mode
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=False)
        return x


class Creat_DlaNet(nn.Module):
    def __init__(self, base_name, heads, pretrained, plot, down_ratio, final_kernel,
                 last_level, head_conv, out_channel=0):
        self.plot = plot
        super(Creat_DlaNet, self).__init__()
        # print(down_ratio) #4
        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level #5
        
        # globals()[base_name](pretrained=pretrained) 意思是在全局寻找一个叫 dla34 的函数或者类，
        # 他的参数 pretrained 为 true
        self.base = globals()[base_name](pretrained=pretrained)
        channels = self.base.channels #[16, 32, 64, 128, 256, 512]
        # print(self.first_level) #2 从第index=2,也就是第三阶段，也就是128 64开始上采样
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        # print(scales) #[1, 2, 4, 8]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)

        if out_channel == 0:
            out_channel = channels[self.first_level] #64

        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level], 
                            [2 ** i for i in range(self.last_level - self.first_level)])
        
        self.heads = heads
        # heads = {'hm': 1, 'wh': 2, 'ang':1, 'reg': 2}
        for head in self.heads:
            classes = self.heads[head]
            # head_conv=256
            if head_conv > 0:
              fc = nn.Sequential(
                  nn.Conv2d(channels[self.first_level], head_conv,
                    kernel_size=3, padding=1, bias=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(head_conv, classes, 
                    kernel_size=final_kernel, stride=1, 
                    padding=final_kernel // 2, bias=True))
              if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
              else:
                fill_fc_weights(fc)
            else:
              fc = nn.Conv2d(channels[self.first_level], classes, 
                  kernel_size=final_kernel, stride=1, 
                  padding=final_kernel // 2, bias=True)
              if 'hm' in head:
                fc.bias.data.fill_(-2.19)
              else:
                fill_fc_weights(fc)
            '''
            这行代码将构建好的卷积层 fc 动态地赋值给对象的 head 属性。
            例如，self.hm = fc，self.wh = fc，以便模型可以通过 self.hm、self.wh 等来直接访问对应的任务头。
            '''
            self.__setattr__(head, fc)

    def forward(self, x):
        x = self.base(x)
        x = self.dla_up(x)
        # for i ,item in enumerate(x):
        #     print(str(i)+str(item.shape))
        '''
            0torch.Size([4, 64, 128, 128])
            1torch.Size([4, 128, 64, 64])
            2torch.Size([4, 256, 32, 32])
            3torch.Size([4, 512, 16, 16])
        '''
        y = []
        for i in range(self.last_level - self.first_level): #0 1 2
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))
        '''
            0torch.Size([4, 64, 128, 128])
            1torch.Size([4, 64, 128, 128])
            2torch.Size([4, 64, 128, 128])
        '''
        # for i ,item in enumerate(y):
        #     print(str(i)+str(item.shape))
        z = {}
        res = [] # 为了画图
        # torch.Size([4, 1, 128, 128]) 'hm': 1, 'wh': 2, 'ang':1, 'reg': 2
        for head in self.heads:
            z[head] = self.__getattr__(head)(y[-1])
            res.append(self.__getattr__(head)(y[-1])) #为了画图，不画图就返回ret
        # print(len(res)) #4
        # print(res[2].shape) #torch.Size([2, 1, 128, 128])
        # print(res[2])
        return res if self.plot else z
        
        



def DlaNet(num_layers=34, heads = {'hm': 1, 'wh': 2, 'ang':1, 'reg': 2}, head_conv=256, plot=False):
    # 指定网络的多个输出头（heads）以及每个输出头的通道数。
    model = Creat_DlaNet('dla{}'.format(num_layers), heads,
                 pretrained=True,#从预训练模型加载权重，以加速训练或提高性能。
                 down_ratio=4,
                 final_kernel=1,
                 last_level=5,
                 head_conv=head_conv,#指定每个输出头前的卷积层通道数
                 plot = plot)
    return model

