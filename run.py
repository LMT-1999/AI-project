import os
import io
import math
import random
from itertools import count

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from IPython.display import display

import paddle
from paddle.io import Dataset
from paddle.vision.transforms import transforms


paddle.disable_static()
print(paddle.__version__)
# !cd data/data55873/ && unzip -qn images.zip
# 3.数据集
### 3.1 数据集下载
# 本案例使用BSR_bsds500数据集，下载链接：http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz
### 3.2 数据集概览

# 可以看到我们需要的图片文件在data/data55873/images文件夹下，train0、test各200张，val为100张。
### 3.3 数据集类定义
# 飞桨（PaddlePaddle）数据集加载方案是统一使用Dataset（数据集定义） + DataLoader（多进程数据集加载）。
#
# 首先我们先进行数据集的定义，数据集定义主要是实现一个新的Dataset类，继承父类paddle.io.Dataset，并实现父类中以下两个抽象方法，__getitem__和__len__：
class BSD_data(Dataset):
    """
    继承paddle.io.Dataset类
    """

    def __init__(self, mode='train0', channel=1, image_path="data/data55873/images/"):
        """
        实现构造函数，定义数据读取方式，划分训练和测试数据集
        """
        super(BSD_data, self).__init__()

        self.mode = mode.lower()
        self.channel = channel

        if self.mode == 'train0':
            self.image_path = os.path.join(image_path, 'train0')
        elif self.mode == 'val':
            self.image_path = os.path.join(image_path, 'val')
        else:
            raise ValueError('mode must be "train0" or "val"')

        # 原始图像的缩放大小
        self.crop_size = 300
        # 缩放倍率
        self.upscale_factor = 3
        # 缩小后送入神经网络的大小
        self.input_size = self.crop_size // self.upscale_factor
        # numpy随机数种子
        self.seed = 1337
        # 图片集合
        self.temp_images = []
        # 加载数据
        self._parse_dataset()

    def transforms(self, img):
        """
        图像预处理工具，用于将升维(100, 100) => (100, 100,1)，
        并对图像的维度进行转换从HWC变为CHW
        """
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        return img.transpose((2, 0, 1))

    def __getitem__(self, idx):
        """
        返回 缩小3倍后的图片 和 原始图片
        """

        # 加载原始图像
        img = self._load_img(self.temp_images[idx])
        # 将原始图像缩放到（3, 300, 300）
        img_big = img.resize([self.crop_size, self.crop_size], Image.BICUBIC)
        img_big = np.asarray(img_big, dtype='float32')
        img_big = img_big / 255.0

        # 缩放后的图像和前面采取一样的操作
        img_small = img.resize([self.input_size, self.input_size], Image.BICUBIC)
        img_small = np.asarray(img_small, dtype='float32')
        img_small = img_small / 255.0

        # 升纬并将HWC转换为CHW
        img_big = self.transforms(img_big)
        img_small = self.transforms(img_small)

        # x为缩小3倍后的图片（3, 100, 100） y是原始图片（3, 300, 300）
        return img_small, img_big

    def __len__(self):
        """
        实现__len__方法，返回数据集总数目
        """
        return len(self.temp_images)

    def _sort_images(self, img_dir):
        """
        对文件夹内的图像进行按照文件名排序
        """
        files = []

        for item in os.listdir(img_dir):
            if item.split('.')[-1].lower() in ["jpg", 'jpeg', 'png']:
                files.append(os.path.join(img_dir, item))

        return sorted(files)

    def _parse_dataset(self):
        """
        处理数据集
        """
        self.temp_images = self._sort_images(self.image_path)
        random.Random(self.seed).shuffle(self.temp_images)

    def _load_img(self, path):
        """
        从磁盘读取图片
        """
        with open(path, 'rb') as f:
            img = Image.open(io.BytesIO(f.read()))
            img = img.convert('RGB')
            return img
### 3.4 DataSet数据集抽样展示
# 实现好BSD_data数据集后，我们来测试一下数据集是否符合预期，因为BSD_data是一个可以被迭代的Class，我们通过for循环从里面读取数据进行展示。

# 测试定义的数据集
train_dataset = BSD_data(mode='train0')
val_dataset = BSD_data(mode='val')

print('=============train0 dataset=============')
x, y = train_dataset[0]
display(x.shape)
display(y.shape)

x = x * 255.0
y = y * 255.0

img_small = Image.fromarray(x.astype(np.uint8).transpose((1,2,0)), 'RGB')
img_big = Image.fromarray(y.astype(np.uint8).transpose((1,2,0)), 'RGB')

display(img_small)
display(img_small.size)
display(img_big)
display(img_big.size)

# 4.模型组网
# Sub_Pixel_CNN是一个全卷机网络，网络结构比较简单，这里采用Layer类继承方式组网。
class Sub_Pixel_CNN(paddle.nn.Layer):

    def __init__(self, upscale_factor=3, channels=3):
        super(Sub_Pixel_CNN, self).__init__()
        self.upscale_factor = upscale_factor
        self.conv1 = paddle.nn.Conv2D(channels,64,5,stride=1, padding=2)
        self.conv2 = paddle.nn.Conv2D(64,64,3,stride=1, padding=1,)
        self.conv3 = paddle.nn.Conv2D(64,32,3,stride=1, padding=1,)
        self.conv4 = paddle.nn.Conv2D(32,channels * (upscale_factor ** 2),3,stride=1, padding=1,weight_attr=paddle.nn.initializer.KaimingNormal())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = paddle.fluid.layers.pixel_shuffle(x, self.upscale_factor)
        return x

### 4.1 模型封装
model = paddle.Model(Sub_Pixel_CNN())

# 4.2 模型可视化
# 调用飞桨提供的summary接口对组建好的模型进行可视化，方便进行模型结构和参数信息的查看和确认。
model.summary((1, 3, 100, 100))

# 5.模型训练
# 5.1 模型训练准备
# 使用模型代码进行Model实例生成，使用prepare接口定义优化器、损失函数和评价指标等信息，用于后续训练使用。在所有初步配置完成后，调用fit接口开启训练执行过程，调用fit时只需要将前面定义好的训练数据集、测试数据集、训练轮次（Epoch）和批次大小（batch_size）配置好即可。
model.prepare(paddle.optimizer.Adam(learning_rate=0.0001,parameters=model.parameters()),
              paddle.nn.MSELoss()
             )
# 5.2 开始训练

# model.load('checkpoint_800/model_final',skip_mismatch=False,reset_optimizer=False);
model.fit(train_dataset,
          epochs=1,
          batch_size=32,
          verbose=1)

# model.load('checkpoint_200/model_final',skip_mismatch=False,reset_optimizer=False);
### 6.1 预测
# 我们可以直接使用model.predict接口来对数据集进行预测操作，只需要将预测数据集传递到接口内即可。
predict_results = model.predict(val_dataset)
### 6.2 定义预测结果可视化函数
import math
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


def psnr(img1, img2):
    """
    PSMR计算函数
    """
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def plot_results(img, title='results', prefix='out'):
    """
    画图展示函数
    """
    img_array = np.asarray(img, dtype='float32')
    img_array = img_array.astype("float32") / 255.0

    fig, ax = plt.subplots()
    im = ax.imshow(img_array[::-1], origin="lower")

    plt.title(title)
    axins = zoomed_inset_axes(ax, 2, loc=2)
    axins.imshow(img_array[::-1], origin="lower")

    x1, x2, y1, y2 = 200, 300, 100, 200
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    plt.yticks(visible=False)
    plt.xticks(visible=False)

    mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="blue")
    plt.savefig(str(prefix) + "-" + title + ".png")
    plt.show()


def get_lowres_image(img, upscale_factor):
    """
    缩放图片
    """
    return img.resize(
        (img.size[0] // upscale_factor, img.size[1] // upscale_factor),
        Image.BICUBIC,
    )


def upscale_image(model, img):
    '''
    输入小图，返回上采样三倍的大图像
    '''

    img = np.array(img, dtype='float32')
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)  # 升维度到（1,3,w,h）一个batch
    img = np.expand_dims(img, axis=0)  # 升维度到（1,1,3,w,h）可迭代的batch
    img = paddle.to_tensor(img)
    out = model.predict(img)  # predict输入要求为可迭代的batch
    out_img = out[0][0][0]  # 得到predict输出结果
    out_img = out_img.transpose((1, 2, 0))

    return out_img