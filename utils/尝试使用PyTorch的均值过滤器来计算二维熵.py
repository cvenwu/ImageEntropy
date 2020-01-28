"""
@File : 尝试使用PyTorch的均值过滤器来计算二维熵.py

@Author: sivan Wu

@Date : 2020/1/27 7:48 上午

@Desc :  参考https://blog.csdn.net/lyl771857509/article/details/84113177

"""

import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torch.nn.functional import conv2d
from collections import Counter



transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])


class Entropy_debug(nn.Module):
    def __init__(self, transform=transform):
        super(Entropy_debug, self).__init__()
        self.kernel = np.ones([3, 3], dtype=np.float32)
        self.kernel = torch.FloatTensor(self.kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=self.kernel, requires_grad=False)
        self.transform = transform

    def forward(self, input):
        input = self.transform(input)
        # print(input.shape[0])
        # 判断输入图片的维度
        if input.shape[0] == 1:
            print(type(input.shape))
            print(input.shape)
        else:
            self.kernel = torch.FloatTensor(self.kernel).expand(1, 3, 3, 3)
            # 如果图片有3个通道则分别裁减3个通道
            # x1 = input[0].unsqueeze(0)
            # x1 = conv2d(x1, self.kernel, stride=[1,1], padding=2)
            # x2 = input[1].unsqueeze(0)
            # x2 = conv2d(x2, self.kernel, stride=[1,1], padding=2)
            # x3 = input[2].unsqueeze(0)
            # x3 = conv2d(x3, self.kernel, stride=[1,1], padding=2)
            # input = torch.cat([x1, x2, x3], dim=1)
            # print(x1.shape)
            # print(input.shape[0])
        input = input.unsqueeze(0)
        # 因为对于一张二维图，stride也是二维的，否则报如下错误，参考https://blog.csdn.net/gdymind/article/details/82933534
        # RuntimeError: expected stride to be a single integer value or a list of 1 values to match the convolution dimensions, but got stride=[1, 1]
        # def conv2d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride: Union[_int, _size] = 1,
        #            padding: Union[_int, _size] = 0, dilation: Union[_int, _size] = 1, groups: _int = 1) -> Tensor:
        #     ...
        print(self.kernel.shape)
        return conv2d(input, self.kernel, stride=[1, 1], padding=1)
        # return self.main(self.transform(input))


class LinearMap(object):

    def __init__(self):
        self.items = []

    def add(self, k, v):
        self.items.append((k, v))

    def get(self, k):
        for key, val in self.items:
            if key == k:
                return val
        raise KeyError



class Entropy(nn.Module):
    def __init__(self, transform=transform):
        super(Entropy, self).__init__()
        self.kernel = np.ones([3, 3], dtype=np.float32)
        self.kernel = torch.FloatTensor(self.kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=self.kernel, requires_grad=False)
        self.transform = transform
        self.count = []

    def forward(self, input):
        # 统计二元组出现次数，其中第二个元素为原图的均值滤波器对应位置的entropy值，自己发现都小于10，因此将其乘以0.1转换为整数
        # 第一个元素为input的对应的像素内容，但是归一化了
        output = self.calc_entropy(input)
        output_np = output.numpy().ravel()
        print(output_np.shape)
        input_np = np.array(self.transform(input).numpy().ravel() * 100000, np.int32)
        input_np = input_np.astype(np.int32)
        # print(input_np)
        # print(input_np.shape)
        print("判断是否都小于10(最终的值应该为921600):",np.sum(output_np < 10))
        output_np = output_np * 1000
        output_np = output_np.astype(np.int32)
        result = input_np * 10000 + output_np
        print(result)
        print(input_np)
        print(output_np)

        value_list = np.bincount(result)
        total_num = np.sum(value_list)
        probability = value_list.astype(np.float32) / total_num
        # print("概率之和为：", np.sum(probability))
        probability = probability[np.flatnonzero(probability)]
        print(probability.shape)
        probability = np.expand_dims(probability, axis=0)
        print("概率之和为：", np.sum(probability))

        entropy = -np.matmul(probability, np.log2(probability).T)
        print(entropy)  #[[12.57139422]]
        return entropy

    def calc_entropy(self, input):
        input = self.transform(input)
        # 判断输入图片的维度
        if input.shape[0] == 3:
            result = torch.rand([1, 3, input.shape[1], input.shape[2]])
            # 分别计算每个维度的过滤器均值，然后进行拼接
            for i in range(3):
                # input[i]得到的是[image_height, image_width]
                x = input[i].unsqueeze(0).unsqueeze(0)
                output = conv2d(x, self.kernel, stride=[1, 1], padding=1)
                result[:, i] = output
            return result
        else:
            return conv2d(input.unsqueeze(0), self.kernel, stride=[1, 1], padding=1)


if __name__ == '__main__':
    import time
    start_time = time.time()
    image_path = "../data/cartoon/0.png"
    image = Image.open(image_path)
    net = Entropy(transform)
    entropy = net(image)
    end_time = time.time()
    print(end_time-start_time)  #18.979233980178833
    # print(entropy.shape)
    # print(image.size)
    # print(entropy)

    # 概率之和： 1.0
    # 熵: 12.736221438469448
    # 读取： / Users / yirufeng / Desktop / 科研 / 数据集 / AnimeGAN实验数据集 / 千与千寻 / 6_1.j
    # pg
    # 熵为： 12.736221438469448

    # (921600,)
    # 判断是否都小于10(最终的值应该为921600): 921600
    # [890193529 890195313 898035341... 278431670 278431670 278431113]
    # [89019 89019 89803... 27843 27843 27843]
    # [3529 5313 5341... 1670 1670 1113]
    # (61602,)
    # 概率之和为： 1.0
    # [[12.57139422]]



