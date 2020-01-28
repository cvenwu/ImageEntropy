"""
@File : entqweqwropy.py

@Author: sivan Wu

@Date : 2020/1/19 8:05 上午
e
@Desc : 计算图像熵的多种方法

"""

import cv2
import numpy as np
from collections import Counter


def calc_entropy_two_dims_gray_debug(image_path, area=3):
    """
    计算图像空间二维熵：https://blog.csdn.net/marleylee/article/details/78813630
        p(i, j) = f(i, j) / N*N  其中i表示像素的灰度值(0 <= i <= 255)，j 表示邻域灰度均值(0 <= j <= 255)
    上式能反应某像素位置上的灰度值与其周围像素灰度分布的综合特征，其中f(i, j)为特征二元组(i, j)出现的频数，N 为图像的尺度。
        H = -sum_i(sum_j(p(i, j) * log2 p(i, j)))
    :param image: 要计算熵的图像
    :param area: 空间区域
    :return: 返回计算好的熵
    """
    # image_gray = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
    image_gray = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    np.random.seed(12)
    original_height, original_width = image_gray.shape
    print("原图：height:", original_height, ", width:", original_width)
    # 首行增加0
    image_gray = np.insert(image_gray, 0, values=0.0, axis=0)
    # 首列增加0
    image_gray = np.insert(image_gray, 0, values=0.0, axis=1)
    # print(image_gray.shape)
    # 最后一行加全0
    image_gray = np.row_stack((image_gray, np.zeros([original_width + 1])))
    # print(image_gray.shape)
    # 最后一列加全0
    image_gray = np.column_stack((image_gray, np.zeros([original_height + 2])))
    # print(image_gray.shape)
    # print(image_gray)

    # 计算f(i, j)
    kernel = np.ones((area, area), np.float32) / (area ** 2)
    dst_image = cv2.filter2D(image_gray, -1, kernel)
    # print(dst_image.shape)
    # print(dst_image)

    # 根据均值滤波器计算得到的像素均值
    dst_image = dst_image[1:-1, 1:-1]
    # print(dst_image.shape)  # 和原图(不填充两行两列的0)一样大小

    # 计算均值滤波器
    height, width = image_gray.shape
    f_i_j = []
    for i in range(height - 2):
        for j in range(width - 2):
            # 存储均值滤波器中心点位于原图的像素
            f_i = image_gray[i + 1, j + 1]
            # 存储均值滤波器计算时候的均值像素
            f_j = dst_image[i, j]
            temp = (f_i, f_j)
            f_i_j.append(temp)

    # times2 = Counter(f_i_j)
    # print(times2)
    # print(type(times2))
    # for i in dict(times2):
    #     print(i)
    print('---')
    print(len(f_i_j))  # 253200
    # print(f_i_j)
    f_i_j_set = set(f_i_j)
    print(len(f_i_j_set))  # 28880
    probability = []
    count = 1
    for ele in f_i_j_set:
        probability.append(f_i_j.count(ele))
        print("开始第", count, "次")
        count += 1
    probability = np.array(probability, np.float32) / (original_width * original_height)
    print("概率之和：", np.sum(probability))
    print(f_i_j_set)
    print(len(f_i_j_set))  # 28880
    # 统计位于列表中出现特征二元组的次数，同时计算概率 ,参考：https://segmentfault.com/q/1010000016716175?utm_source=tag-newest
    # value, times = np.unique(f_i_j, return_counts=True)
    # print(value)
    # print(times)
    # print("times长度：", len(times))
    # print(type(times))
    # print(np.sum(times))
    # probability = times * 1.0 / (original_height * original_width)
    # print(probability)
    # print(sum(probability))
    # print("he")
    # for i in f_i_j_set:
    #     probability.append(f_i_j.count(i))
    # probability = np.array(probability, np.float32)
    # probability = probability * 1.0 / (area * area)
    print(sum(probability))
    print(probability)
    # 根据计算出的概率和熵公式计算熵
    H = -np.matmul(probability, np.log2(probability).T)
    print(H)
    return H


def calc_entropy_two_dims_gray(image_path, area=3):
    """
    计算图像空间二维熵：https://blog.csdn.net/marleylee/article/details/78813630
        p(i, j) = f(i, j) / N*N  其中i表示像素的灰度值(0 <= i <= 255)，j 表示邻域灰度均值(0 <= j <= 255)
    上式能反应某像素位置上的灰度值与其周围像素灰度分布的综合特征，其中f(i, j)为特征二元组(i, j)出现的频数，N 为图像的尺度。
        H = -sum_i(sum_j(p(i, j) * log2 p(i, j)))
    :param image: 要计算熵的图像
    :param area: 空间区域
    :return: 返回计算好的熵
    """
    image_gray = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
    np.random.seed(12)
    original_height, original_width = image_gray.shape
    # 首行增加0
    image_gray = np.insert(image_gray, 0, values=0.0, axis=0)
    # 首列增加0
    image_gray = np.insert(image_gray, 0, values=0.0, axis=1)
    # 最后一行加全0
    image_gray = np.row_stack((image_gray, np.zeros([original_width + 1])))
    # 最后一列加全0
    image_gray = np.column_stack((image_gray, np.zeros([original_height + 2])))

    # 计算f(i, j)
    kernel = np.ones((area, area), np.float32) / (area ** 2)
    dst_image = cv2.filter2D(image_gray, -1, kernel)

    # 根据均值滤波器计算得到的像素均值
    dst_image = dst_image[1:-1, 1:-1]

    # 计算均值滤波器
    height, width = image_gray.shape
    f_i_j = []
    for i in range(height - 2):
        for j in range(width - 2):
            # 存储均值滤波器中心点位于原图的像素
            f_i = image_gray[i + 1, j + 1]
            # 存储均值滤波器计算时候的均值像素
            f_j = dst_image[i, j]
            temp = (f_i, f_j)
            f_i_j.append(temp)

    f_i_j_set = set(f_i_j)
    probability = []
    count = 1
    for ele in f_i_j_set:
        probability.append(f_i_j.count(ele))
        count += 1
    probability = np.array(probability, np.float32) / (original_width * original_height)
    print("概率之和：", np.sum(probability))
    H = -np.matmul(probability, np.log2(probability).T)
    print("熵:", H)
    return H


# 填充0计算均值
if __name__ == '__main__':
    # image = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    # kernel = np.ones([3, 3], np.float32) / 9.0
    # print(image.shape)
    # print(kernel)
    # dst_image = cv2.filter2D(image, -1, kernel)
    # print(dst_image)

    # calc_entropy_two_dims_gray_debug("../data/real.jpeg", area=3)  # 2.6674821
    calc_entropy_two_dims_gray("../data/real.jpeg", area=3)  # 12.21207028369082
    # calc_entropy_two_dims_gray3("../data/real.jpeg", area=3)  # 12.21207028369082
