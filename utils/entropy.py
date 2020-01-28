"""
@File : entropy.py

@Author: sivan Wu

@Date : 2020/1/20 2:19 下午

@Desc :

"""


"""
@File : image_entropy.py

@Author: sivan Wu

@Date : 2020/1/20 1:26 下午

@Desc : calculate the entropy of image in one-dim, two-dim of gray image and other color spaces.

"""

import face_recognition
import cv2
import numpy as np
import matplotlib.pyplot as plt


def cut_photo(image_path):
    """
    根据图片文件路径读取图片，返回裁剪后的图片
    :param image_path:
    :param locations:
    :return:
    """
    image = cv2.imread(image_path)
    image_locations = face_recognition.face_locations(face_recognition.load_image_file(image_path))
    if len(image_locations) != 0:
        image = image[image_locations[0][0]:image_locations[0][2], image_locations[0][3]:image_locations[0][1], :]
    return image

def calc_entropy_gray_image(image):
    """
    计算灰度图的熵，使用最简单的熵计算公式
    :param image: 使用opencv读取的灰度图
    :return:
    """
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist[hist>0]
    probablity = hist / np.sum(hist)
    print('sum of probability:', np.sum(probablity))
    result = np.sum(np.matmul(probablity.T, np.log2(probablity)))
    return -result

def calc_image_entropy_HSV(image):
    """
    :param image:
    :return:
    """
    image_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    size = image_HSV.shape
    result = []
    for i in range(size[len(size) - 1]):
        hist = cv2.calcHist([image_HSV], [i], None, [256], [0, 256])
        hist = hist[hist > 0]
        # print(hist)
        probablity = hist / np.sum(hist)
        print('sum of probability:', np.sum(probablity))
        result.append(-(np.matmul(probablity.T, np.log2(probablity))))
    print(result)
    return result


def calc_image_entropy_YCrCb(image):
    """
    :param image:
    :return:
    """
    image_YCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    size = image_YCrCb.shape
    result = []
    for i in range(size[len(size) - 1]):
        hist = cv2.calcHist([image_YCrCb], [i], None, [256], [0, 256])
        hist = hist[hist > 0]
        # print(hist)
        probablity = hist / np.sum(hist)
        print('sum of probability:', np.sum(probablity))
        result.append(-(np.matmul(probablity.T, np.log2(probablity))))
    print(result)
    return result


def calc_image_entropy_HLS(image):
    """
    :param image:
    :return:
    """
    image_HLS = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    size = image_HLS.shape
    result = []
    for i in range(size[len(size) - 1]):
        hist = cv2.calcHist([image_HLS], [i], None, [256], [0, 256])
        hist = hist[hist > 0]
        # print(hist)
        probablity = hist / np.sum(hist)
        print('sum of probability:', np.sum(probablity))
        result.append(-(np.matmul(probablity.T, np.log2(probablity))))
    print(result)
    return result

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
    # image_gray = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
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
    # 统计位于列表中出现特征二元组的次数，同时计算概率 ,参考：https://segmentfault.com/q/1010000016716175?utm_source=tag-newest
    # for i in f_i_j_set:
    #     probability.append(f_i_j.count(i))
    # 根据计算出的概率和熵公式计算熵
    H = -np.matmul(probability, np.log2(probability).T)
    print("熵:", H)
    return H


# 还没有改
def calc_entropy_two_dims_gray_qianghua(image_path, area=3):
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
    # image_gray = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
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
    # 统计位于列表中出现特征二元组的次数，同时计算概率 ,参考：https://segmentfault.com/q/1010000016716175?utm_source=tag-newest
    # for i in f_i_j_set:
    #     probability.append(f_i_j.count(i))
    # 根据计算出的概率和熵公式计算熵
    H = -np.matmul(probability, np.log2(probability).T)
    print("熵:", H)
    return H

if __name__ == '__main__':
    # get_face_image()返回的结果：
    face_image_list = [
        # '/Users/yirufeng/Desktop/科研/数据集/AnimeGAN实验数据集/千与千寻/6_1.jpg',
        # '/Users/yirufeng/Desktop/科研/数据集/AnimeGAN实验数据集/千与千寻/6_2.jpg',
        # '/Users/yirufeng/Desktop/科研/数据集/AnimeGAN实验数据集/千与千寻/18_2.jpg',
        # '/Users/yirufeng/Desktop/科研/数据集/AnimeGAN实验数据集/千与千寻/29_1.jpg',
        # '/Users/yirufeng/Desktop/科研/数据集/AnimeGAN实验数据集/千与千寻/40_2.jpg',
        # '/Users/yirufeng/Desktop/科研/数据集/AnimeGAN实验数据集/千与千寻/54_2.jpg',
        # '/Users/yirufeng/Desktop/科研/数据集/AnimeGAN实验数据集/千与千寻/67_2.jpg',
        # '/Users/yirufeng/Desktop/科研/数据集/AnimeGAN实验数据集/千与千寻/104_2.jpg',
        # '/Users/yirufeng/Desktop/科研/数据集/AnimeGAN实验数据集/千与千寻/108_1.jpg',
        # '/Users/yirufeng/Desktop/科研/数据集/AnimeGAN实验数据集/千与千寻/133_2.jpg',
        # '/Users/yirufeng/Desktop/科研/数据集/AnimeGAN实验数据集/千与千寻/185_2.jpg',
        # '/Users/yirufeng/Desktop/科研/数据集/AnimeGAN实验数据集/千与千寻/239_2.jpg',
        # '/Users/yirufeng/Desktop/科研/数据集/AnimeGAN实验数据集/千与千寻/307_1.jpg',
        # '/Users/yirufeng/Desktop/科研/数据集/AnimeGAN实验数据集/千与千寻/315_1.jpg',
        # '/Users/yirufeng/Desktop/科研/数据集/AnimeGAN实验数据集/千与千寻/327_2.jpg',
        # '/Users/yirufeng/Desktop/科研/数据集/AnimeGAN实验数据集/千与千寻/339_2.jpg',
        # '/Users/yirufeng/Desktop/科研/数据集/AnimeGAN实验数据集/千与千寻/385_1.jpg',
        # '/Users/yirufeng/Desktop/科研/数据集/AnimeGAN实验数据集/千与千寻/419_2.jpg',
        # '/Users/yirufeng/Desktop/科研/数据集/AnimeGAN实验数据集/千与千寻/420_2.jpg',
        # '/Users/yirufeng/Desktop/科研/数据集/AnimeGAN实验数据集/千与千寻/468_2.jpg',
        # '/Users/yirufeng/Desktop/科研/数据集/AnimeGAN实验数据集/千与千寻/513_2.jpg',
        # '/Users/yirufeng/Desktop/科研/数据集/AnimeGAN实验数据集/千与千寻/526_2.jpg',
        # '/Users/yirufeng/Desktop/科研/数据集/AnimeGAN实验数据集/千与千寻/575_2.jpg',
        # '/Users/yirufeng/Desktop/科研/数据集/AnimeGAN实验数据集/千与千寻/592_2.jpg',
        # '/Users/yirufeng/Desktop/科研/数据集/AnimeGAN实验数据集/千与千寻/644_1.jpg',
        # '/Users/yirufeng/Desktop/科研/数据集/AnimeGAN实验数据集/千与千寻/720_2.jpg',
        # '/Users/yirufeng/Desktop/科研/数据集/AnimeGAN实验数据集/千与千寻/722_2.jpg',
        # '/Users/yirufeng/Desktop/科研/数据集/AnimeGAN实验数据集/千与千寻/741_2.jpg',
        # '/Users/yirufeng/Desktop/科研/数据集/AnimeGAN实验数据集/千与千寻/789_1.jpg',
        # '/Users/yirufeng/Desktop/科研/数据集/AnimeGAN实验数据集/千与千寻/821_2.jpg'
    ]
    # entropy_two_dims = []
    #
    # for img in face_image_list:
    #     entropy_temp = calc_entropy_two_dims_gray(img)
    #     print("读取：", img)
    #     entropy_two_dims.append(entropy_temp)
    #     print("熵为：", entropy_temp)
    # plt.plot(entropy_two_dims)
    # plt.show()


    # 计算一个真人的熵
    print(calc_entropy_two_dims_gray("../data/cartoon/0.png"))
    # 概率之和： 1.0
    # 熵: 12.59532037494607
    # 12.59532037494607





