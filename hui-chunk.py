import cv2
import numpy as np
import os

pic_path = 'D:/python_work/GLCM-EW-TOPSIS/image/3.jpg'  # 分割的图片的位置
pic_target = 'D:/python_work/GLCM-EW-TOPSIS/processed-images/3res/'  # 分割后的图片保存的文件夹
if not os.path.exists(pic_target):  # 判断是否存在文件夹如果不存在则创建为文件夹
    os.makedirs(pic_target)
# 要分割后的尺寸
cut_width = 100
cut_length = 300
# 读取要分割的图片，以及其尺寸等数据
picture = cv2.imread(pic_path)
gray_picture = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
adjusted_gray = cv2.addWeighted(gray_picture, 1.0, np.zeros_like(gray_picture), 0, 0)
(width, length,) = gray_picture.shape
# 预处理生成0矩阵
pic = np.zeros((cut_width, cut_length,))
# 计算可以划分的横纵的个数
num_width = int(width / cut_width)
num_length = int(length / cut_length)
# for循环迭代生成
for i in range(0, num_width):
    for j in range(0, num_length):
        pic = adjusted_gray[i * cut_width: (i + 1) * cut_width, j * cut_length: (j + 1) * cut_length, ]
        result_path = pic_target + '{}_{}.jpg'.format(i + 1, j + 1)
        cv2.imwrite(result_path, pic)

print("done!!!")
