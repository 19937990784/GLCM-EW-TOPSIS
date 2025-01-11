import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import greycomatrix, greycoprops


df = pd.DataFrame(columns=['Contrast Feature', 'Homogeneity Feature',  'Correlation Feature', 'Entropy Feature'])

matrix1 = []
path_of_images = r"D:/python_work/GLCM-EW-TOPSIS/processed-images/6res/"


# 解析文件名中的数字部分
def parse_number(image):
    return int(image.split("-")[1].split(" ")[0])


list_of_images = os.listdir(path_of_images)
list_of_images.sort(key=lambda x: int(x.split(".")[0]))
print(list_of_images)
for image in list_of_images:
    img = cv2.imread(os.path.join(path_of_images, image), cv2.IMREAD_GRAYSCALE)
    equ_gray = cv2.equalizeHist(img)
    bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255])
    inds = np.digitize(equ_gray, bins)
    max_value = inds.max() + 1
    matrix_coocurrence = greycomatrix(inds, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=max_value)
    matrix1.append(matrix_coocurrence)

CF = []
HF = []
COR = []
En = []


for matrix in matrix1:
    def contrast_feature(matrixc):
        contrast = greycoprops(matrixc, prop='contrast')
        return list(contrast)


    def homogeneity_feature(matrixh):
        homogeneity = greycoprops(matrixh, prop='homogeneity')
        return list(homogeneity)


    def correlation_feature(matrixco):
        correlation = greycoprops(matrixco, prop='correlation')
        return list(correlation)


    def entropy_feature(matrix_entropy):
        hist = np.histogram(matrix_entropy, bins=256, range=(0, 255))[0]
        hist = hist / hist.sum()
        entropy = -(hist[hist > 0] * np.log2(hist[hist > 0])).sum()
        return entropy


    CF.append(np.mean(contrast_feature(matrix)))
    HF.append(np.mean(homogeneity_feature(matrix)))
    COR.append(np.mean(correlation_feature(matrix)))
    En.append(entropy_feature(matrix))


Features = [CF, HF, COR, En]

for i, j in zip(df.columns[:8], Features):
    df[i] = j

df.index = [f'Image{i}' for i in range(1, len(list_of_images) + 1)]
df.index.name = 'Image ID'
output_path = 'D:/python_work/GLCM-EW-TOPSIS/features/6Extraction.xlsx'
df.to_excel(output_path)
