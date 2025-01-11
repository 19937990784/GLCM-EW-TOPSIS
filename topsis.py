import pandas as pd
import numpy as np

df = pd.read_excel('D:/python_work/GLCM-EW-TOPSIS/total Extraction.xlsx')
columns = ['Contrast Feature', 'Homogeneity Feature', 'Correlation Feature', 'Entropy Feature']
data_original = df[columns]
df = pd.DataFrame(data_original)


def mintomax(datas):
    return np.max(datas) - datas


df['Homogeneity Feature'] = mintomax(df['Homogeneity Feature'])
df['Correlation Feature'] = mintomax(df['Correlation Feature'])
column_sums = df[columns].sum()
normalized_data = df[columns].div(column_sums)
normalized_data.replace(0.000000, 0.002000, inplace=True)
df_normalized_only = normalized_data.copy()
df_normalized_only.columns = [f'Normalized {col}' for col in columns]   # 只包含归一化后的数据的新 DataFrame
# Convert non-numeric values to NaN and then fill NaN with a numeric value
df_numeric = df_normalized_only.apply(pd.to_numeric, errors='coerce').fillna(0)
# 熵
num_rows = len(df_numeric)
column_entropy_values = -np.sum(np.log(df_numeric.iloc[:, :]) * df_numeric.iloc[:, :], axis=0) / np.log(num_rows)
column_entropy_df = pd.DataFrame({'Column': df_numeric.columns, 'Column_Entropy': column_entropy_values})
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# 冗余度与权重
column_redundancy_values = 1 - column_entropy_df['Column_Entropy']
column_weights = column_redundancy_values / column_redundancy_values.sum()
redundancy_weights_df = pd.DataFrame({'Column': df_numeric.columns, 'Redundancy': column_redundancy_values,
                                      'Weight': column_weights})
# Z+/Z-
weighted_data = df_numeric.mul(redundancy_weights_df['Weight'])
max_values = weighted_data.max()
min_values = weighted_data.min()
# 初始化一个列表来保存所有行的最优距离
optimal_distances = []

# 循环计算每一行的最优距离
for i in range(len(df_numeric)):
    optimal_distance_i = np.sqrt(
        (redundancy_weights_df['Weight'][0] * (max_values['Normalized Contrast Feature']
                                               - weighted_data['Normalized Contrast Feature'].iloc[i])**2) +
        (redundancy_weights_df['Weight'][1] * (max_values['Normalized Homogeneity Feature']
                                               - weighted_data['Normalized Homogeneity Feature'].iloc[i])**2) +
        (redundancy_weights_df['Weight'][2] * (max_values['Normalized Correlation Feature']
                                               - weighted_data['Normalized Correlation Feature'].iloc[i])**2) +
        (redundancy_weights_df['Weight'][3] * (max_values['Normalized Entropy Feature']
                                               - weighted_data['Normalized Entropy Feature'].iloc[i])**2))
    optimal_distances.append(optimal_distance_i)

# 初始化一个列表来保存所有行的最劣距离
worst_distances = []

# 循环计算每一行的最劣距离
for i in range(len(df_numeric)):
    worst_distance_i = np.sqrt(
        (redundancy_weights_df['Weight'][0] * (min_values['Normalized Contrast Feature']
                                               - weighted_data['Normalized Contrast Feature'].iloc[i])**2) +
        (redundancy_weights_df['Weight'][1] * (min_values['Normalized Homogeneity Feature']
                                               - weighted_data['Normalized Homogeneity Feature'].iloc[i])**2) +
        (redundancy_weights_df['Weight'][2] * (min_values['Normalized Correlation Feature']
                                               - weighted_data['Normalized Correlation Feature'].iloc[i])**2) +
        (redundancy_weights_df['Weight'][3] * (min_values['Normalized Entropy Feature']
                                               - weighted_data['Normalized Entropy Feature'].iloc[i])**2))
    worst_distances.append(worst_distance_i)

# Creating DataFrames for optimal and worst distances
optimal_distances_df = pd.DataFrame({'Optimal Distance': optimal_distances})
worst_distances_df = pd.DataFrame({'Worst Distance': worst_distances})

# Calculate the ratio for each row
df_ratio = worst_distances_df['Worst Distance'] / (optimal_distances_df['Optimal Distance'] +
                                                   worst_distances_df['Worst Distance'])
# 创建一个新的 DataFrame 包含比率
ratios_df = pd.DataFrame({'Ratio': df_ratio})
output_path = 'D:/python_work/GLCM-EW-TOPSIS/total S.xlsx'
# 导出 DataFrame 到 Excel 文件
ratios_df.to_excel(output_path, index=False)
