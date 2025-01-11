import os
import pandas as pd


# 定义函数来计算每列的总和
def calculate_column_sums(file_path):
    # 读取.xlsx文件
    df = pd.read_excel(file_path)
    # 计算第二列到第五列的总和
    column_sums = df.iloc[:, 1:5].sum().tolist()
    return column_sums


# 文件夹路径
folder_path = "D:/python_work/GLCM-EW-TOPSIS/features/"
# 初始化一个空的DataFrame来存储所有文件的计算结果，并定义列名
result_df = pd.DataFrame(columns=['Filename', 'Column1', 'Column2', 'Column3', 'Column4'])
# 获取文件夹中所有以'.xlsx'结尾的文件，并按文件名排序
xlsx_files = sorted([filename for filename in os.listdir(folder_path) if filename.endswith('.xlsx')])

# 遍历文件夹中的每个.xlsx文件
for filename in xlsx_files:
    file_path = os.path.join(folder_path, filename)
    # 计算每个文件的列总和
    column_sums = calculate_column_sums(file_path)
    # 将文件名与列总和相关联，并添加到DataFrame中
    result_df = result_df.append({'Filename': filename, 'Column1': column_sums[0], 'Column2': column_sums[1],
                                  'Column3': column_sums[2], 'Column4': column_sums[3]}, ignore_index=True)

# 将结果保存到新的.xlsx文件中
result_file_path = "total Extraction.xlsx"
result_df.to_excel(result_file_path, index=False)  # 不保存索引

print("Calculation completed and result saved to:", result_file_path)
