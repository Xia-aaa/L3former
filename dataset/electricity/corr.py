import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# 读取 CSV 文件
df = pd.read_csv('electricity.csv')

# 选择需要的列
selected_columns = df.columns[1:]

# 选择数据
selected_data = df[selected_columns]
# start_row = 18317+2633  # 例如，从第101行开始
start_row = 0
end_row = 18317   # 例如，到第1100行结束
selected_data = selected_data.iloc[18317:]
# selected_data = selected_data.iloc[0:18317]

# 计算相关矩阵
correlation_matrix = selected_data.corr()
correlation_matrix = np.abs(correlation_matrix)

# 查看相关矩阵
print(correlation_matrix)

# # 创建一个自定义的颜色映射，从紫色到黄色
# colors = [(0.5, 0, 0.5), (1, 1, 0)]  # 紫色到黄色
# cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

# 创建一个图形
plt.figure(figsize=(6, 5))
sns.heatmap(
    correlation_matrix,
    cmap='viridis',
    square=True,
    linewidths=0,
    linecolor='white',
    cbar_kws={"shrink": 0.8},
    xticklabels=False,
    yticklabels=False,
    cbar=False,
    vmin=0,  # 修改点2：传递数值范围
    vmax=1
)

# 设置标题
plt.tight_layout()
# 显示图形
plt.show()