import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置字体属性和全局变量
font_path = 'C:/Windows/Fonts/times.ttf'  # 请确保路径正确
font_properties = FontProperties(fname=font_path)
font_size = 16

# 确保所有文本使用相同的字体和大小
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = font_size

# Data
data = {
    "Number of Atoms": [108, 216, 432, 864],
    "CGAT Prediction (s)": [26, 55, 143, 420],
    "DFT Calculation (days)": [53.37, 106.74, 213.48, 426.96]
}

df = pd.DataFrame(data)

# Create figure and dual axis
fig, ax1 = plt.subplots(figsize=(12, 8))

color = 'tab:blue'
ax1.set_xlabel('Number of Li-Ni pair/Cell', fontsize=font_size, fontproperties=font_properties)
ax1.set_ylabel('CGAT Prediction (seconds)', color=color, fontsize=font_size, fontproperties=font_properties)
bar1 = ax1.bar([x - 0.175 for x in range(len(df["Number of Atoms"]))], df["CGAT Prediction (s)"], color=color, width=0.35, align='center', label='CGAT Prediction (seconds)')
ax1.tick_params(axis='y', labelcolor=color)
ax1.tick_params(axis='x', labelsize=font_size)
ax1.set_xticks(range(len(df["Number of Atoms"])))
ax1.set_xticklabels(df["Number of Atoms"], fontsize=font_size, fontproperties=font_properties)

# Create second axis
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('DFT Calculation (days)', color=color, fontsize=font_size, fontproperties=font_properties)
bar2 = ax2.bar([x + 0.175 for x in range(len(df["Number of Atoms"]))], df["DFT Calculation (days)"], color=color, width=0.35, align='center', label='DFT Calculation (days)')
ax2.tick_params(axis='y', labelcolor=color)

# Add data labels
def add_labels(bars, ax):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=font_size, fontproperties=font_properties)

add_labels(bar1, ax1)
add_labels(bar2, ax2)

# Add title
plt.title('CGAT Prediction vs. DFT Calculation Time Comparison', fontsize=(font_size+2), fontproperties=font_properties)

# Show legend
#fig.tight_layout()
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9), bbox_transform=ax1.transAxes, fontsize=font_size, prop={'family': 'Times New Roman'})

# Save image
plt.savefig('timecost_compare_dual_axis.png', dpi=1200)
plt.show()
