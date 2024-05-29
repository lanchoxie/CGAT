import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import re

code_dir=sys.path[0]
y_down_lim=0.6   #y axis bottom limit

# Load the data
df = pd.read_csv("NN_batch_state", delimiter='\t', header=0)

# Replace 'None' strings with actual None (Python's null value)
df.replace('None', np.nan, inplace=True)

# Fill NA/NaN values with 0 for plotting
df['True_R2'] = df['True_R2'].fillna(0).astype(float)
df['False_R2'] = df['False_R2'].fillna(0).astype(float)

true_r2_labels = df['True_R2'].fillna('None')
false_r2_labels = df['False_R2'].fillna('None')

# Extract seed number using regex and remove it from model names
seed_number = re.search(r'seed_(\d+)', df['Model_name'][0]).group(1)
df['Model_name'] = df['Model_name'].apply(lambda x: re.sub(r'seed_\d+_', '', x))

# Bar plot configuration
x = np.arange(len(df['Model_name']))  # label locations
width = 0.42  # width of the bars

# Plot
fig, ax = plt.subplots(figsize=(14, 7))
true_bars = ax.bar(x - width/2, df['True_R2'], width, label='True R2', color='lightpink')
false_bars = ax.bar(x + width/2, df['False_R2'], width, label='False R2', color='lightblue')


# Label the bars with the R2 value
'''
for bar, value in zip(true_bars, df['True_R2']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
            'None' if value == 0 else f'{value:.3f}',
            ha='center', va='bottom', rotation=0, fontsize=5)
            #ha='center', va='bottom', rotation='vertical', fontsize=8)


for bar, value in zip(false_bars, df['False_R2']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
            'None' if value == 0 else f'{value:.3f}',
            ha='center', va='bottom', rotation=0, fontsize=5)
            #ha='center', va='bottom', rotation='vertical', fontsize=8)
'''

# Label the bars with the R2 value or 'None'
for bar, label in zip(true_bars, true_r2_labels):
    height = bar.get_height() if bar.get_height() != 0 else y_down_lim  # Adjust base height for zero values
    ax.text(bar.get_x() + bar.get_width() / 2, height,
            "None" if label == 0 else f'{float(label):.3f}',
            ha='center', va='bottom', rotation=0, fontsize=5)

for bar, label in zip(false_bars, false_r2_labels):
    height = bar.get_height() if bar.get_height() != 0 else y_down_lim  # Adjust base height for zero values
    ax.text(bar.get_x() + bar.get_width() / 2, height,
            "None" if label == 0 else f'{float(label):.3f}',
            ha='center', va='bottom', rotation=0, fontsize=5)

# Highlighting the best bars with a border
best_true_bar = true_bars[np.argmax(df['True_R2'])]
best_false_bar = false_bars[np.argmax(df['False_R2'])]
best_true_bar.set_edgecolor('red')
best_true_bar.set_linewidth(2)
best_false_bar.set_edgecolor('blue')
best_false_bar.set_linewidth(2)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('R2 Value')
ax.set_ylim(y_down_lim,1)
ax.set_xlabel('Model Parameters')
ax.set_title(f'R2 Comparison for Seed {seed_number}')
ax.set_xticks(x)
ax.set_xticklabels(df['Model_name'], rotation=45, ha="right")
ax.legend()
#Adjust the plot edge range
plt.subplots_adjust(top=0.85, bottom=0.15, left=0.10, right=0.95)
#plt.tight_layout()
fig_col_name=f"column_plot_seed_{seed_number}_14_7.png"
plt.savefig(fig_col_name,dpi=1200)
print(f"column_plot_seed_{seed_number}_14_7.png Created!")

#Box plot
# Prepare data for plotting
data_to_plot = [df['True_R2'].dropna().astype(float), df['False_R2'].dropna().astype(float)]
# Create a figure and an axes.
fig, ax = plt.subplots()
# Boxplot
# Remove zeros and NaNs for mean calculation
filtered_true_r2 = df['True_R2'][df['True_R2'] != 0].dropna()
filtered_false_r2 = df['False_R2'][df['False_R2'] != 0].dropna()

# Prepare data for plotting
data_to_plot = [filtered_true_r2, filtered_false_r2]

# Create a figure and an axes.
fig, ax = plt.subplots()

# Boxplot
bp = ax.boxplot(data_to_plot, patch_artist=True, notch=True, meanline=False, showmeans=False,
                medianprops={"linestyle":"-", "color":"black"})

# Customizing the box colors
colors = ['lightpink', 'lightblue']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# Adding mean values as diamond markers
mean_true_r2 = np.mean(filtered_true_r2)
mean_false_r2 = np.mean(filtered_false_r2)
print(mean_true_r2)
print(mean_false_r2)
ax.scatter([1], [mean_true_r2], color='gold', marker='D', s=60, zorder=3, label='Mean')
ax.scatter([2], [mean_false_r2], color='gold', marker='D', s=60, zorder=3)

# Adding labels for the X-axis
ax.set_xticklabels(['True R2', 'False R2'])

# Adding titles and labels
ax.set_title(f'Distribution of True R2 and False R2 Values: seed {seed_number}')
ax.set_ylabel('R2 Values')
# Add legend for mean
ax.legend()


fig_box_name=f"box_plot_seed_{seed_number}_8_8.png"
plt.savefig(fig_box_name,dpi=1200)
print(f"box_plot_seed_{seed_number}_8_8.png Created!")

import subprocess as sb
sb.Popen(["python",f"{code_dir}/show_fig.py",fig_col_name])
sb.Popen(["python",f"{code_dir}/show_fig.py",fig_box_name])
