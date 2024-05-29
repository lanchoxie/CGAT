import pandas as pd
import matplotlib.pyplot as plt
import subprocess as sb
import seaborn as sns
import scipy.stats as stats
import sys

# Load the data from the .data file
code_dir = sys.path[0]
file_path = 'gnn_data.save/Calculated_result_info.data'  # Please adjust the file path as needed
data = pd.read_csv(file_path, delimiter='\t')

# Descriptive statistics
delta_E_stats = data['delta_E'].describe()

# Shapiro-Wilk test
shapiro_test = stats.shapiro(data['delta_E'])
shapiro_stat = shapiro_test.statistic
shapiro_p_value = shapiro_test.pvalue

# Standardize the delta_E column
mean_value = data['delta_E'].mean()
median_value = data['delta_E'].median()
std_dev = data['delta_E'].std()
data['delta_E_standardized'] = (data['delta_E'] - mean_value) / std_dev

# Set up the matplotlib figure
plt.figure(figsize=(18, 6))

# Histogram
plt.subplot(1, 3, 1)
sns.histplot(data['delta_E'], bins=30, kde=True)
plt.title('Distribution of delta_E')
plt.xlabel('delta_E (eV)')
plt.ylabel('Frequency')

# Box plot with statistics in the center
plt.subplot(1, 3, 2)
sns.boxplot(x=data['delta_E'])
plt.title('Box plot of delta_E')
plt.xlabel('delta_E (eV)')
# Calculate and plot the mean as a golden diamond
plt.scatter(mean_value, 0, color='gold', marker='D', s=100, label='Mean', zorder=5)

# Adjust the box plot position
ax = plt.gca()
box = ax.get_position()
ax.set_position([box.x0, box.y0 + 0.2, box.width, box.height * 0.8])

# Add text for descriptive statistics below the box plot
stats_text = (f'Count: {int(delta_E_stats["count"])}\n'
              f'Mean: {delta_E_stats["mean"]:.4f}eV\n'
              f'Standard Deviation: {delta_E_stats["std"]:.4f}eV\n'
              f'Minimum: {delta_E_stats["min"]:.4f}eV\n'
              f'25th Percentile (Q1): {delta_E_stats["25%"]:.4f}eV\n'
              f'Median (Q2): {delta_E_stats["50%"]:.4f}eV\n'
              f'75th Percentile (Q3): {delta_E_stats["75%"]:.4f}eV\n'
              f'Maximum: {delta_E_stats["max"]:.4f}eV')
#plt.text(1.05, 0.5, stats_text, transform=plt.gca().transAxes,
#         horizontalalignment='left', verticalalignment='center', fontsize=10,
#         bbox=dict(facecolor='white', alpha=0.5))

plt.scatter(mean_value, 1, color='black', marker='D', s=0.1, label=f'{stats_text}', zorder=5)
# Show legend for mean in the upper right corner
plt.legend(loc='lower center')

# Q-Q plot
plt.subplot(1, 3, 3)
stats.probplot(data['delta_E'], dist="norm", plot=plt)
plt.title('Q-Q plot of delta_E')
plt.xlabel('delta_E (eV)')

# Add Shapiro-Wilk test result to Q-Q plot legend
#shapiro_text = f'Shapiro-Wilk Test:\nStatistic: {shapiro_stat:.4f}\nP-value: {shapiro_p_value:.4f}\n\nInterpretation:\nP-value > 0.05, cannot reject\nnormality hypothesis'
shapiro_text = f'Shapiro-Wilk Test:\nStatistic: {shapiro_stat:.4f}\nP-value: {shapiro_p_value:.4f}'
plt.gca().text(0.05, 0.95, shapiro_text, transform=plt.gca().transAxes,
               verticalalignment='top', fontsize=10,
               bbox=dict(facecolor='white', alpha=0.5))

print("#"*60)
print(stats_text)
print("#"*60)
print(shapiro_text)
print("#"*60)

# Show the plots
plt.tight_layout()
plt.savefig("dataset_analyze_result_18_6.png", dpi=900)
# Show the plots
#sb.Popen(["python",f"{code_dir}/show_fig.py",f"dataset_analyze_result_18_6.png"])
print("dataset_analyze_result_18_6.png has been saved!")
