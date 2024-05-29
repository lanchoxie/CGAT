import os
import sys
import re
import argparse
import numpy as np
import subprocess as sb
import matplotlib.pyplot as plt

code_dir = sys.path[0]
xsf_dir = 'Doping_out'
model = 'model_GAT_train-valence-Resnet-aggregate_k4-FC3_4_heads_12_seed_414_R2_0.814_opt_False.pth'

if not os.path.exists(xsf_dir):
    os.system("python multi-doping.py")
predict_dir = os.path.join(code_dir, xsf_dir)

prefix = 'LiNiO2-331_NCMT_3132_'
predict_prefix = 'prediction_results_of_'

def calculate_total_mixing_ratio(predict_file_done):
    predict_file_result = []
    for subdir in predict_file_done:
        timelog_lines = ''.join(open(os.path.join(predict_dir, subdir, 'timelog.log')).readlines())
        pattern = r"Mixing number:(\d+)\s*Total count:(\d+)\s*Mixing ratio:(\d+\.\d+)%"
        match = re.search(pattern, timelog_lines)
        if match:
            mixing_number = int(match.group(1))
            total_count = int(match.group(2))
            mixing_ratio = float(match.group(3))
            predict_file_result.append([mixing_number, total_count, mixing_ratio])
    
    if len(predict_file_result) > 0:
        total_mixing_ratio = sum([i[0] for i in predict_file_result]) / sum([i[1] for i in predict_file_result])
        return total_mixing_ratio, predict_file_result
    return None, None

def plot_total_ratio_curve(predict_file_result, prefix, resolution):
    ratios = []
    for i in range(1, len(predict_file_result) + 1):
        partial_result = predict_file_result[:i]
        total_mixing_ratio = sum([x[0] for x in partial_result]) / sum([x[1] for x in partial_result])
        ratios.append(total_mixing_ratio)
    
    fig, ax = plt.subplots()
    ax.plot(range(len(ratios)), ratios, label='Total Ratio')
    
    # If nearing convergence, plot with a red dashed line
    if len(ratios) > 1 and abs(ratios[-1] - ratios[-2]) < 0.001:
        ax.plot(range(len(ratios)), ratios, 'r--', label='Converging')
    
    ax.set_xlabel('Index')
    ax.set_ylabel('Total Ratio')
    ax.legend()
    
    # Extract the number after 'NCMT' in the prefix
    ncmt_number = prefix.split('NCMT_')[1].split('_')[0]
    output_filename = f"total_ratio_{ncmt_number}_{int(resolution[0]/100)}_{int(resolution[1]/100)}.png"
    
    fig.set_size_inches(resolution[0]/100, resolution[1]/100)
    plt.savefig(output_filename)
    plt.close()
    
    # Display the figure using subprocess
    sb.Popen(["python", f"{code_dir}/show_fig.py", output_filename])

def plot_total_ratio(predict_file_result, prefix, resolution):
    ratios = []
    # Extract the number after 'NCMT' in the prefix
    ncmt_number = prefix.split('NCMT_')[1].split('_')[0]

    for i in range(1, len(predict_file_result) + 1):
        partial_result = predict_file_result[:i]
        total_mixing_ratio = sum([x[0] for x in partial_result]) / sum([x[1] for x in partial_result])
        ratios.append(total_mixing_ratio)
    
    fig, ax = plt.subplots()
    ax.plot(range(len(ratios)), ratios, label='Total Ratio')
    
    # 绘制最后一个值的水平虚线
    if ratios:
        last_value = ratios[-1]
        ax.axhline(y=last_value, color='r', linestyle='--', label=f'Converging Ratio')
   
    ax.set_title(f'NCMT:{ncmt_number} Approximate Ratio:{float(ratios[-1])*100:.3f}%')
    ax.set_xlabel('Index')
    ax.set_ylabel('Total Ratio')
    ax.legend()
    
    output_filename = f"total_ratio_{ncmt_number}_{int(resolution[0]/100)}_{int(resolution[1]/100)}.png"
      
    fig.set_size_inches(resolution[0]/100, resolution[1]/100)
    plt.savefig(output_filename)
    plt.close()
    # Display the figure using subprocess
    sb.Popen(["python", f"{code_dir}/show_fig.py", output_filename])

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-f', '--file', action='store_true', help='If set, plot the total ratio instead of running calculations')
    parser.add_argument('-r', '--resolution', type=str, default='800x600', help='Resolution for the output plot, e.g., 800x600')
    args = parser.parse_args()

    # Parse the resolution
    resolution = [int(x) for x in args.resolution.split('x')]

    if args.file:
        predict_file_done = [
            d for d in os.listdir(predict_dir) 
            if predict_prefix + prefix in d 
            and 'timelog.log' in os.listdir(os.path.join(predict_dir, d)) 
            and 'ratio' in os.path.join(predict_dir, d, 'timelog.log')
        ]
        
        _, predict_file_result = calculate_total_mixing_ratio(predict_file_done)
        plot_total_ratio(predict_file_result, prefix, resolution)
        return
    
    previous_mixing_ratio = None
    convergence_threshold = 0.0001
    influence_threshold = 0.0001
    max_iterations = 100
    iteration = 0

    while iteration < max_iterations:
        predict_file = [f for f in os.listdir(predict_dir) if prefix in f]
        predict_file_done = [
            d for d in os.listdir(predict_dir) 
            if predict_prefix + prefix in d 
            and 'timelog.log' in os.listdir(os.path.join(predict_dir, d)) 
            and 'ratio' in os.path.join(predict_dir, d, 'timelog.log')
        ]
        
        total_mixing_ratio, predict_file_result = calculate_total_mixing_ratio(predict_file_done)
        
        if total_mixing_ratio is not None:
            print(f"Iteration {iteration}: Total Mixing Ratio = {total_mixing_ratio}")
            
            if previous_mixing_ratio is not None:
                # 判断是否收敛
                if abs(total_mixing_ratio - previous_mixing_ratio) < convergence_threshold:
                    print("Convergence reached!")
                    break
                
                # 计算每次增加一个值对总 mixing ratio 的影响
                influences = []
                for i in predict_file_result:
                    temp_result = predict_file_result[:]
                    temp_result.remove(i)
                    temp_mixing_ratio = sum([x[0] for x in temp_result]) / sum([x[1] for x in temp_result])
                    influence = abs(total_mixing_ratio - temp_mixing_ratio)
                    influences.append(influence)
                
                max_influence = max(influences)
                if max_influence < influence_threshold:
                    print("The influence of adding more structures is minimal. Stopping calculations.")
                    break
            
            previous_mixing_ratio = total_mixing_ratio
        
        not_done_f = [
            f for f in predict_file 
            if predict_prefix + f.split(".xsf")[0] not in predict_file_done 
            and '.xsf' in f
        ]

        # Call multi-doping.py to create 8 more structures
        if not not_done_f:
            new_files_line=[line for line in open("multi-doping.py").readlines() if "sample_number" in line and '=' in line][0]
            new_file_number=new_files_line.split("=")[-1].strip().split(' ')[0]
            print(f"Creating {new_file_number} more structures...")
            os.system("python multi-doping.py")

            # 重新读取预测目录中的文件列表，以便包含新生成的结构
            predict_file = [f for f in os.listdir(predict_dir) if prefix in f]
            predict_file_done = [
                d for d in os.listdir(predict_dir) 
                if predict_prefix + prefix in d 
                and 'timelog.log' in os.listdir(os.path.join(predict_dir, d)) 
                and 'ratio' in os.path.join(predict_dir, d, 'timelog.log')
            ]
            
            not_done_f = [
                f for f in predict_file 
                if predict_prefix + f.split(".xsf")[0] not in predict_file_done 
                and '.xsf' in f
            ]
        
        for str_name in not_done_f:
            os.chdir(xsf_dir)
            print(f"Predicting on {str_name}")
            jj = os.popen(f"python ../predict_from_scratch_data.py ../{model} {str_name}").readlines()
            print([i for i in jj if 'Mixing number' in i])
            os.chdir(code_dir)
        
        iteration += 1

    if iteration == max_iterations:
        print("Max iterations reached without convergence.")

if __name__ == "__main__":
    main()

