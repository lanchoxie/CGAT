import os
import sys
import argparse
import subprocess

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description="GNN Train Code")
# 添加参数
parser.add_argument('model_example', type=str, help='The mode file')
parser.add_argument('database', type=str, help='The mode training database')
parser.add_argument('seed', type=int, help='seed for initialize neural network')
#parser.add_argument('--k_att', '-k', choices=[2,3,4,5], type=int, help='k_hop_attention, the k-th neighbor calculated')
#parser.add_argument('--gat_heads', '-h', type=int, help='GAT heads number')

# 解析命令行参数
args = parser.parse_args()

model=args.model_example
model_name=model.split("/")[-1].strip().split(".py")[0] if "/" in model else model.strip().split(".py")[0]
seed = args.seed   # 选择一个种子
if not os.path.exists(f"SEED_{seed}"):
    os.makedirs(f"SEED_{seed}", exist_ok=True)
#k_hop_attention=args.k_att  # Attention额外添加边的邻居
#gat_heads=args.gat_heads    # Multi-head-attention number
k_att_list=[2,3,4,5]  # Attention额外添加边的邻居
gat_heads_list=[4,6,8,12,16,20]    # Multi-head-attention number
opt_dis_list=['True','False']    # True for optimized str train(DFT related)

sub_mode=1

dir_file=(os.popen("pwd").read())
current_dir=max(dir_file.split('\n'))
db_name=args.database.split("/")[-1].strip() if "/" in args.database else args.database.strip()
db_dir=current_dir+"/"+args.database.split("db_name")[0] if "/" in args.database else current_dir+"/"

code_dir=sys.path[0]
print(f"Model example:{model}")
print(f"Database dir:{db_dir}")

def modify_flat(dir_, str_name, parameter, value, after_line, before_content):
    #***************************************#
    #输入的分别是：目标目录，目标文件，参数，参数修改值，参数（如果不存在）添加在哪个句子后面，参数前面需要添加什么
    #***************************************#

    input_file = f'{dir_}/{str_name}'  # 输入文件名
    #print("*********",input_file,dir_)
    with open(input_file, 'r') as file:
        lines = file.readlines()

    parameter_line = None  # 初始化参数行为None
    parameter_found = False  # 标志参数是否已找到

    note=''

    for i, line in enumerate(lines):
        if line.strip().startswith(after_line):  # 查找指定行
            parameter_line = i  # 记录参数应该插入或修改的位置
        if line.strip().startswith(parameter):  # 检查参数是否存在
            parameter_found = True
            parameter_line = i  # 更新参数行为当前行
            if "#" in line:
                note="   #"+line.split("#")[-1].strip("\n")
            break  # 退出循环，因为已找到参数

    formatted_parameter = f'{before_content}{parameter}={value}'  # 格式化参数
    if note!='':
        formatted_parameter+=note    

    if parameter_found:
        lines[parameter_line] = formatted_parameter + '\n'  # 修改存在的参数
    else:
        if parameter_line is not None:
            lines.insert(parameter_line + 1, formatted_parameter + '\n')  # 插入新参数
        else:
            lines.append(formatted_parameter + '\n')  # 或者在文件末尾添加新参数

    with open(input_file, 'w') as file:
        file.writelines(lines)  # 写入输出文件


def job_create(dir_job,opt):
    
    job_cont=[
    '#!/bin/bash\n',
    f'#PBS  -N   GAT-train-{opt}\n',
    '#PBS  -l   nodes=1:ppn=52\n',
    '#PBS  -l   walltime="01:30:00"\n',
    '#PBS  -S   /bin/bash\n',
    '#PBS  -j   oe\n',
    '#PBS  -q   zf_normal\n',
    ' \n',
    'cd $PBS_O_WORKDIR\n',
    ' \n',
    f'exec 2> ERROR-{opt}\n',
    ' \n',
    'module load tools/conda/anaconda.2023.09\n',
    'bash\n',
    'module unload apps/python/3.7.1\n',
    'source activate gat_env\n',
    f'exec > out_train-{opt}.log 2>&1\n',
    f'python train_gat_coords_edge_3-{opt}.py\n',]

    f=open(f"{dir_job}/gat-{opt}.pbs","w+")
    for i in job_cont:
        f.writelines(i)
    f.close()

def submit_job(directory, opt):
    type_hpc_out=os.popen(f"python {code_dir}/cgat_job_script.py {directory} {opt}").readline().strip("\n").strip().split("###")[-1]
    sub_method=""
    if type_hpc_out=='pbs':
        sub_method="qsub"
    elif type_hpc_out=="slurm":
        sub_method="sbatch"
    if sub_mode:
        os.chdir(directory)
        if os.path.isfile(f"out_train-{opt}.log"):
            subprocess.call(['mv', f"out_train-{opt}.log", f"out_train-{opt}.bufferLog"])
        subprocess.call([sub_method, f"gat-{opt}.pbs"])
        os.chdir(current_dir)  # Return to the initial directory
#make dirs:
#k_att_list=[2,3,4,5]  # Attention额外添加边的邻居
#gat_heads_list=[4,6,8,12,16,20]    # Multi-head-attention number
#opt_dis_list=['True','False']    # True for optimized str train(DFT related)
for k in k_att_list:
    for h in gat_heads_list:
        dir_make=f"SEED_{seed}/seed_{seed}_k_{k}_h_{h}"
        os.makedirs(f"{dir_make}",exist_ok=True)
        for opt in opt_dis_list:
            os.system(f"cp {model} {dir_make}/{model_name}-{opt}.py")
            #job_create(dir_make,opt) 
            modify_flat(dir_make,f"{model_name}-{opt}.py", "seed", seed, "#Construction related parameters", '    ') 
            modify_flat(dir_make,f"{model_name}-{opt}.py", "k_hop_attention", k, "#Construction related parameters", '    ') 
            modify_flat(dir_make,f"{model_name}-{opt}.py", "gat_heads", h, "#Construction related parameters", '    ') 
            modify_flat(dir_make,f"{model_name}-{opt}.py", "opt_dis", opt, "#Construction related parameters", '    ') 
            modify_flat(dir_make,f"{model_name}-{opt}.py", "db_dir", f"\'{db_dir}\'", "#Construction related parameters", '    ') 
            #if sub_mode:
            #    os.chdir(dir_make)
            #    os.system(f"qsub gat-{opt}.pbs")
            #    os.chdir(current_dir)
            submit_job(dir_make,opt)
print(f"katt:{k_att_list},head:{gat_heads_list},opt:{opt_dis_list}")
print(f"Total {len(k_att_list)*len(gat_heads_list)*len(opt_dis_list)} CGAT Models created!")
