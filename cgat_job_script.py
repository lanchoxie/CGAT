import os 
import sys
import argparse

parser = argparse.ArgumentParser(description="This is a scripts for job creation")

parser.add_argument('file_name', type=str, help='Calculated filename')
parser.add_argument('opt', type=str, help='True/False')
parser.add_argument('--check_flat', '-c', action='store_true', help='Check the flat type, do not create jobscript, and return pbs or slurm')

args = parser.parse_args()

#args:

dir_job=args.file_name
opt=args.opt

root_dir = os.path.expandvars('$HOME')
#code_dir="%s/bin/QE-batch"%root_dir
code_dir=sys.path[0]


############job_script_settings##########
###You can modify the FLAT_SAVE file in QE-batch/flat.save/ to manually add default settings of your flats
def read_files(infiles):
    f=open(infiles,"r+")
    f1=f.readlines()
    read_data=[]
    for lines in f1:
        read_data_row=[]
        if "#" in lines:
            continue
        if len(lines) <= 1:
            continue
        if "\t" in lines:
            a_bf=(lines.split("\n")[0]).split("\t")
        else:
            a_bf=(lines.split("\n")[0]).split(" ")
        for i in a_bf:
            if len(i)>0:
                read_data_row.append(i)
        read_data.append(read_data_row)
   # print(len(read_data),len(read_data[0]))
    f.close()
    return read_data

flat_save=read_files("%s/flat.save/FLAT_SAVE"%code_dir)
#flat save has files format like below:
#1  zf_normal  pbs  52
#flat_number  queue_name  type_flat  ppn_num_defaut

flat_info=open("%s/flat.save/FLAT_INFO"%code_dir).readlines()
for lines in flat_info:
    if lines.find("flat_number=")!=-1:
        flat_number=int(lines.split("flat_number=")[-1].split("#")[0])
    if lines.find("node_num=")!=-1:
        node_num=int(lines.split("node_num=")[-1].split("#")[0])
    if lines.find("ppn_num")!=-1:
        ppn_num_man=int(lines.split("ppn_num=")[-1].split("#")[0])
        if ppn_num_man==0:
            ppn_set=0
        elif ppn_num_man!=0:
            ppn_set=1
    if lines.find("wall_time=")!=-1:
        wall_time=lines.split("wall_time=")[-1].split("#")[0].strip("\n")


#flat_number=2           # 1 for"zf_normal",2 for "spst_pub",3 for "dm_pub_cpu"
#node_num=2
#ppn_num=16
#wall_time="116:00:00"
####################"zf_normal" has 52ppn,"spst_pub" has 32 ppn ,"dm_pub_cpu" has 32 ppn

def JOB_func(flat_number,dir_job,opt):
    flat_ind=[int(i[0]) for i in flat_save].index(flat_number)
    type_flat=flat_save[flat_ind][1]
    type_hpc=flat_save[flat_ind][2]
    ppn_num=int(flat_save[flat_ind][3])
    if ppn_set==1:
        ppn_num=ppn_num_man
    ppn_tot=node_num*ppn_num

    #print(type_flat,node_num,ppn_num,wall_time)
    if type_hpc=="pbs":
        jobscript_file_in=['#!/bin/bash\n',
       f'#PBS  -N   GAT-train-{opt}\n',
        '#PBS  -l   nodes=%d:ppn=%d\n'%(node_num,ppn_num),
        '#PBS  -l   walltime=%s\n'%wall_time,
        '#PBS  -S   /bin/bash\n',
        '#PBS  -j   oe\n', 
        '#PBS  -q   %s\n'%(type_flat),
        '\n',
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
    if type_hpc=="slurm":
        jobscript_file_in=[
        "#!/bin/bash\n",
       f"#SBATCH --job-name=GAT-train-{opt}\n",
        "#SBATCH -D ./\n",
        "#SBATCH --nodes=%d\n"%node_num,
        "#SBATCH --ntasks-per-node=%d\n"%ppn_num,
        "#SBATCH -o output.%j\n",
        "##SBATCH -e error.%j\n",
        "#SBATCH --time=%s\n"%wall_time,
        "#SBATCH --partition=%s\n"%(type_flat),
        "\n",
        "##SBATCH --gres=gpu:4 #if use gpu, uncomment this\n",
        "#export I_MPI_PMI_LIBRARY=/opt/gridview/slurm/lib/libpmi.so\n",
        "ulimit -s unlimited\n",
        "ulimit -l unlimited\n",
        "\n",
        "#setup intel oneapi environment \n",
        "source /dm_data/apps/intel/oneapi/setvars.sh\n",
        "#source /etc/profile\n",
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
    return jobscript_file_in,type_hpc

def JOB_modify(dir_job,job_cont,opt):

    f=open(f"{dir_job}/gat-{opt}.pbs","w+")
    for i in job_cont:
        f.writelines(i)
    f.close()

job_cont,type_hpc_out=JOB_func(flat_number,dir_job,opt)
JOB=f"{dir_job}/gat-{opt}.pbs"
if os.path.isfile(JOB)==1:                                      #job file modify
    os.system("rm %s"%JOB)
os.mknod(JOB)
print(f"###{type_hpc_out}")
if not args.check_flat:
    JOB_modify(dir_job,job_cont,opt)
    print(f"{JOB} created!")
