import os
import sys
import subprocess


code_dir=sys.path[0]
dir_file=(os.popen("pwd").read())
current_dir=max(dir_file.split('\n'))


def submit_job(directory, opt):

    type_hpc_out=os.popen(f"python {code_dir}/cgat_job_script.py {directory} {opt}").readline().strip("\n").strip().split("###")[-1]
    sub_method=""
    if type_hpc_out=='pbs':
        sub_method="qsub"
    elif type_hpc_out=="slurm":
        sub_method="sbatch"

    os.chdir(directory)
    if os.path.isfile(f"out_train-{opt}.log"):
        subprocess.call(['mv', f"out_train-{opt}.log", f"out_train-{opt}.bufferLog"])
    subprocess.call([sub_method, f"gat-{opt}.pbs"])
    os.chdir(current_dir)  # Return to the initial directory

def check_and_submit(x):
    # Read the NN_batch_state file
    with open('NN_batch_state', 'r') as file:
        lines = file.readlines()

    # Skip the header
    lines = lines[1:]

    for line in lines:
        parts = line.split()
        model_name = parts[0]
        status = parts[1]
        true_r2 = parts[2] if parts[2] != "None" else None
        false_r2 = parts[3] if parts[3] != "None" else None

        if "ERROR" in status and (true_r2 is None or false_r2 is None):
            if true_r2 is None:
                print(f"{model_name}/True")
                if x == 1:
                    submit_job(model_name, 'True')
            if false_r2 is None:
                print(f"{model_name}/False")
                if x == 1:
                    submit_job(model_name, 'False')

    print("Continue Calculation Done!!")


if __name__ == "__main__":
    x = 1  # Set this to 1 to enable job submission, 0 to disable
    check_and_submit(x)

