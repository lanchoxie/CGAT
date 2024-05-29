import os
import re
import subprocess
import sys

code_dir = sys.path[0]

def parse_job_output(output):
    lines = output.decode().split('\n')
    running_jobs = []
    waiting_jobs = []
    current_list = None
    
    for line in lines:
        if "Running job(s):" in line:
            current_list = running_jobs
        elif "Waiting job(s):" in line:
            current_list = waiting_jobs
        elif line.strip().isdigit():
            if current_list is not None:
                current_list.append(line.strip())
    
    return running_jobs, waiting_jobs

def check_job_status(directory):
    process = subprocess.Popen(['python', f'{code_dir}/job_on_hpc_id.py', directory], stdout=subprocess.PIPE)
    output, _ = process.communicate()
    running_jobs, waiting_jobs = parse_job_output(output)
    logs = [f for f in os.listdir(directory) if f.endswith('.log')]
    done_status = []
    for log in logs:
        with open(os.path.join(directory, log), 'r') as file:
            lines = file.readlines()
            done_status.append("CGAT DONE" in lines[-1] if lines else False)

    def count_trues(input_list):
        # Using the sum function to count True values since True is treated as 1
        return sum(1 for item in input_list if item is True)

    job_count = len(running_jobs) + len(waiting_jobs)
    final_status = "UNKNOWN"
    if job_count == 2:
        final_status = "ALL_RUN " if len(running_jobs) == 2 else "ALL_WAIT" if len(waiting_jobs) == 2 else "RUN+WAIT"
    elif job_count == 1:
        if running_jobs:
            final_status = "DONE+RUN" if count_trues(done_status)==1 else "ERROR+RUN"
        else:
            final_status = "DONE+WAIT" if count_trues(done_status)==1 else "ERROR+WAIT"
    else:
        if done_status:
            final_status = "ALL_DONE" if all(done_status) else "ERROR+DONE" if count_trues(done_status)==1 else "ALL_ERROR"
        else:
            final_status = 'ALL_ERROR'
    return final_status, done_status

def extract_r2_and_update_status(subdir, status):
    true_r2 = None
    false_r2 = None
    if "DONE" in status:
        for d in os.listdir(subdir):
            sub_path = os.path.join(subdir, d)
            if os.path.isdir(sub_path) and re.match(r"\d+\.\d+.*", d):
                r2_match = re.match(r"(\d+\.\d+)_.*opt_(True|False).*", d)
                if r2_match:
                    r2_value = r2_match.group(1)
                    opt_value = r2_match.group(2)
                    if opt_value == "True":
                        true_r2 = r2_value
                    else:
                        false_r2 = r2_value
    return true_r2, false_r2

# 提取k和h的值用于排序，返回一个元组，先按k降序，再按h降序
def extract_k_h_values(path):
    dirname = os.path.basename(path)  # 获取路径的最后一部分
    matches = re.match(r'seed_\d+_k_(\d+)_h_(\d+)', dirname)
    if matches:
        k_value = int(matches.group(1))
        h_value = int(matches.group(2))
        # 返回负值进行降序排序
        return (k_value, h_value)
    else:
        return (float('inf'), float('inf'))  # 如果没有匹配，放到列表最后


def main():
    current_dir = os.getcwd()
    subdirs = [os.path.join(current_dir, d) for d in os.listdir(current_dir) if os.path.isdir(d)]
    subdirs.sort(key=extract_k_h_values)
    with open("NN_batch_state", "w+") as f:
        f.write("Model_name\tstatus\tTrue_R2\tFalse_R2\n")
        print("Model_name\tstatus\tTrue_R2\tFalse_R2")
        for subdir in subdirs:
            subdir_name = subdir.split("/")[-1]
            status, done_status = check_job_status(subdir)
            true_r2, false_r2 = extract_r2_and_update_status(subdir, status) if any(done_status) else (None, None)
            f.write(f"{subdir_name}\t{status}\t{true_r2}\t{false_r2}\n")
            print(f"{subdir_name}\t{status}\t{true_r2}\t{false_r2}")
    print("NN_batch_state Generated!")

if __name__ == "__main__":
    main()

