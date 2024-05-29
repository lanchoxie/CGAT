1.batch_gnn_train.py {model_example} {dataset_directory} {seed_num}    20240520
会创建SEED_{seed_num}目录然后批量训练，而且有些参数可以改，后续也可以训练别的超参用:
k_att_list=[2,3,4,5]  # Attention额外添加边的邻居
gat_heads_list=[4,6,8,12,16,20]    # Multi-head-attention number
opt_dis_list=['True','False']    # True for optimized str train(DFT related)

2.analyze_dataset.py          20240520
file_path = 'gnn_data.save/Calculated_result_info.data'下面的dataset进行数据分析

3.cgat_job_script.py {file_name} {opt}             20240520
创建脚本文件用，file_name是目录，opt是true或者false

4.continue_cal_gcat.py           20240520
在SEED_{seed_num}目录下面使用，会重新创建job文件进行运算

5.job_on_hpc_id.py这老朋友了配合qdel使用        20240520

6.qdel_all.py 1 批量取消用           20240520

7.read_cgat_train_process.py             20240520
在SEED_{seed_num}目录下面使用，会读取每个SEED_{seed_num}目录下子模型运行状态

8.read_result.py                20240520
在SEED_{seed_num}目录下面使用，会统计并且对比每个目录下True和False的模型精度，并且进行直方图和箱式图的作图

9.show_fig.py {filename}/{-h}               20240520
这老朋友了画图用,具体看-h

10.train_gat_coords_edge_3.py: 模型架构脚本     20240520
改成一个库文件了    20240527


