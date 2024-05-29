更新：
predict_gat_coords_edge_3_*三个脚本改用process处理每个子进程防止内存溢出或者污染，然后同时第一个参数位置加上了database目录，输出文件产生在当前目录 20240527，且里面的预测函数都调用了train_gat_coords_edge_3.py



1.predict_from_scratch.py {model} {xsf_file} -f
用模型头开始预测一个结构的互换比例
会生成全图数据库和子图数据库

2.predict_from_scratch_super.py {model} {xsf_file} -f 
用新的扩胞数据进行预测的
会生成全图数据库和子图数据库

3.predict_gat_coords_edge_3_batch.py {database} {opt} {SEED_{seed_num}} -f
会生成：
1.output_prd_num_200_std_crt_0.1.csv
2.Top_199_prd_num_200_std_crt_0.1_diverse_datas.csv
3.std_of_predictions_prd_num_200_std_crt_0_1_10_6.png
4.value_prediction_Query-By-Committee_prd_num_200_std_crt_0_1_10_6.png
控制参数：
n_top = 200
predict_num=200  #0 for all prediction
用batch_gnn_train.py生成的SEED_{seed_num}来预测Prediction的数据库，并且判断有哪些是不稳定的点

4.predict_gat_coords_edge_3_manual.py {database} {model1,model2,...modeln} -f
会生成：
1.output_prd_num_200_std_crt_0.1.csv
2.Top_199_prd_num_200_std_crt_0.1_diverse_datas.csv
3.std_of_predictions_prd_num_200_std_crt_0_1_10_6.png
4.value_prediction_Query-By-Committee_prd_num_200_std_crt_0_1_10_6.png
控制参数：
n_top = 200
predict_num=200  #0 for all prediction
用多个模型来预测Prediction的数据，但是记住，和opt_dis=True参数不一样的模型会被忽略

5.predict_gat_coords_edge_3_opt_both_manual.py {database} {model1,model2,...modeln} -f
会生成：
1.output_prd_num_200_std_crt_0.1.csv
2.Top_199_prd_num_200_std_crt_0.1_diverse_datas.csv
3.std_of_predictions_prd_num_200_std_crt_0_1_10_6.png
4.value_prediction_Query-By-Committee_prd_num_200_std_crt_0_1_10_6.png
控制参数：
n_top = 200
predict_num=200  #0 for all prediction
用多个模型来预测Prediction的数据

6.extract_subgraph_from_predict_db.py {graph_ind} {["xsf", "xsf_std", "vasp"]} {knn}
从prediction数据库里面提取数据，应该是从predict_from_scratch*.py生成的数据库里面提取的

7.batch_extract_exchange_str_from_csv.py {csv_file} {全图database} {["xsf", "xsf_std", "vasp"]} -o -p
从Top_199_prd_num_200_std_crt_0.1_diverse_datas.csv这种文件里面提取top多少的结构，方便后续用QE-batch直接进行计算

8.predict_by_concen/multi-doping.py
doping_propor = [3,1,3,2]
设置一下，生成批量结构用

9.predict_by_concen/predict_from_scratch_data.py
和1差不多，但是会把预测的混排比例放到timelog.log里面

10.predict_by_concen/predict_from_concentration.py
prefix = 'LiNiO2-331_NCMT_3132_'设置一下参数，和8里面的参数应该要一致
设置完就会进行批量预测，然后会判断批次和上一批次平均值是不是小于一个阈值，否则就会用multi-doping产生新的一批结构进行预测

11.predict_by_concen/show_fig.py 画图老朋友了
