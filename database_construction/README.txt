1.create_gnn_database.py      20240520
在带有exchange_dir的目录下使用，会创建一个gnn_data.save目录，并且会：
1.1
如果extract_energy=1
会提取互换前后的能量，li-ni对，原始结构名称以及计算能量差，储存到gnn_data.save/Calculated_result_info.data
1.2
如果extract_xsf_file=1
提取原有结构，储存到gnn_data.save/xsf_str_original
提取互换结构，储存到gnn_data.save/xsf_str_variance
提取优化后的原有结构，储存到gnn_data.save/calculated_xsf_str_original
提取优化后的互换结构，储存到gnn_data.save/calculated_xsf_str_variance
1.3
如果creating_gnn_graph=1（需要在gat_env下运行）  
会创建总图gnn_data.save/wholegraphs_53d_features.db

2.read_graphs_batch_super.py & read_graphs_single_super.py & read_graphs_unitcell.py     20240520
从gnn_data.save/wholegraphs_53d_features.db提取li-ni边生成subgraph数据库，默认扩胞到包含5阶邻居且生成2-5的数据库
注意unitcell慎用，因为他产生的图数据库是错的(除了k=2的)，他只会提取单胞里面的图，但是你可以对照看扩胞以后那些原子收缩到单胞里面长啥样的
subgraphs_k_neighbor_{katt}_gnn_53d_feature_predict.db
subgraphs_k_neighbor_{katt}_gnn_53d_feature_train.db

3.database_gnn_merge.py     20240520
会创建一个gnn_data.save目录在当前目录，把其他目录的gnn_data.save下的文件都copy过来，但是你需要重新用1.3创建总图数据库

4.extract_exchange_str_from_db.py {graph_name} {Liindex_Niindex} {["xsf", "xsf_std", "vasp"]} -o -p -h     20240520
创建互换结构用
db_file = 'gnn_data.save/wholegraphs_53d_features.db'
output_dir='gnn_data.save/created_str_buffer'

5.extract_subgraph_from_db.py {subgraph_index} {["xsf", "xsf_std", "vasp"]} {knn} -p -f     20240520
查看子图以及绘制子图用

6.qe_out2RELAXLOG.py & qe_out_reader_gnn.py 应该是create_gnn_database.py 1.2调用的

7.show_fig.py画图老朋友了