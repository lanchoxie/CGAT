from concurrent.futures import ProcessPoolExecutor, as_completed
from torch_geometric.utils import add_self_loops, degree
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data, DataLoader
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core.periodic_table import Element
from torch_geometric.nn import global_mean_pool
from sklearn.preprocessing import MinMaxScaler
from pymatgen.core.structure import Structure
from torch_geometric.nn import MessagePassing
from mendeleev import element as MEN_element
from torch_geometric.nn import GATConv
from sklearn.metrics import r2_score
from pymatgen.io.xcrysden import XSF
from collections import Counter
from time import sleep
import matplotlib.pyplot as plt
import torch.nn.functional as F
import plotly.graph_objs as go
import subprocess as sb
import networkx as nx
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import warnings
import sqlite3
import random
import torch
import json
import math
import copy
import time
import sys
import os
import re

start_time_total=time.time()
code_dir=sys.path[0]

def input_check(str_in): 
    if str_in.endswith(".xsf"):
        return str_in
    else:
        raise argparse.argumenttypeerror('.xsf file expected.')

def model_check(model_in): 
    if model_in.endswith(".pth"):
        return model_in
    else:
        raise argparse.argumenttypeerror('.pth file expected.')

parser = argparse.ArgumentParser(description="Predict from scratch structure using certain model")
#parser.add_argument('--models','-m', nargs='+', help='Enter any number of models')
parser.add_argument('model',type=str, help='Enter predict model')
parser.add_argument('input_file', type=input_check, help='Enter predict .xsf structure file')
parser.add_argument("--fig",'-f',action='store_true',default=False,help='draw the figures of the predictions and std')

args = parser.parse_args()
model_path=args.model
str_name=args.input_file
str_name_base=os.path.basename(str_name).split(".xsf")[0]

predict_path=f"prediction_results_of_{str_name_base}"
os.makedirs(predict_path,exist_ok=True)
whole_db_name=f"{predict_path}/whole_graph_{str_name_base}.db"
sub_db_name=f"{predict_path}/subgraphs_{str_name_base}.db" 

# 忽略所有UserWarning警告
warnings.filterwarnings("ignore", category=UserWarning, module='pymatgen.*')
# 忽略所有DeprecationWarning警告
warnings.filterwarnings("ignore", category=DeprecationWarning, module='pymatgen.*')

batch_size=40
k_predict=5  #used for predictions

k_nn=2  #Not used paras
gat_eps=1e-5  #Not change paras

#Convert structure into graph

def create_whole_graphs(str_name_i):
      
    verbosity=1
    str_name=os.path.basename(str_name_i).split(".xsf")[0]
    xsf_file = f'{str_name}.xsf'

    def read_xsf(file_path):
        with open(file_path, 'r') as file:
            xsf_content = file.read()
        xsf = XSF.from_str(xsf_content)
        return xsf.structure

    crystal_original = read_xsf(xsf_file)
    # Create a deep copy of the crystal structure
    crystal = copy.deepcopy(crystal_original)

    def create_original_index_list(crystal):
        original_indices = []
        for i, site in enumerate(crystal):
            if str(site.specie) != 'O':  # 只为非氧原子创建映射
                original_indices.append(i)
        return original_indices

    #index projection from original to delete
    original_index_map = create_original_index_list(crystal)

    def map_old_indices_to_new(original_index_map, crystal):
        new_index_map = {}
        for new_index, old_index in enumerate(original_index_map):
            new_index_map[old_index] = new_index
        return new_index_map

    #index projection from Oxygen_delete structure to original structure
    new_index_map = map_old_indices_to_new(original_index_map, crystal)

    # Remove all oxygen sites from the copy
    crystal_metal = copy.deepcopy(crystal_original)
    crystal_metal.remove_species(["O"])

    # 使用pymatgen的CrystalNN找到临近的原子
    crystal_nn = CrystalNN()
    neighbors = [crystal_nn.get_nn_info(crystal, n=i) for i, _ in enumerate(crystal.sites)]

    neighbors_metal = [crystal_nn.get_nn_info(crystal_metal, n=i) for i, _ in enumerate(crystal_metal.sites)]


    # 构建图
    G = nx.Graph()
    for i, site in enumerate(crystal_original.sites):
        #lattice_info = crystal_original.lattice.matrix
        lattice_info = crystal_original.lattice.matrix.flatten()  # 将3x3矩阵拉平成9维列表
        coord = site.coords  # 获取原子的坐标，本身是一个三维向量

        lattice_coord = list(lattice_info) + list(coord) + [i]  # 将两者合并成12维列表+1
        G.add_node(i, element=site.specie.symbol, lattice_coord=lattice_coord)
        #G.nodes[node]['lattice_coord'] = lattice_coord
        
        for neighbor in neighbors[i]:
            neighbor_index = neighbor['site_index']
            if not G.has_edge(i, neighbor_index):
                G.add_edge(i, neighbor_index, edge_type='original')

    # 构建特定原子互换能的边
    for i, site in enumerate(crystal_metal.sites):
        for neighbor in neighbors_metal[i]:
            neighbor_index = neighbor['site_index']
            if ((site.specie == Element("Li")) and (crystal_metal[neighbor_index].specie == Element("Ni"))) or \
               ((site.specie == Element("Ni")) and (crystal_metal[neighbor_index].specie == Element("Li"))):
                if not G.has_edge(original_index_map[i], original_index_map[neighbor_index]):
                    G.add_edge(original_index_map[i], original_index_map[neighbor_index], edge_type='li_ni_edge')
                    

    # 初始化Ni，Mn，Ti，Co的计数为0
    counts = {"Li":0, "O":0, "Mg":0, "Al":0, "Ti":0, "V":0, "Mn":0, "Co":0, "Ni":0, "Zr":0}
    
    elements_total=[str(i.specie) for i in crystal_original.sites]    
    # 使用Counter来统计列表中每个元素的出现次数
    element_counter = Counter(elements_total)
    # 更新counts字典中的计数
    counts.update(element_counter)
    concen=[count for element, count in counts.items()]
    
    #节点信息嵌入：元素独热编码+电负性
    # 预设的元素列表
    elements_list = ["Li", "O", "Mg", "Al", "Ti", "V", "Mn", "Co", "Ni", "Zr"]
    # 创建独热编码字典
    element_to_onehot = {elem: [int(i == elem) for i in elements_list] for elem in elements_list}

    # 为图中的每个节点添加特征
    for node in G.nodes():
        element = G.nodes[node]['element']  # 获取节点的元素类型
        one_hot = element_to_onehot[element]  # 获取独热编码
        
        # 获取其他化学属性
        pymatgen_elem = Element(element)
        mendeleev_elem = MEN_element(element)

        electronegativity = [pymatgen_elem.X if pymatgen_elem.X else 0]  # 电负性
        atomic_radius = [pymatgen_elem.atomic_radius if pymatgen_elem.atomic_radius else 0]  # 原子半径
        ionization_energy = [mendeleev_elem.ionenergies.get(1, 0)]  # 离子化能
        atomic_mass = [mendeleev_elem.atomic_weight if mendeleev_elem.atomic_weight else 0]  # 原子质量
        melting_point = [mendeleev_elem.melting_point if mendeleev_elem.melting_point else 0]  # 熔点
        density = [mendeleev_elem.density if mendeleev_elem.density else 0]  # 密度
        thermal_conductivity = [mendeleev_elem.thermal_conductivity if mendeleev_elem.thermal_conductivity else 0]  # 热导率

        # 合并特征
        lattice_coord = G.nodes[node]['lattice_coord'] # 获取原子坐标和晶格信息
        features = one_hot + concen + electronegativity + atomic_radius + ionization_energy + atomic_mass + melting_point + density + thermal_conductivity + lattice_coord + [str_name]
        G.nodes[node]['feature'] = features

        # 特征向量的维度说明：
        # - 独热编码：10维（对应10种元素）
        # - 浓度：10维
        
        # - 电负性：1维
        # - 原子半径：1维
        # - 离子化能：1维
        # - 原子质量：1维
        # - 熔点：1维
        # - 密度：1维
        # - 热导率：1维

        # - 晶格：9维
        # - 坐标：3维
        # - 原子序号：1维

        # - 结构名称：1维
        # 总共：41维
    


        # 边向量的维度说明：
        # - 邻接列表：1维（对应2个节点）
        # 总共：1维
            
    return G

def store_wholegraphs_in_db(wholegraphs,db_name):
    # 连接到SQLite数据库
    conn = sqlite3.connect(db_name)
    c = conn.cursor()

    # 创建表格
    c.execute('''CREATE TABLE IF NOT EXISTS wholegraphs
                 (id INTEGER PRIMARY KEY, name TEXT, graph_data TEXT)''')

    # 清空表中的现有记录
    c.execute("DELETE FROM wholegraphs")

    # 存储每个子图
    for wholegraph in wholegraphs:
        name, graph = wholegraph[0], wholegraph[1]
        # 将图数据转换为JSON格式
        graph_data = json.dumps(nx.node_link_data(graph))
        # 插入图的名称和图数据
        c.execute("INSERT INTO wholegraphs (name, graph_data) VALUES (?, ?)", (name, graph_data))

    # 提交事务并关闭连接
    conn.commit()
    conn.close()

def get_all_wholegraphs(db_file):
    # 连接到SQLite数据库
    conn = sqlite3.connect(db_file)
    c = conn.cursor()

    # 查询所有图数据
    c.execute("SELECT name, graph_data FROM wholegraphs")
    rows = c.fetchall()

    # 关闭数据库连接
    conn.close()

    # 创建一个字典，用于存储名称和图结构的映射
    graphs = {}
    for name, graph_data in rows:
        graph = nx.node_link_graph(json.loads(graph_data))
        graphs[name] = graph

    return graphs

def extract_subgraphs_with_periodic_structure(graph, k):
    # 存储已处理的li-ni边的索引对
    processed_edges = set()

    def get_neighbors_xty(node,depth):
        get_nodes=[node]
        node_buffer=[]
        for i in range(1,depth+1):
            #print(i,"depth")
            for node in get_nodes:
                #print(node,"node")
                for neighbor in graph.neighbors(node):
                    if graph.edges[node,neighbor]['edge_type']=='original':
                        #if depth==1:
                            #print(neighbor,"select_neighbor")
                        node_buffer.append(neighbor)
            node_buffer=list(set(node_buffer))
            get_nodes.extend(node_buffer)
            
        return set(get_nodes)

    subgraphs_info = []  # 将存储 (u, v, subgraph) 的元组
    for u, v, data in graph.edges(data=True):
        # 只考虑有能量信息的li_ni_edge边
        if data.get('edge_type') == 'li_ni_edge':
            #print(u,v)
            # 无视索引对的顺序
            edge_tuple = tuple(sorted([u, v]))
   
            # 检查此索引对是否已经处理过
            if edge_tuple not in processed_edges:
                processed_edges.add(edge_tuple)
   
                collected_nodes_u=get_neighbors_xty(u,k)
                collected_nodes_v=get_neighbors_xty(v,k)
                
                #if (u==9)and(v==37):
                    #print("U:",collected_nodes_u)
                    #print("V:",collected_nodes_v)
                # 合并两个集合，同时去除重复元素
                collected_nodes = collected_nodes_u.union(collected_nodes_v)
   
                # 创建子图
                subgraph = nx.Graph()
                for node in collected_nodes:
                    subgraph.add_node(node, **graph.nodes[node])
                    subgraph.nodes[node]['original_index'] = node  # Store the original index
                    subgraph.nodes[node]['original_element'] = graph.nodes[node]['element']  # Store the original index
                # 添加边，确保边的两个端点都在子图中
                for node in collected_nodes:
                    for neighbor in graph.neighbors(node):
                        if neighbor in collected_nodes:
                            if graph.edges[node,neighbor]['edge_type']=='original':
                                subgraph.add_edge(node, neighbor, **graph[node][neighbor])
   
                subgraph.add_edge(u, v, **graph[u][v])
                subgraphs_info.append((u, v, subgraph))
            
                #subgraphs.append(subgraph)

    # 排序 subgraphs_info 列表，首先按照 u，然后按照 v
    subgraphs_info.sort(key=lambda x: (x[0], x[1]))
    # 从 subgraphs_info 中提取出排序后的子图
    sorted_subgraphs = [info[2] for info in subgraphs_info]
    return sorted_subgraphs

def store_subgraphs_in_db(subgraphs,name,k):
    # 连接到SQLite数据库
    conn = sqlite3.connect(name.replace(".db",f"_k_{k}.db"))
    c = conn.cursor()

    # 创建表格
    c.execute('''CREATE TABLE IF NOT EXISTS subgraphs
                 (id INTEGER PRIMARY KEY, graph_data TEXT)''')

    # 清空表中的现有记录
    c.execute("DELETE FROM subgraphs")
    # 存储每个子图
    for i, subgraph in enumerate(subgraphs):
        # 将图数据转换为JSON格式
        graph_data = json.dumps(nx.node_link_data(subgraph))
        c.execute("INSERT INTO subgraphs (graph_data) VALUES (?)", (graph_data,))

    # 提交事务并关闭连接
    conn.commit()
    conn.close()

def convert_sets_to_lists(obj):
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: convert_sets_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_sets_to_lists(i) for i in obj]
    else:
        return obj

def calculate_real_distance(pos1, pos2):
    """Calculate the real Euclidean distance between two points."""
    diff = pos2 - pos1
    return np.linalg.norm(diff)


def find_boundary_atoms(graph_in,prefix='Unitcell'):
    graph = copy.deepcopy(graph_in)
    verbosity=0
    #检查内部原子：都是正的，所有邻居距离都离自己不超过任意半个晶格矢量
    elements_list = ["Li", "O", "Mg", "Al", "Ti", "V", "Mn", "Co", "Ni", "Zr"]
    boundary_atoms = set()
    inner_atoms = set()
    lattice_vectors = None
    
    # 首先遍历一次来获取晶格矢量
    for node,attr in graph.nodes(data=True):
        feature = attr['feature']
        lattice_vectors = np.array(feature[-14:-5]).reshape((3, 3))
        break  # 只需要一次即可获取晶格矢量

    lat_lengths = np.linalg.norm(lattice_vectors, axis=1)

    #print(prefix,lat_lengths)
    
    # 检查每个原子与其邻居的距离
    for node,attr in graph.nodes(data=True):
        feature = attr['feature']
        node_id = int(feature[-2])  # 预计算节点ID以便重用
        pos_car = np.array(feature[-5:-2])  # 提取笛卡尔坐标
        pos = cartesian_to_fractional(pos_car, lattice_vectors)
        #pos = np.array([i+1 if i<0 else i for i in pos])
        #pos = np.array([i-1 if i>1 else i for i in pos])
        mark_inner_atom_step=1  
        element_index = np.argmax(feature[:10])  # 获取原子种类的索引
        element = elements_list[element_index]    # 根据索引获取原子种类

        # 打印当前原子的特定特征(-2)和邻居数量
        #print("Feature[-2] of current atom:", element,int(feature[-2])+1, "Number of neighbors:", neighbors_count,end='')
        #print("Neighbors:",[elements_list[np.argmax(graph.nodes[x]['feature'][:10])]+str(int(graph.nodes[x]['feature'][-2])+1)+f"NB{x}" for x in neighbors_])
        #print("Feature[-2] of current atom:", element,int(feature[-2])+1, "Number of neighbors:", neighbors_count)

        #for neighbor in graph.neighbors(node[0]):
        for neighbor in graph.neighbors(node):
            if graph.edges[neighbor,node]['edge_type']=='li_ni_edge':
                continue
            neighbor_feature = graph.nodes[neighbor]['feature']
            neighbor_pos_car = np.array(neighbor_feature[-5:-2])
            neighbor_pos = cartesian_to_fractional(neighbor_pos_car, lattice_vectors) 
            #neighbor_pos = np.array([i+1 if i<0 else i for i in neighbor_pos])
            #neighbor_pos = np.array([i-1 if i>1 else i for i in neighbor_pos])
            real_distance = calculate_real_distance(np.array(pos), np.array(neighbor_pos))
            #if 'bond_loc' in graph.edges[node,neighbor] and graph.edges[node,neighbor]['bond_loc']=='pbc2':
                #print(real_distance)
            if verbosity==1:
                if node_id==(atom-1):
                    #print(elements_list[np.argmax(node[1]['feature'][:10])],str(node_id+1),"NB-",elements_list[np.argmax(graph.nodes[neighbor]['feature'][:10])],str(int(graph.nodes[neighbor]['feature'][-2])+1),real_distance,[f"{m:.8f}" for m in neighbor_feature[-5:-2]])
                    print(elements_list[np.argmax(node[1]['feature'][:10])],str(node_id+1),"NB-",elements_list[np.argmax(graph.nodes[neighbor]['feature'][:10])],str(int(graph.nodes[neighbor]['feature'][-2])+1),real_distance,[f"{m:.8f}" for m in neighbor_feature[-5:-2]],[f"{m:.8f}" for m in cartesian_to_fractional(np.array(neighbor_feature[-5:-2]),lattice_vectors)])
            # 如果距离all not reach任一晶格矢量的一半
            if real_distance > 0.5:
            #if real_distance > any(0.5*lat_length for lat_length in lat_lengths):
                mark_inner_atom_step=-2
        if verbosity==1:
            if mark_inner_atom_step==-2: 
                reason='connect PBC atoms which absolute distance is larger than one of the 1/2 norm of lattice vector'
            elif mark_inner_atom_step==1:
                reason="!!INNER ATOM!!"
            if verbosity==1:
                print(element,node_id+1,mark_inner_atom_step,reason)
        if mark_inner_atom_step==1:
            inner_atoms.add(node)
            graph.nodes[node]['atom_loc']='inner'
    for node,attr in graph.nodes(data=True):
        attr['old_cnct']=set()
        attr['old_cnct_rm']=set()
        attr['old_cnct_cn']=set()
        if 'atom_loc' not in attr:
            graph.nodes[node]['atom_loc']='border'
            boundary_atoms.add(node)    
        elif 'atom_loc' in attr and attr['atom_loc']!='inner':
            graph.nodes[node]['atom_loc']='border'
            boundary_atoms.add(node)    
    for u,v,data in graph.edges(data=True):
        if data.get("edge_type")=='li_ni_edge':
            continue
        graph.edges[u,v]['bond_loc']='merge_bond'
        if graph.nodes[u]['atom_loc']=='border' and graph.nodes[v]['atom_loc']=='border':
            pos_u=cartesian_to_fractional(np.array(graph.nodes[u]['feature'][-5:-2]), lattice_vectors)
            pos_v=cartesian_to_fractional(np.array(graph.nodes[v]['feature'][-5:-2]), lattice_vectors)
            dis_uv=calculate_real_distance(np.array(pos_u), np.array(pos_v))
            #pos_u=np.array(graph.nodes[u]['feature'][-5:-2])
            #pos_v=np.array(graph.nodes[v]['feature'][-5:-2])
            #dis_uv=calculate_real_distance(np.array(pos_u), np.array(pos_v))
            #if any(dis_uv>0.5*lat_length for lat_length in lat_lengths):
            if dis_uv>0.5:
                graph.edges[u,v]['bond_loc']='pbc'
        '''
        if graph.nodes[u]['atom_loc']=='border' and graph.nodes[v]['atom_loc']=='border':
            graph.edges[u,v]['bond_loc']='pbc'
        elif graph.nodes[u]['atom_loc']=='inner' or graph.nodes[v]['atom_loc']=='inner':
            graph.edges[u,v]['bond_loc']='merge_bond'
        '''
    return list(boundary_atoms), lattice_vectors, graph

def color(directions):
    if len(directions) > 1:
        return None
    if '-a' in directions:
        return "Au"
    elif '+a' in directions:
        return "Ag"
    elif '-b' in directions:
        return "Li"
    elif '+b' in directions:
        return "Mn"
    elif '-c' in directions:
        return "O"
    elif '+c' in directions:
        return "H"
    elif 'c' in directions:
        return "Ti"

def write_xsf(filename, graph, lattice_vectors, selected_atoms=None, directs=None):
    elements_list = ["Li", "O", "Mg", "Al", "Ti", "V", "Mn", "Co", "Ni", "Zr"]
    if directs is not None:
        boundary_info={x[1]:[x[2],x[0],x[3]] for x in boundary_directions}
         
    with open(filename, 'w') as file:
        file.write("DIM-GROUP\n")
        file.write("         3              1\n")
        file.write(" PRIMVEC\n")
        for vec in lattice_vectors:
            file.write(f"  {vec[0]:.10f}    {vec[1]:.10f}    {vec[2]:.10f}\n")
        file.write(" PRIMCOORD\n")
        n_atoms = len(graph.nodes) if selected_atoms is None else len(selected_atoms)
        file.write(f"   {n_atoms}       1\n")
        
        if selected_atoms is None:
            nodes_data = graph.nodes(data=True)
            nodes_data = sorted(graph.nodes(data=True), key=lambda x:int(x[1]['feature'][-2]))
        else:
            nodes_data = [(node, graph.nodes[node]) for node in selected_atoms]
        
        for node_id, node_data in nodes_data:
            features = node_data['feature']
            atom_ind_vesta=int(features[-2])+1
            if directs is not None:
                #print(atom_ind_vesta,boundary_info[atom_ind_vesta][0])
                element=color(boundary_info[atom_ind_vesta][0])
            else:
                element_index = np.argmax(features[:10])  # 获取原子种类的索引
                element = elements_list[element_index]    # 根据索引获取原子种类
            pos = np.array(features[-5:-2])           # 获取原子的笛卡尔坐标
            file.write(f"  {element}   {pos[0]:.10f}   {pos[1]:.10f}   {pos[2]:.10f}\n")

def cartesian_to_fractional(pos, lattice_vectors):
    """Convert Cartesian coordinates to fractional coordinates."""
    inv_lattice = np.linalg.inv(lattice_vectors)
    frac_coords_array = np.dot(pos,inv_lattice)
    fractional_coord=frac_coords_array.tolist()
    #fractional_coord=[i+1 if i<0 else i for i in fractional_coord]
    #fractional_coord=[i-1 if i>1 else i for i in fractional_coord]

    return fractional_coord

def find_boundary_direction_by_fractional(boundary_atoms, graph, lattice_vectors):

    #First we got the boundary atoms by check wether the neighbors has absolute distance(not pbc distance) > 0.5 in frational
    #Second we judge PBC boundary: On which direction the neighbor and the center atom has fractional diff > 0.5 (since we correct the fractional coords into 0~1 in cartesian_to_fractional() function)
    #Third we drain all over the neighbors to collect how many PBC boundary has the center atom and the neighbor connect across
    #Fourth we judge the list in step 3 , judge the atoms is on positive or negative direction by judege its fractional coordinates is > 0.5 or not
    #Fifth we list(set()) the list we aquired in step 4, that stores the PBC boundary property of the center atom
    
    lat_lengths = np.linalg.norm(lattice_vectors, axis=1)
    boundary_directions = []
    axis_labels = ['a', 'b', 'c']
    atom_ind_dir_pair_in={}
    #global atom_ind_dir_pair
    for atom in boundary_atoms:
        feature = graph.nodes[atom]['feature']
        elements_list = ["Li", "O", "Mg", "Al", "Ti", "V", "Mn", "Co", "Ni", "Zr"]
        element_index = np.argmax(feature[:10])  # Identify the type of atom
        element = elements_list[element_index]
        ind = int(feature[-2])+1
        pos = np.array(feature[-5:-2])  # Cartesian coordinates of the atom
        frac_coords = cartesian_to_fractional(pos, lattice_vectors)  # Convert to fractional coordinates

        directions = []

        # Check neighbors to determine boundary direction based on distance
        for neighbor in graph.neighbors(atom):
            if graph.edges[atom, neighbor]['edge_type'] == 'original':
                #if frozenset([atom,neighbor]) not in atom_ind_dir_pair_in:
                    #atom_ind_dir_pair_in[frozenset([atom, neighbor])]=set()
                if (atom,neighbor) not in atom_ind_dir_pair_in:
                    atom_ind_dir_pair_in[(atom, neighbor)]=set()
                neighbor_pos_cartesian = np.array(graph.nodes[neighbor]['feature'][-5:-2])
                neighbor_pos_fractional = cartesian_to_fractional(neighbor_pos_cartesian, lattice_vectors)  # 转换为分数坐标
                diff = [abs(frac_coords[i]-v) for i,v in enumerate(neighbor_pos_fractional)]
                distance = calculate_real_distance(np.array(frac_coords), np.array(neighbor_pos_fractional))  # 计算基于分数坐标的距离
                #diff = [abs(pos[i]-v) for i,v in enumerate(neighbor_pos_cartesian)]
                #distance = calculate_real_distance(np.array(pos), np.array(neighbor_pos_cartesian))  # 计算基于分数坐标的距离

                #if any(distance>0.5*lat_length for lat_length in lat_lengths):
                if distance > 0.5:  # Threshold for identifying boundary neighbors
                    # Check which axes are involved in the boundary
                    for i, d in enumerate(diff):
                        if abs(d) > 0.5:  # Check if the difference along any axis is significant
                        #if abs(d) > 0.5*lat_lengths[i]:  # Check if the difference along any axis is significant
                            if frac_coords[i] > 0.5:
                            #if pos[i] > 0.5*lat_lengths[i]:
                                directions.append(f"+{axis_labels[i]}")
                                atom_ind_dir_pair_in[(atom, neighbor)].add(f"+{axis_labels[i]}") #无视a，b顺序
                            elif frac_coords[i] <= 0.5:
                            #elif frac_coords[i] <= 0.5*lat_lengths[i]:

                                directions.append(f"-{axis_labels[i]}")
                                atom_ind_dir_pair[(atom, neighbor)].add(f"-{axis_labels[i]}") #无视a，b顺序
        corrected_directions=list(set(directions))
        #print(atom,element,ind,corrected_directions,[f"{i:.8f}" for i in frac_coords],[f"{i:.8f}" for i in pos])

        boundary_directions.append([element, int(feature[-2])+1, corrected_directions, frac_coords])

    return boundary_directions


def check_boundary_atom_to_chose_edge(boundary_atoms,boundary_directions,graph_in,k,prefix="Unitcell"):
    
    #Check the boundary atom was on k-1 th neighbor of all li_ni pair or not, if so, enlarge twice on that boundary direction

    #Add bd_direction attribute on atoms on the boundary
    '''
    Input parameter1: boundary_atoms index in graph (No in vesta format,which is not same in VESTA index)
    Input parameter2: boundary_direction index in vesta and direction infos:element,ind(VESTA),directions,frac_coords 
    Input parameter3: graph store the coords and bonds connection relations
    Input parameter4: k which is the subgraph range

    Output parameter1: the boundary where lattice should enlarge,list format
    Output parameter2: the graph with boundary direction in node attr
    '''
    def get_neighbors_xty(node,depth):
        get_nodes=[node]
        node_buffer=[]
        for i in range(1,depth+1):
            #print(i,"depth")
            for node in get_nodes:
                #print(node,"node")
                for neighbor in graph.neighbors(node):
                    if graph.edges[node,neighbor]['edge_type']=='original':
                        #if depth==1:
                            #print(neighbor,"select_neighbor")
                        node_buffer.append(neighbor)
            node_buffer=list(set(node_buffer))
            get_nodes.extend(node_buffer)
            
        return set(get_nodes)
  

    graph=copy.deepcopy(graph_in)

    boundary_atoms_vesta=[]
    for node,attr in graph.nodes(data=True):
        if node in boundary_atoms:  
            boundary_atoms_vesta.append(int(attr['feature'][-2])+1)

    boundary_info={x[1]:[x[2],x[0],x[3]] for x in boundary_directions}
    '''
    #Extract the li-ni edge
    processed_edges = set()
    subgraphs = []
    for u, v, data in graph.edges(data=True):
        # 只考虑有能量信息的li_ni_edge边
        if data.get('edge_type') == 'li_ni_edge':
            #print(u,v)
            # 无视索引对的顺序
            edge_tuple = tuple(sorted([u, v]))
    
            # 检查此索引对是否已经处理过
            if edge_tuple not in processed_edges:
                processed_edges.add(edge_tuple)
    
                collected_nodes_u=get_neighbors_xty(u,k-1)   #!!!Here is the key point, if the boundary atoms is not in the k-1 nearest neighbors, the k must be contained in the current cell and need no expansion of the lattice, otherwise you must enlarge the lattice at direction where the boundary atoms was contained 
                collected_nodes_v=get_neighbors_xty(v,k-1)
                
                #if (u==9)and(v==37):
                    #print("U:",collected_nodes_u)
                    #print("V:",collected_nodes_v)
                # 合并两个集合，同时去除重复元素
                collected_nodes = collected_nodes_u.union(collected_nodes_v)
    
    
                # 创建子图
                subgraph = nx.Graph()
                for node in collected_nodes:
                    subgraph.add_node(node, **graph.nodes[node])
                    subgraph
                # 添加边，确保边的两个端点都在子图中
                for node in collected_nodes:
                    for neighbor in graph.neighbors(node):
                        if neighbor in collected_nodes:
                            if graph.edges[node,neighbor]['edge_type']=='original':
                                subgraph.add_edge(node, neighbor, **graph[node][neighbor])
    
                subgraph.add_edge(u, v, **graph[u][v])
                subgraphs.append(subgraph)

    #Judge directions
    directions=set()
    for subgraph in subgraphs:
        for node,attr in subgraph.nodes(data=True):
            node_ind=int(attr['feature'][-2])+1
            if node_ind in boundary_atoms_vesta:
                for dir_i in boundary_info[node_ind][0]:   #boundary_info:{ind(VESTA):[[directions],element,[frac_coords]]}
                    directions.add(dir_i)
    print(directions)
    '''

    for node,attr in graph.nodes(data=True):
        feature = attr['feature']
        ind_vesta=int(feature[-2])+1
        if int(attr['feature'][-2])+1 in [key for key,v in boundary_info.items()]:
            graph.nodes[node]['bd_directions']=boundary_info[ind_vesta][0]
            graph.nodes[node]['bd_directions_init']=boundary_info[ind_vesta][0].copy()
        else:
            graph.nodes[node]['bd_directions']=[]
            graph.nodes[node]['bd_directions_init']=[]

    dir_graph=copy.deepcopy(graph)
    a_pair=[]
    b_pair=[]
    c_pair=[]

    #for u,v,data in dir_graph.edges(data=True):
    #    if 'bond_loc' in data:
    #        print(u,v,data)
    #sys.exit(0)


    for node,attr in dir_graph.nodes(data=True):
        if dir_graph.nodes[node]['bd_directions']==['-a']:
            #print('-a :found!',node,[[dir_graph.nodes[nb]['bd_directions'],dir_graph.edges[node,nb]['edge_type'],dir_graph.edges[node,nb]['bond_loc']] for nb in dir_graph.neighbors(node) if dir_graph.edges[node,nb]['edge_type']=='original'])
            opposite_dir = [nb for nb in dir_graph.neighbors(node) if dir_graph.nodes[nb]['bd_directions']==['+a'] and dir_graph.edges[node,nb]['edge_type']=='original' and (dir_graph.edges[node,nb]['bond_loc']=='pbc' or dir_graph.edges[node,nb]['bond_loc']=='pbc2')]
            if opposite_dir!=[]:
                #print("--------------------------------------",opposite_dir)
                a_pair.append((node,opposite_dir[0]))
                #a_pair=(node,opposite_dir[0])
                #break

    for node,attr in dir_graph.nodes(data=True):
        if dir_graph.nodes[node]['bd_directions']==['-b']:
            opposite_dir = [nb for nb in dir_graph.neighbors(node) if dir_graph.nodes[nb]['bd_directions']==['+b'] and dir_graph.edges[node,nb]['edge_type']=='original' and (dir_graph.edges[node,nb]['bond_loc']=='pbc' or dir_graph.edges[node,nb]['bond_loc']=='pbc2')]
            if opposite_dir!=[]:
                b_pair.append((node,opposite_dir[0]))
                #b_pair=(node,opposite_dir[0])
                #break

    for node,attr in dir_graph.nodes(data=True):
        if dir_graph.nodes[node]['bd_directions']==['-c']:
            opposite_dir = [nb for nb in dir_graph.neighbors(node) if dir_graph.nodes[nb]['bd_directions']==['+c'] and dir_graph.edges[node,nb]['edge_type']=='original' and (dir_graph.edges[node,nb]['bond_loc']=='pbc' or dir_graph.edges[node,nb]['bond_loc']=='pbc2')]
            if opposite_dir!=[]:
                c_pair.append((node,opposite_dir[0]))
                #c_pair=(node,opposite_dir[0])
                #break

    if a_pair==[] or b_pair==[] or c_pair==[]:
        if a_pair==[]:
            print("No a dir match!")
        if b_pair==[]:
            print("No b dir match!")
        if c_pair==[]:
            print("No c dir match!")
        sys.exit(0)

    #print(prefix,a_pair,b_pair,c_pair)

    ka = min([nx.shortest_path_length(dir_graph, source=a_pair_i[0], target=a_pair_i[1]) for a_pair_i in a_pair])
    kb = min([nx.shortest_path_length(dir_graph, source=b_pair_i[0], target=b_pair_i[1]) for b_pair_i in b_pair])
    kc = min([nx.shortest_path_length(dir_graph, source=c_pair_i[0], target=c_pair_i[1]) for c_pair_i in c_pair])

    pre_rm_edge=[]
    for u,v,data in dir_graph.edges(data=True):
        if data['edge_type']=='original' and (data['bond_loc']=='pbc' or data['bond_loc']=='pbc2'):
            pre_rm_edge.append((u,v))

    for u,v in pre_rm_edge:
        if dir_graph.has_edge(u,v):
            dir_graph.remove_edge(u,v)
            #print("Remove:",u,v)
    
    ka2 = min([nx.shortest_path_length(dir_graph, source=a_pair_i[0], target=a_pair_i[1]) for a_pair_i in a_pair])
    kb2 = min([nx.shortest_path_length(dir_graph, source=b_pair_i[0], target=b_pair_i[1]) for b_pair_i in b_pair])
    kc2 = min([nx.shortest_path_length(dir_graph, source=c_pair_i[0], target=c_pair_i[1]) for c_pair_i in c_pair])
    #ka2 = nx.shortest_path_length(dir_graph, source=a_pair[0], target=a_pair[1])
    #kb2 = nx.shortest_path_length(dir_graph, source=b_pair[0], target=b_pair[1])
    #kc2 = nx.shortest_path_length(dir_graph, source=c_pair[0], target=c_pair[1])
    
    #print(f"A: {ka} ---> {ka2}")
    #print(f"B: {kb} ---> {kb2}")
    #print(f"C: {kc} ---> {kc2}")

    directions=[]
    further_expand=[]

    if ka2>=k:
        pass
    elif ka2<k and k<=ka2*2:
        directions.append('+a')
    elif ka2*2<k and k<=ka2*3:
        directions.append('+a')
        directions.append('-a')
    else:
        directions.append('+a')
        directions.append('-a')
        further_expand.append('a')

    if kb2>=k:
        pass
    elif kb2<k and k<=kb2*2:
        directions.append('+b')
    elif kb2<k*2 and k<=kb2*3:
        directions.append('+b')
        directions.append('-b')
    else:
        directions.append('+b')
        directions.append('-b')
        further_expand.append('b')

    if kc2>=k:
        pass
    elif kc2<k and k<=kc2*2:
        directions.append('+c')
    elif kc2<k*2 and k<=kc2*3:
        directions.append('+c')
        directions.append('-c')
    else:
        directions.append('+c')
        directions.append('-c')
        further_expand.append('c')

    #print(prefix,f": Input {k}-th neighbor subgraph recommanded:")
    #print(prefix,f": Max k-th neighbor on A:{ka2}, B:{kb2}, C:{kc2}")
    #if further_expand:
    #    print(prefix,f": Notice!!! The {further_expand} dir reach a maximum and need another expand iteration!")
    #print(prefix,": The expand dir on this round:",directions)
    #sys.exit(0)

    return directions,graph,further_expand


#image_vector_list = ['-a', '+c', '+a', '-c', '-b', '+b']

def creat_image_graphs(G_original,image_vector_list):

    #create imagines of the orginal graph and makes the image node number unique by the image_vector_list index and the number of the nodes in a graph
    #Remove li_ni edges in the image graph since we don't want duplicated one
    #give each graph a direct_vec attribute with vector labels a,b,c direction weight and combos are sorted as a,b,c.

    combinations_to_create = []

    def is_valid_combination(combo):
        """检查组合是否有效，即不包含相对的矢量"""
        seen_axes = set()
        for vec in combo:
            if vec[-1] in seen_axes:
                return False
            seen_axes.add(vec[-1])
        return True
    
    def sort_combination(combo):
        """根据'a', 'b', 'c'的顺序排序组合"""
        priority = {'a': 1, 'b': 2, 'c': 3}
        return tuple(sorted(combo, key=lambda x: priority[x[-1]]))
    
    def combo_to_vector(combo):
        """将组合转换成矢量表示"""
        vector = np.zeros(3)  # 初始化为 [0, 0, 0]
        for vec in combo:
            sign = -1 if '-' in vec else 1
            axis = 'abc'.index(vec[-1])  # 'a' -> 0, 'b' -> 1, 'c' -> 2
            vector[axis] = sign
        return vector
    
    # 生成所有可能的组合
    combinations_to_create = []
    for i in range(1, len(image_vector_list) + 1):
        for combo in combinations(image_vector_list, i):
            if is_valid_combination(combo):
                sorted_combo = sort_combination(combo)
                combinations_to_create.append(sorted_combo)
    
    # 打印组合和对应的矢量表示
    graph_vec_dir_list=[]
    for combo in combinations_to_create:
        vector_dir = combo_to_vector(combo)
        graph_vec_dir_list.append([int(i) for i in vector_dir])
        #print(f"{combo}: {vector_dir}")

    #print(combinations_to_create,len(combinations_to_create))  
    
    # 创建一个字典来存储所有生成的镜像图
    graphs = {}
    node_count = G_original.number_of_nodes()
    # 遍历所有组合以创建镜像图
    for idx, combo in enumerate(combinations_to_create):
        G_new = copy.deepcopy(G_original)
        translation_vector = np.zeros((3,))
        translation_vector2 = np.zeros((3,))
    
        for vec in combo:
            sign = -1 if '-' in vec else 1
            axis = vec[-1]  # 'a', 'b', 或 'c'
    
            # 找到对应的晶格矢量
            if axis == 'a':
                vector = sign * np.array(G_new.nodes[0]['feature'][-14:-11]).reshape((3,))
                vector2 = sign * np.array(G_new.nodes[0]['feature'][-26:-23]).reshape((3,))
            elif axis == 'b':
                vector = sign * np.array(G_new.nodes[0]['feature'][-11:-8]).reshape((3,))
                vector2 = sign * np.array(G_new.nodes[0]['feature'][-23:-20]).reshape((3,))
            elif axis == 'c':
                vector = sign * np.array(G_new.nodes[0]['feature'][-8:-5]).reshape((3,))
                vector2 = sign * np.array(G_new.nodes[0]['feature'][-20:-17]).reshape((3,))
    
            translation_vector += vector
            translation_vector2 += vector2
    
        # 更新所有节点的坐标
        for node in G_new.nodes:
            original_coords = np.array(G_new.nodes[node]['feature'][-5:-2])
            original_coords2 = np.array(G_new.nodes[node]['feature'][-17:-14])
            G_new.nodes[node]['feature'][-5:-2] = original_coords + translation_vector
            G_new.nodes[node]['feature'][-17:-14] = original_coords2 + translation_vector2
            vector_dir = combo_to_vector(combo)
            G_new.nodes[node]['direct_vec'] = [int(i) for i in vector_dir]
        # 更新所有节点的编号
        mapping = {}
        for node in list(G_new.nodes):
            new_node = node + (1 + idx) * node_count
            mapping[node] = new_node
            G_new.nodes[node]['image'] = "".join(combo)
        
        nx.relabel_nodes(G_new, mapping, copy=False)

        #Cannot remove edge during edge iteration, thus use this method:dictionary changed size during iteration
        to_remove = []
        for u, v, data in G_new.edges(data=True):
            if data['edge_type'] == 'li_ni_edge':
                to_remove.append((u, v))
                data['keep_ext'] = False
        for u, v in to_remove:
            G_new.remove_edge(u, v)
        # 保存平移后的图
        graphs["".join(combo)] = G_new
 
    return graphs,combinations_to_create
    # 此时 `graphs` 包含了所有生成的镜像图，每个镜像图都以其矢量组合的字符串标识为键（如 '-a+b-c'）

def expand_image_graphs(G_images,vector_list):

    G_new = copy.deepcopy(G_images)

    count_a = sum(1 for x in vector_list if 'a' in x)+1
    count_b = sum(1 for x in vector_list if 'b' in x)+1
    count_c = sum(1 for x in vector_list if 'c' in x)+1
    #print("+"*20,count_a,count_b,count_c)

    #     晶格矢量的每个方向都乘以其对应的次数


    for node,attr in G_new.nodes(data=True):
        attr['original_lat']=attr['feature'][-14:-5]
        for i in range(3):
            attr['feature'][-14+i] *= count_a
        
        # 中间三个元素对应b方向
        for i in range(3, 6):
            attr['feature'][-14+i] *= count_b
        
        # 最后三个元素对应c方向
        for i in range(6, 9):
            attr['feature'][-14+i] *= count_c
    return G_new

def add_unique_tuple(sets, new_tuple):
    # 对元组进行排序，确保较小的元素总是在前
    sorted_tuple = tuple(sorted(new_tuple))
    # 添加排序后的元组到集合中
    sets.add(sorted_tuple)

def connect_graphs(G_images, all_directs, atom_ind_dir_pair):

    for name,graph in G_images.items():
        if graph is None:
            print(f"{name} is None!")
            return None

    G_new = copy.deepcopy(G_images)

    count_a = sum(1 for x in all_directs if 'a' in x)+1
    count_b = sum(1 for x in all_directs if 'b' in x)+1
    count_c = sum(1 for x in all_directs if 'c' in x)+1
    base_dir=['a','b','c']
    expand_vecs={}
    for i,v in enumerate(base_dir):
        expand_vecs[i]=[1 if '+' in x else -1 for x in all_directs if v in x]
        expand_vecs[i].append(0)

    #print("+"*20,count_a,count_b,count_c)

    #for name,graph in G_new.items():
    #    for node,attr in graph.nodes(data=True):
    #        attr['original_lat']=attr['feature'][-14:-5]
    #     晶格矢量的每个方向都乘以其对应的次数
    for name,graph in G_new.items():
        for node,attr in graph.nodes(data=True):
            for i in range(3):
                attr['feature'][-14+i] *= count_a
                attr['feature'][-26+i] *= count_a
            
            # 中间三个元素对应b方向
            for i in range(3, 6):
                attr['feature'][-14+i] *= count_b
                attr['feature'][-26+i] *= count_b
            
            # 最后三个元素对应c方向
            for i in range(6, 9):
                attr['feature'][-14+i] *= count_c
                attr['feature'][-26+i] *= count_c

    #for name,graph in G_images.items():
    #    G_merged = nx.union(G_merged, graph)  # 注意重命名避免节点冲突
    #merged_graph = reduce(lambda x, y: nx.union(x, y, [graph for name,graph in G_images.items]))
    G_merged=None
    for name,graph in G_new.items():
        if G_merged is None:
            G_merged = graph.copy()
        else:
            G_merged = nx.union(G_merged, graph)

    G_merged_info=copy.deepcopy(G_merged)
 
    boundary_dict = {feature: {} for feature in {G_merged.nodes[node]['feature'][-2] for node in G_merged}}
    elements_list = ["Li", "O", "Mg", "Al", "Ti", "V", "Mn", "Co", "Ni", "Zr"]

    for node,attr in G_merged.nodes(data=True):
        lattice_vectors_expand = np.array(G_merged.nodes[node]['feature'][-14:-5]).reshape((3, 3))
        orig_lattice_vectors = np.array(G_merged.nodes[node]['original_lat']).reshape((3, 3))
        break

    #print("EXP_LV:",lattice_vectors_expand)
    lat_lengths = np.linalg.norm(orig_lattice_vectors, axis=1)
    #print("LAT_LENGTH:",lat_lengths)
    pre_remove_edge=set()
    for node,attr in G_merged.nodes(data=True):
        if attr['atom_loc']=='border':
            for neighbor in G_merged.neighbors(node):
                #if G_merged.edges[node,neighbor]['edge_type']=='original' and G_merged.nodes[neighbor]['atom_loc']=='border':
                if G_merged.edges[node,neighbor]['edge_type']=='original' and G_merged.edges[node,neighbor]['bond_loc']=='pbc':
                    attr['old_cnct'].add(f"{neighbor}-{G_merged.nodes[neighbor]['feature'][-2]}")
                    G_merged.nodes[neighbor]['old_cnct'].add(f"{node}-{G_merged.nodes[node]['feature'][-2]}")
                    add_unique_tuple(pre_remove_edge, (node,neighbor))
                    boundary_dict[G_merged.nodes[node]['feature'][-2]][tuple(G_merged.nodes[node]['direct_vec'])]=node
                    boundary_dict[G_merged.nodes[neighbor]['feature'][-2]][tuple(G_merged.nodes[neighbor]['direct_vec'])]=neighbor #tuple可以作索引
    #check boundary dict is true or not
    #count=0
    #for i in range(108):
    #    if len(boundary_dict[i])!=27:
    #        print(i, boundary_dict[i])
    #        count+=1
    #print(count)
    #sys.exit(0)

    for u,v in pre_remove_edge:
        if G_merged.has_edge(u,v): 
            G_merged.remove_edge(u,v) 

    bd_atom=set()
    pbc_edge=set() 
    remove_cnct_symbol={}

    new_add_bond={}
    new_add_bond_infos={}

    for node,attr in G_merged.nodes(data=True):
        remove_cnct_symbol[node]=set()
    for node,attr in G_merged.nodes(data=True):
        #remove_cnct_symbol[node]=set()
        node_dir=G_merged.nodes[node]['direct_vec']
        node_ind=G_merged.nodes[node]['feature'][-2]
        if attr['atom_loc']=='border':
            #print(f"{node}:break_bonds:",len(G_merged.nodes[node]['old_cnct']),end='')
            cnct_i=[]
            for old_cnct in list(G_merged.nodes[node]['old_cnct']): 
                next_loop=0
                av_node_orig = int(old_cnct.split("-")[0])
                ind = int(old_cnct.split("-")[1])
                #Use ind and direct_vec attr on node to get connect atom 

                bond_type='merge_bond2'
                match_dir=atom_ind_dir_pair[(node_ind,ind)]
                add_vec_raw=[0,0,0]
                for vec in match_dir:
                    sign = -1 if '-' in vec else 1
                    axis = 'abc'.index(vec[-1])  # 'a', 'b', 或 'c'
                    add_vec_raw[axis]=sign
                
                target_dir_nb=[add_vec_raw[i]+v for i,v in enumerate(node_dir)]
                for i,v in enumerate(target_dir_nb):
                    if v>max(expand_vecs[i]):
                        target_dir_nb[i]=min(expand_vecs[i])
                        bond_type='pbc2'                
                    elif v<min(expand_vecs[i]):
                        target_dir_nb[i]=max(expand_vecs[i])
                        bond_type='pbc2'                
                node_nb=boundary_dict[ind][tuple(target_dir_nb)]

                if frozenset([node_nb,node]) in new_add_bond:
                    continue
                
                matching_keys = [key for key in new_add_bond if node in key]
                for key in matching_keys:
                    if new_add_bond[key]==frozenset([node_ind,ind]):
                        if frozenset([node_nb,node])!=key:
                            old_atom1=list(key)[0]
                            old_atom2=list(key)[1]
                            old_key1=list(new_add_bond[key])[0]
                            old_key2=list(new_add_bond[key])[1]
                            print("ERROR: duplicated match!",key,frozenset([node_nb,node]),"ft:",new_add_bond[key])
                            print("Directions two pbc atom cnct:",match_dir)
                            print(f"Original atom {node} direct vector:",G_merged.nodes[node]['direct_vec'])
                            print(f"Exsting atom {old_atom1 if old_atom1!=node else old_atom2}direct vector:",G_merged.nodes[old_atom1 if old_atom1!=node else old_atom2]['direct_vec'])
                            print(f"New atom {node_nb} direct vector:",G_merged.nodes[node_nb]['direct_vec'])
                            print(f"Original atom {node} pos:",G_merged.nodes[node]['feature'][-5:-2])
                            print(f"Exsting atom {old_atom1 if old_atom1!=node else old_atom2} pos:",G_merged.nodes[old_atom1 if old_atom1!=node else old_atom2]['feature'][-5:-2])
                            print(f"New atom {node_nb} pos:",G_merged.nodes[node_nb]['feature'][-5:-2])
                            sys.exit(0)
                        else:
                            next_loop=1
                if next_loop:
                    continue

                new_add_bond[frozenset([node,node_nb])]=frozenset([node_ind,ind])
                new_add_bond_infos[frozenset([node,node_nb])]=[frozenset([node,av_node_orig]),bond_type]

    for key in new_add_bond:
        bd_list=list(key)
        if not G_merged.has_edge(bd_list[0],bd_list[1]):
             orig_bond=list(new_add_bond_infos[key][0])
             bond_type=new_add_bond_infos[key][1]
             G_merged.add_edge(bd_list[0],bd_list[1])
             copy_edge_attributes(G_merged_info, G_merged, orig_bond[0], orig_bond[1], bd_list[0], bd_list[1])
             G_merged.edges[bd_list[0],bd_list[1]]['bond_loc']=bond_type

    #print("half LAT_LENGTH:",[0.5*i for i in lat_lengths])

    for node,attr in G_merged.nodes(data=True):
        current_nb2=[i for i in G_merged.neighbors(node) if G_merged.edges[node,i]['edge_type']=='original']
        if len(current_nb2)!=6:
            print("NODE",node,G_merged.nodes[node]['atom_loc'],"*"*40)
    #sys.exit(0)

    border_atom=set()
    for u,v,data in G_merged.edges(data=True):
        if data['edge_type']=='original' and data['bond_loc']=='pbc2':
            border_atom.add(u)
            border_atom.add(v) 
        #if G_merged.nodes[u]['atom_loc']!='border':
        #    print(u)
        #if G_merged.nodes[v]['atom_loc']!='border':
        #    print(v)

    #print("BDATOM:",len(border_atom))
    for node,attr in G_merged.nodes(data=True):
        if node not in border_atom:
            G_merged.nodes[node]['bd_directions']=[]
            G_merged.nodes[node]['atom_loc']='inner'
    return G_merged,expand_vecs
                    
def copy_edge_attributes(G_source, G_target, u_s, v_s, u_t, v_t):
    # 检查两个图中的边是否存在
    if G_source.has_edge(u_s, v_s) and G_target.has_edge(u_t, v_t):
        # 获取源图中边的所有属性
        attributes = G_source[u_s][v_s]
        # 将所有属性设置到目标图的对应边上
        for key, value in attributes.items():
            if key != 'bond_loc':
                G_target[u_t][v_t][key] = value
    else:
        return None
        if G_source.has_edge(u_s, v_s):
            print("copied edge does not exist on graphs.")
        else:
            print("original edge does not exist on graphs.")


def handle_other_manual_edges(G_expand,expand_vecs,spe_edge_attr):
    elements_list = ["Li", "O", "Mg", "Al", "Ti", "V", "Mn", "Co", "Ni", "Zr"]
    #print("Current Graph:",G_expand)
    for node,attr in G_expand.nodes(data=True):
        lattice_vectors_expand = np.array(G_expand.nodes[node]['feature'][-14:-5]).reshape((3, 3))
        orig_lattice_vectors = np.array(G_expand.nodes[node]['original_lat']).reshape((3, 3))
        break
    
    lat_lengths = np.linalg.norm(orig_lattice_vectors, axis=1)

    axis_labels = ['a', 'b', 'c']
    bond_break={}
    li_ft=[]
    count_nm=0
    atom_ind_dir_pair_lini={}
    for u,v,data in G_expand.edges(data=True):
        if data['edge_type']=='li_ni_edge': 
            count_nm+=1
            pos_u=cartesian_to_fractional(np.array(G_expand.nodes[u]['feature'][-5:-2]), orig_lattice_vectors)
            pos_v=cartesian_to_fractional(np.array(G_expand.nodes[v]['feature'][-5:-2]), orig_lattice_vectors)
            #pos_u=np.array(G_expand.nodes[u]['feature'][-5:-2])
            #pos_v=np.array(G_expand.nodes[v]['feature'][-5:-2])
            dis_uv=calculate_real_distance(np.array(pos_u), np.array(pos_v))
            #if any(dis_uv>0.5*lat_length for lat_length in lat_lengths):
            if dis_uv>0.5:
                if (u,v) not in atom_ind_dir_pair_lini and (v,u) not in atom_ind_dir_pair_lini:
                    atom_ind_dir_pair_lini[(u,v)]=set()
                    atom_ind_dir_pair_lini[(v,u)]=set()
                diff = [abs(pos_u[i]-v) for i,v in enumerate(pos_v)]
                ft_u=G_expand.nodes[u]['feature'][-2]
                ft_v=G_expand.nodes[v]['feature'][-2]
                #ft_u=G_expand.nodes[u]['old_ind']
                #ft_v=G_expand.nodes[v]['old_ind']
                ele_u=elements_list[np.argmax(G_expand.nodes[u]['feature'][:10])]
                ele_v=elements_list[np.argmax(G_expand.nodes[v]['feature'][:10])]
                if ele_u=='Li':
                    li_ft.append(ft_u)
                    for i, d in enumerate(diff):
                        if abs(d) > 0.5:  # Check if the difference along any axis is significant
                            if pos_v[i] > 0.5:
                                atom_ind_dir_pair_lini[(v, u)].add(f"+{axis_labels[i]}") #无视a，b顺序
                            elif pos_v[i] <= 0.5:
                                atom_ind_dir_pair_lini[(v, u)].add(f"-{axis_labels[i]}") #无视a，b顺序
                elif ele_v=='Li':
                    li_ft.append(ft_v)
                    for i, d in enumerate(diff):
                        if abs(d) > 0.5:  # Check if the difference along any axis is significant
                            if pos_u[i] > 0.5:
                                atom_ind_dir_pair_lini[(u, v)].add(f"+{axis_labels[i]}") #无视a，b顺序
                            elif pos_u[i] <= 0.5:
                                atom_ind_dir_pair_lini[(u, v)].add(f"-{axis_labels[i]}") #无视a，b顺序

                if (u,v) not in bond_break and (v,u) not in bond_break:
                    bond_break[(u,v)]=[[ft_u,ft_v],[ele_u,ele_v],dis_uv]


   
    li_nodes={feature:{} for feature in li_ft}
    for node,attr in G_expand.nodes(data=True):
        if attr['feature'][-2] in li_ft:
            li_nodes[attr['feature'][-2]][tuple(attr['direct_vec'])]=node

    new_add_bond={}

 
    for key,data in bond_break.items():
        next_loop=0
        li_atom_ind=data[1].index('Li')
        ni_atom_ind=data[1].index('Ni')
        li_atom_orig=key[li_atom_ind]
        ni_atom_orig=key[ni_atom_ind]
        li_atom_ft=data[0][li_atom_ind]
        dis_orig=data[2]


        node_dir=G_expand.nodes[ni_atom_orig]['direct_vec']
        match_dir=atom_ind_dir_pair_lini[(ni_atom_orig,li_atom_orig)]
        add_vec_raw=[0,0,0]
        for vec in match_dir:
            sign = -1 if '-' in vec else 1
            axis = 'abc'.index(vec[-1])  # 'a', 'b', 或 'c'
            add_vec_raw[axis]=sign
        
        target_dir_nb=[add_vec_raw[i]+v for i,v in enumerate(node_dir)]
        for i,v in enumerate(target_dir_nb):
            if v>max(expand_vecs[i]):
                target_dir_nb[i]=min(expand_vecs[i])
                bond_type='pbc2'                
            elif v<min(expand_vecs[i]):
                target_dir_nb[i]=max(expand_vecs[i])
                bond_type='pbc2'                
        closest_av_node=li_nodes[li_atom_ft][tuple(target_dir_nb)]
        if frozenset([closest_av_node,ni_atom_orig]) in new_add_bond:
            continue
        matching_keys = [key for key in new_add_bond if ni_atom_orig in key]
        for key in matching_keys:
            if new_add_bond[key]==frozenset([li_atom_ft,G_expand.nodes[ni_atom_orig]['feature'][-2]]):
                if frozenset([closest_av_node,ni_atom_orig])!=key:
                    old_atom1=list(key)[0]
                    old_atom2=list(key)[1]
                    old_key1=list(new_add_bond[key])[0]
                    old_key2=list(new_add_bond[key])[1]
                    print("ERROR: duplicated match!",key,frozenset([closest_av_node,ni_atom_orig]),"ft:",new_add_bond[key])
                    print("Directions two pbc atom cnct:",match_dir)
                    print(f"Original atom {ni_atom_orig} direct vector:",G_expand.nodes[ni_atom_orig]['direct_vec'])
                    print(f"Exsting atom {old_atom1 if old_atom1!=node else old_atom2}direct vector:",G_expand.nodes[old_atom1 if old_atom1!=node else old_atom2]['direct_vec'])
                    print(f"New atom {closest_av_node} direct vector:",G_expand.nodes[closest_av_node]['direct_vec'])
                    print(f"Original atom {ni_atom_orig} pos:",G_expand.nodes[ni_atom_orig]['feature'][-5:-2])
                    print(f"Exsting atom {old_atom1 if old_atom1!=ni_atom_orig else old_atom2} pos:",G_expand.nodes[old_atom1 if old_atom1!=ni_atom_orig else old_atom2]['feature'][-5:-2])
                    print(f"New atom {closest_av_node} pos:",G_expand.nodes[closest_av_node]['feature'][-5:-2])
                    sys.exit(0)
                else:
                    next_loop=1
        if next_loop:
            continue

        new_add_bond[frozenset([ni_atom_orig,closest_av_node])]=frozenset([li_atom_ft,G_expand.nodes[ni_atom_orig]['feature'][-2]])

        dis_orig = calculate_real_distance(np.array(G_expand.nodes[ni_atom_orig]['feature'][-5:-2]) , np.array(G_expand.nodes[li_atom_orig]['feature'][-5:-2]))
        closest_dis = calculate_real_distance(np.array(G_expand.nodes[ni_atom_orig]['feature'][-5:-2]) , np.array(G_expand.nodes[closest_av_node]['feature'][-5:-2]))
        closest_dis_car = calculate_real_distance(np.array(cartesian_to_fractional(G_expand.nodes[ni_atom_orig]['feature'][-5:-2], orig_lattice_vectors)) , np.array(cartesian_to_fractional(G_expand.nodes[closest_av_node]['feature'][-5:-2], orig_lattice_vectors)) )
        #print("Change:",ni_atom_orig+1,li_atom_orig+1,dis_orig,"-->",ni_atom_orig+1,closest_av_node+1,closest_dis,closest_dis_car)
        #print("Change:",G_expand.nodes[ni_atom_orig]['feature'][-5:-2],G_expand.nodes[li_atom_orig]['feature'][-5:-2],dis_orig,"-->",G_expand.nodes[ni_atom_orig]['feature'][-5:-2],G_expand.nodes[closest_av_node]['feature'][-5:-2],closest_dis)

        if not G_expand.has_edge(ni_atom_orig,closest_av_node):
            G_expand.add_edge(ni_atom_orig,closest_av_node)
            copy_edge_attributes(G_expand, G_expand, ni_atom_orig,li_atom_orig, ni_atom_orig,closest_av_node)
            #print("Change:",ni_atom_orig+1,li_atom_orig+1,dis_orig,G_expand.edges[ni_atom_orig,li_atom_orig]['delta_E'] if 'delta_E' in G_expand.edges[ni_atom_orig,li_atom_orig] else 'None',"-->",ni_atom_orig+1,closest_av_node+1,closest_dis,closest_dis_car,G_expand.edges[ni_atom_orig,closest_av_node]['delta_E'] if 'delta_E' in G_expand.edges[ni_atom_orig,li_atom_orig] else 'None')
            G_expand.remove_edge(ni_atom_orig,li_atom_orig)
       # elif G_expand.has_edge(ni_atom_orig,closest_av_node):
            #print("+++KEEP:",ni_atom_orig+1,li_atom_orig+1,dis_orig,G_expand.edges[ni_atom_orig,li_atom_orig]['delta_E'] if 'delta_E' in G_expand.edges[ni_atom_orig,li_atom_orig] else 'None',"-->",ni_atom_orig+1,closest_av_node+1,closest_dis,closest_dis_car,G_expand.edges[ni_atom_orig,closest_av_node]['delta_E'] if 'delta_E' in G_expand.edges[ni_atom_orig,li_atom_orig] else 'None')

    #print("total lini:",count_nm,"change lini:", len(bond_break))
    return G_expand

def clean_graphs(graph_init,graph_in):
    """
    Creates a copy of graph_in and modifies it to only retain attributes of nodes and edges that also exist in graph_init.
    
    Parameters:
    - graph_init (nx.Graph): The reference graph with the desired node and edge attribute keys.
    - graph_in (nx.Graph): The graph from which a modified copy will be made.
    
    Returns:
    - nx.Graph: A modified copy of graph_in with limited attributes.
    """
    # 创建 graph_in 的深拷贝
    graph_in_copy = copy.deepcopy(graph_in)

    for node,attr in graph_in_copy.nodes(data=True):
        attr['feature'][-2]=node
    # 处理节点属性
    for node in graph_in_copy.nodes():
        if node in graph_init:
            # 找到图1中节点的所有属性键
            attr_keys = graph_init.nodes[node].keys()
            # 创建新的属性字典，只包含在图1中存在的属性
            new_attrs = {key: graph_in_copy.nodes[node][key] for key in attr_keys if key in graph_in_copy.nodes[node] or key=='orig_ind'}
            # 更新图2副本中的节点属性
            graph_in_copy.nodes[node].clear()
            graph_in_copy.nodes[node].update(new_attrs)

    # 处理边属性
    for edge in graph_in_copy.edges():
        if graph_init.has_edge(*edge):
            # 找到图1中边的所有属性键
            attr_keys = graph_init.get_edge_data(*edge).keys()
            # 创建新的属性字典，只包含在图1中存在的属性
            new_attrs = {key: graph_in_copy.edges[edge][key] for key in attr_keys if key in graph_in_copy.edges[edge]}
            # 更新图2副本中的边属性
            graph_in_copy.edges[edge].clear()
            graph_in_copy.edges[edge].update(new_attrs)

    return graph_in_copy 

def main_convert_expand_graph(graph_in,k_nn):
    atom_ind_dir_pair={}
    graph_test=copy.deepcopy(graph_in)

    for node,attr in graph_test.nodes(data=True):
        attr['old_ind']=attr['feature'][-2]
        attr['original_lat']=attr['feature'][-14:-5]
    graph_cleaned=copy.deepcopy(graph_test)
    '''
    for node,attr in graph_test.nodes(data=True):
        elements_list = ["Li", "O", "Mg", "Al", "Ti", "V", "Mn", "Co", "Ni", "Zr"]
        if int(attr['feature'][-2])==5:
            print(node,'*'*20)
            for neighbor in graph_test.neighbors(node):
                if graph_test.edges[node,neighbor]['edge_type']=='original':
                    print(neighbor,int(graph_test.nodes[neighbor]['feature'][-2])+1,elements_list[np.argmax(graph_test.nodes[neighbor]['feature'][:10])])
    '''
    #for nodes,attr in graph_test.nodes(data=True):
    #    attr['locate']=[]
    boundary_atoms, lattice_vectors, graph_loc = find_boundary_atoms(graph_test)
    #print("Boundary atoms:", len(boundary_atoms),boundary_atoms)
    boundary_directions,atom_ind_dir_pair = find_boundary_direction_by_fractional(boundary_atoms, graph_loc, lattice_vectors)
    #print("\nLV",lattice_vectors,"\nFT",[attr for node,attr in graph_test.nodes(data=True)][0]['feature'])
    #write_xsf("all_atoms_u.xsf", graph_loc, lattice_vectors)
    #write_xsf("boundary_atoms.xsf", graph_loc, lattice_vectors, selected_atoms=boundary_atoms)
    #write_xsf("boundary_atoms_marked.xsf", graph_loc, lattice_vectors, selected_atoms=boundary_atoms, directs=boundary_directions)
    
    expand_dir,graph_loc,further_expand=check_boundary_atom_to_chose_edge(boundary_atoms,boundary_directions,graph_loc,k_nn)
    #if not expand_dir:
    #    print('expand_dir:',expand_dir,"\nlattice_vector",lattice_vectors)
    #    print("No need for expand")
    
    exp_count=1
    
    if not further_expand and expand_dir:
        #print("*"*60,"EXPAND TIME :",exp_count,"*"*60)
        #print('expand_dir:',expand_dir,"\nlattice_vector",lattice_vectors,"\nFT")
        G_images,av_combine_directs=creat_image_graphs(graph_loc,expand_dir)
        for node,attr in graph_loc.nodes(data=True):
            attr['direct_vec'] = [0,0,0]
        G_images['O']=graph_loc
        
        '''
        pbc_count=0
        inner_count=0
        for u,v,data in graph_loc.edges(data=True):
            if data.get("edge_type")=='li_ni_edge':
                continue
            if data.get("bond_loc")=='pbc':
                pbc_count+=1
            elif data.get("bond_loc")=='merge_bond':
                inner_count+=1
            else:
                print("ERROR",u,v)
        print("PPPPPPPPPPPPPP,inner",inner_count,"pbc",pbc_count)
        
        inner_node=0
        out_node=0
        for node,attr in graph_loc.nodes(data=True):
            count_nb=0
            for nb in graph_loc.neighbors(node):
                if graph_loc.edges[node,nb]['edge_type']=="original" and graph_loc.edges[node,nb]["bond_loc"]=='merge_bond':
                    count_nb+=1
            if count_nb==6:
                #print(node,"6")
                inner_node+=1
        print("num of inner node:",inner_node)
        print("+*"*60)
        print("+*"*60)
        '''
        G_expand,exp_vecs = connect_graphs(G_images, expand_dir, atom_ind_dir_pair)
        #print("="*60)
        #print("="*60)
        #print("="*60)
        G_expand=handle_other_manual_edges(G_expand,exp_vecs,'edge_type')
        G_exp=G_expand.copy()
        #sys.exit(0)
        G_exp=clean_graphs(graph_cleaned,G_exp)
        
        #for node,attr in G_exp.nodes(data=True):
        #    attr['atom_loc']=''
        boundary_atoms, lattice_vectors,graph_loc2 = find_boundary_atoms(G_exp,prefix='Supercell')
        #print("Boundary atoms:", len(boundary_atoms),boundary_atoms)
        #print('AV',expand_dir,"\nLV",lattice_vectors,"\nFT",[attr for node,attr in G_expand.nodes(data=True)][0]['feature'])
        boundary_directions,_ = find_boundary_direction_by_fractional(boundary_atoms, graph_loc2, lattice_vectors)
        #write_xsf("all_atoms.xsf", G_expand, lattice_vectors)
        graph_loc=copy.deepcopy(graph_loc2)
        expand_dir,graph_loc,further_expand=check_boundary_atom_to_chose_edge(boundary_atoms,boundary_directions,graph_loc2,k_nn,prefix=f'Supercell_time_{exp_count}')
        #print("Need Futher Expand:",further_expand if further_expand else "No")
        exp_count+=1 
        #print("*"*60,"*"*60)
    
    elif further_expand!=[] and expand_dir:
        while expand_dir!=[]:
            #print("*"*60,"EXPAND TIME :",exp_count,"*"*60)
    
            G_images,av_combine_directs=creat_image_graphs(graph_loc,expand_dir)
            for node,attr in graph_loc.nodes(data=True):
                attr['direct_vec'] = [0,0,0]
            G_images['O']=graph_loc
            
            G_expand,exp_vecs = connect_graphs(G_images, expand_dir, atom_ind_dir_pair)
            G_expand=handle_other_manual_edges(G_expand,exp_vecs,'edge_type')
            G_exp=G_expand.copy()
            
            #for node,attr in G_exp.nodes(data=True):
                #attr['atom_loc']=''
    
            G_exp=clean_graphs(graph_cleaned,G_exp)
    
            boundary_atoms, lattice_vectors,graph_loc2 = find_boundary_atoms(G_exp,prefix='Supercell')
            #print('expand_dir:',expand_dir,"\nlattice_vector",lattice_vectors,"\nFT")
            boundary_directions,atom_ind_dir_pair = find_boundary_direction_by_fractional(boundary_atoms, graph_loc2, lattice_vectors)
    
            expand_dir,graph_loc,further_expand=check_boundary_atom_to_chose_edge(boundary_atoms,boundary_directions,graph_loc2,k_nn,prefix=f'Supercell_time_{exp_count}')
            #print("Need Futher Next Expand:",further_expand if further_expand else "No")
            exp_count+=1 
            #print("*"*60,"*"*60)
    
    write_xsf("boundary_atoms.xsf", graph_loc, lattice_vectors, selected_atoms=boundary_atoms)
    write_xsf("boundary_atoms_marked.xsf", graph_loc, lattice_vectors, selected_atoms=boundary_atoms, directs=boundary_directions)
    for node,attr in graph_loc.nodes(data=True):
        attr['feature'][-2]=attr['old_ind']
    return graph_loc

def process_subgraph_expand(G,k):
    subgraph = main_convert_expand_graph(G,k)
    return subgraph

def convert_subgraphs_expand(graphs,k):
    with ProcessPoolExecutor() as executor:
        # 创建future到索引的映射
        futures = {executor.submit(process_subgraph_expand, graph,k): i for i, graph in enumerate(graphs)}
        converted_graphs = [None] * len(graphs)  # 预先分配结果列表
        total_done = 0  # 已完成的任务数量
        
        while total_done < len(futures):
            done_futures = [f for f in futures if f.done()]  # 获取所有已完成的futures
            for future in done_futures:
                index = futures[future]  # 获取原始图的索引
                if converted_graphs[index] is None:  # 检查是否已更新进度
                    result = future.result()
                    converted_graphs[index] = result if result is not None else None
                    total_done += 1
                    print_progress(total_done, len(graphs))  # 打印进度
            sleep(0.1)  # 稍微等待以减少CPU使用率
    

    #return [subgraph for subgraphs in converted_graphs for subgraph in subgraphs if subgraph is not None]
    return [subgraph for subgraph in converted_graphs if subgraph is not None]
def get_neighbors_xty(graph, node, depth):
    get_nodes = [node]
    node_buffer = []
    for i in range(depth):
        for node in get_nodes:
            for neighbor in graph.neighbors(node):
                if graph.edges[node, neighbor]['edge_type'] == 'original':
                    node_buffer.append(neighbor)
        node_buffer = list(set(node_buffer))
        get_nodes.extend(node_buffer)
    get_nodes.remove(node)
    return set(get_nodes)


def decode_lattice_and_coords(vector):
    """
    从12维向量中解码晶格矩阵和坐标。
    """
    lattice_matrix = np.array(vector[-12:-3]).reshape((3, 3))
    coords = np.array(vector[-3:])
    return lattice_matrix, coords

def calculate_minimum_distance(vector1, vector2):
    """
    计算考虑周期性边界条件的两个原子之间的最短距离。
    """
    lattice_matrix, coords1 = decode_lattice_and_coords(vector1)
    _, coords2 = decode_lattice_and_coords(vector2)
    
    min_distance = np.inf
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                # 计算相邻晶胞中原子B的坐标
                translated_coords = coords2 + i * lattice_matrix[0, :] + j * lattice_matrix[1, :] + k * lattice_matrix[2, :]
                # 计算与固定原子A的距离
                distance = np.linalg.norm(coords1 - translated_coords)
                # 更新最小距离
                if distance < min_distance:
                    min_distance = distance
    return min_distance

def networkx_to_pyg_graph(subgraph,k_hop_attention):
    
    # Convert to zero-based indexing重新标记节点索引：为了确保edge_index中的索引是零起始和连续的，我添加了一个映射来重新标记您的networkx图中的节点。这个映射将每个节点映射到一个新的索引（从0开始）
    mapping = {node: i for i, node in enumerate(subgraph.nodes())}
    subgraph = nx.relabel_nodes(subgraph, mapping)

    range1_lat_coord=-14
    range2_lat_coord=-2

    edge_distances = []
    for u, v, data in subgraph.edges(data=True):
        u_vec=subgraph.nodes[u]['feature'][range1_lat_coord:range2_lat_coord]  
        v_vec=subgraph.nodes[v]['feature'][range1_lat_coord:range2_lat_coord]
        distance = calculate_minimum_distance(u_vec,v_vec)
        edge_distances.append([distance])
    
    #print(edge_distances)
    edge_attr = torch.tensor(edge_distances, dtype=torch.float)
    
    # 提取节点特征
    node_features = [subgraph.nodes[node]['feature'][:27]for node in subgraph.nodes()]
    # 创建归一化器实例
    scaler = MinMaxScaler()
    # 拟合数据并转换
    node_features_normalized = scaler.fit_transform(node_features)
    x = torch.tensor(node_features_normalized, dtype=torch.float)

    # 提取边索引和边类型
    edge_indices = []
    edge_types = []
    for u, v, data in subgraph.edges(data=True):
        edge_indices.append((u, v))
        # 假设 'original' 边为类型 0，'li_ni_edge' 为类型 1
        edge_type = 1 if data.get('edge_type') == 'li_ni_edge' else 0
        edge_types.append(edge_type)

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_types, dtype=torch.long)

    # 计算k跳邻居并标记节点
    for u, v, data in subgraph.edges(data=True):
        if data.get('edge_type') == 'li_ni_edge':
            k_neighbors_u = get_neighbors_xty(subgraph, u, k_nn)
            k_neighbors_v = get_neighbors_xty(subgraph, v, k_nn)
            collected_nodes = k_neighbors_u.union(k_neighbors_v)       

    def get_k_hop_neighbors(graph, node, depth=k_hop_attention):
        get_nodes=[node]
        node_buffer=[]
        for i in range(1,depth+1):
            #print(i,"depth")
            for node in get_nodes:
                #print(node,"node")
                for neighbor in graph.neighbors(node):
                    if graph.edges[node,neighbor]['edge_type']=='original':
                        #if depth==1:
                            #print(neighbor,"select_neighbor")
                        node_buffer.append(neighbor)
            node_buffer=list(set(node_buffer))
            get_nodes.extend(node_buffer)
            
        return set(get_nodes)
            
    # 查找包含能量信息的 'li_ni_edge'
    exchange_energy = None
    li_ni_pairs=None
    additional_edge_indices = []  # 存储额外边的索引
    additional_edge_distances = []  # 存储额外边的距离属性
    printed_neighbors_info = False
    
    for u, v, data in subgraph.edges(data=True):
        if data.get('edge_type') == 'li_ni_edge':
            exchange_energy = data.get('delta_E', None)
            #li_ni_pair=[u,v]# 保存 li_ni_edge 边两端节点的索引
            li_ni_pairs = [[u, v]]
            
            # 获取u和v的k近邻
            k_hop_neighbors_u = get_k_hop_neighbors(subgraph, u, k_hop_attention)
            k_hop_neighbors_v = get_k_hop_neighbors(subgraph, v, k_hop_attention)
            
            # 对每个节点，添加到u和v的额外边
            for node in subgraph.nodes():
                if node != u and node != v:  # 排除u和v自身
                    # 添加到u的边
                    if node in k_hop_neighbors_u:
                        u_vec = subgraph.nodes[u]['feature'][range1_lat_coord:range2_lat_coord]
                        node_vec = subgraph.nodes[node]['feature'][range1_lat_coord:range2_lat_coord]
                        distance = calculate_minimum_distance(u_vec, node_vec)
                        additional_edge_indices.append([node, u])
                        additional_edge_distances.append([distance])

                    # 添加到v的边
                    if node in k_hop_neighbors_v:
                        v_vec = subgraph.nodes[v]['feature'][range1_lat_coord:range2_lat_coord]
                        node_vec = subgraph.nodes[node]['feature'][range1_lat_coord:range2_lat_coord]
                        distance = calculate_minimum_distance(v_vec, node_vec)
                        additional_edge_indices.append([node, v])
                        additional_edge_distances.append([distance])
            break


    # 将能量信息转换为 PyTorch 张量
    li_ni_pairs_tensor = torch.tensor(li_ni_pairs, dtype=torch.long)
    k_neighbors=[list(collected_nodes)]
    k_neighbors_tensor = torch.tensor(k_neighbors, dtype=torch.long)
    u_nb=torch.tensor([list(k_neighbors_u)],dtype=torch.long)
    v_nb=torch.tensor([list(k_neighbors_v)],dtype=torch.long)
    
    # 记录每个图的节点数量
    num_nodes = len(subgraph.nodes())

    # 添加额外边的属性
    edge_distances.extend(additional_edge_distances)
    edge_attr = torch.tensor(edge_distances, dtype=torch.float)

    # 添加额外边的索引
    edge_indices.extend(additional_edge_indices)
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    
    # 创建 PyTorch Geometric Data 对象
    data = Data(x=x, edge_index=edge_index, edge_type=edge_type, edge_attr=edge_attr,  li_ni_pairs=li_ni_pairs_tensor, k_neighbors=k_neighbors_tensor, u_nb=u_nb, v_nb=v_nb, num_nodes=num_nodes)
    return data


def set_seed(seed):
    #"""设置全局随机种子以确保实验的可复现性"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class CustomGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=None, concat=False):
        super(CustomGATConv, self).__init__(aggr='add')  # 使用加法聚合
        #self.linear = torch.nn.Linear(in_channels, out_channels)  # 用于特征转换
        #self.att = torch.nn.Parameter(torch.Tensor(1, 2 * out_channels))  # 注意力系数参数
        
        self.concat=concat       
        
        if heads==None:
            heads=1
        
        self.heads = heads
        
        # 为每个头创建一个独立的线性变换层和注意力向量
        self.linears = torch.nn.ModuleList([
            torch.nn.Linear(in_channels, out_channels) for _ in range(heads)
        ])
        self.atts = torch.nn.ParameterList([
            torch.nn.Parameter(torch.Tensor(1, 2 * out_channels)) for _ in range(heads)
        ])        
        
        self.reset_parameters()

    #def reset_parameters(self):
    #    torch.nn.init.xavier_uniform_(self.linear.weight)
    #    torch.nn.init.xavier_uniform_(self.att)
    #def forward(self, x, edge_index, edge_attr):
    #    x = self.linear(x)
    #    self.x=x
    #    return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def reset_parameters(self):
        for linear in self.linears:
            torch.nn.init.xavier_uniform_(linear.weight)
        for att in self.atts:
            torch.nn.init.xavier_uniform_(att)
    
    def forward(self, x, edge_index, edge_attr):
        # 分别对每个头计算注意力和转换后的特征
        x_heads = [linear(x) for linear in self.linears]
        
        # 将每个头的结果分别传递给propagate并聚合
        outs = [self.propagate(edge_index, x=x_head, edge_attr=edge_attr, head_idx=i)
                for i, x_head in enumerate(x_heads)]
        
        # 聚合所有头的结果（这里使用拼接）
        if self.concat==False:
            self.x = torch.stack(outs, dim=0).mean(dim=0)
        else:    
            self.x = torch.cat(outs, dim=-1)
        
        return self.x
    
    def message(self, edge_index_i, x_i, x_j, edge_attr, head_idx):
        # 使用对应头的注意力参数
        attention = self.atts[head_idx]
        x_j_cat = torch.cat([x_i, x_j], dim=1)
        attention_score = torch.matmul(x_j_cat, attention.t())
        attention_score = F.leaky_relu(attention_score)

        edge_attr_inv_sq = 1.0 / (edge_attr + 1e-7)**2
        edge_attr_inv_sq = edge_attr_inv_sq.view(-1, 1)

        weighted_message = x_j * edge_attr_inv_sq
        x_j_transformed = attention_score * weighted_message
        return x_j_transformed
    
    def message_gcn(self, x_j, edge_attr):
        # 现在我们只使用边属性进行加权，不计算注意力分数
        edge_attr_inv_sq = 1.0 / (edge_attr + 1e-7)
        return x_j * edge_attr_inv_sq.view(-1, 1)    
    
    #def message(self, edge_index_i, x_i, x_j, edge_attr):
        # 计算注意力系数        
        #print("x_j.shape=",x_j.shape)
        #print("x_i.shape=",x_i.shape)        
    #    x_j_cat = torch.cat([x_i, x_j], dim=1)  # 将源节点和目标节点的特征拼接        
        #print("x_j_cat.shape=",x_j_cat.shape)        
    #    attention = torch.matmul(x_j_cat, self.att.t())        
        #print("self.att.t().shape=",self.att.t().shape)        
    #    attention = F.leaky_relu(attention)
        # 将edge_attr用作门控机制
    #    edge_attr_inv_sq = 1.0 / (edge_attr + 1e-7)**2  # 依然计算距离的平方倒数
    #    edge_attr_inv_sq = edge_attr_inv_sq.view(-1, 1)  # 确保尺寸匹配
        #print("edge_att.shape=",edge_attr_inv_sq.shape)        
        # 使用edge_attr加权信息
    #    weighted_message = x_j * edge_attr_inv_sq  # 使用edge_attr调整信息流的权重
    #    x_j_transfrom = attention * weighted_message
        #print("x_j_transfrom.shape",x_j_transfrom.shape)
        #print("self.x.shape",self.x.shape)
        # 应用注意力系数和门控后的信息加权结果
    #    return x_j_transfrom   
    
    def update(self, aggr_out):
        #print("aggr_out.shape=",aggr_out.shape)
        #aggr_out+=self.x
        #aggr_out=F.relu(aggr_out)
        return aggr_out
        #return F.relu(aggr_out)
    
    

class GAT(torch.nn.Module):
    def __init__(self, num_node_features,gat_heads):
        super(GAT,self).__init__()
        #super(GAT, self).__init__()
        # 定义GAT卷积层
        self.conv1 = CustomGATConv(num_node_features, num_node_features, heads=gat_heads, concat=True)
        self.conv2 = CustomGATConv(num_node_features * gat_heads, num_node_features, heads=gat_heads, concat=True)
        self.conv3 = CustomGATConv(num_node_features * gat_heads, num_node_features, heads=gat_heads, concat=True)
        self.conv4 = CustomGATConv(num_node_features * gat_heads, num_node_features, heads=gat_heads, concat=True)
        self.conv5 = CustomGATConv(num_node_features * gat_heads, num_node_features, heads=gat_heads, concat=True)
        self.conv6 = CustomGATConv(num_node_features * gat_heads, num_node_features, heads=gat_heads, concat=True)
        self.conv_f = CustomGATConv(num_node_features * gat_heads, num_node_features, heads=gat_heads, concat=False)
        
        self.bn1 = torch.nn.BatchNorm1d(num_node_features * gat_heads, eps=gat_eps)  # BN for conv1 output,eps为了防止除以0
        self.bn2 = torch.nn.BatchNorm1d(num_node_features * gat_heads, eps=gat_eps)  # BN使用时需要在激活函数之前 
        self.bn3 = torch.nn.BatchNorm1d(num_node_features * gat_heads, eps=gat_eps)  
        self.bn4 = torch.nn.BatchNorm1d(num_node_features * gat_heads, eps=gat_eps)
        self.bn5 = torch.nn.BatchNorm1d(num_node_features * gat_heads, eps=gat_eps)  
        self.bn6 = torch.nn.BatchNorm1d(num_node_features * gat_heads, eps=gat_eps)
        
        self.hidden_layer1 = torch.nn.Linear(num_node_features , num_node_features*2 )
        self.hidden_layer2 = torch.nn.Linear(num_node_features*2 , num_node_features*2 )
        self.hidden_layer3 = torch.nn.Linear(num_node_features*2 , num_node_features*2 )
        self.hidden_layer4 = torch.nn.Linear(num_node_features*2 , num_node_features*2 )
        
        self.bn_hidden1 = torch.nn.BatchNorm1d(num_node_features , eps=gat_eps)
        self.bn_hidden2 = torch.nn.BatchNorm1d(num_node_features * 2, eps=gat_eps)
        self.bn_hidden3 = torch.nn.BatchNorm1d(num_node_features * 2, eps=gat_eps)
        self.bn_hidden4 = torch.nn.BatchNorm1d(num_node_features * 2, eps=gat_eps)
        
        self.predictor = torch.nn.Linear(num_node_features*2, 1)

        #self.first_pair_neighbors_printed = False
        
        self.num_element_types = 10  # 假定有10种不同的元素
        self.embedding_dim = 10  # 假定每个元素的嵌入维度为10
        self.element_embedding = torch.nn.Embedding(self.num_element_types, self.embedding_dim)
        self.reset_parameters()

    def reset_parameters(self):
        # 重置GAT卷积层权重
        for conv in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv_f]:
            conv.reset_parameters()
        
        # 重置全连接层权重
        for linear in [self.hidden_layer1, self.hidden_layer2, self.hidden_layer3, self.predictor]:
            torch.nn.init.xavier_uniform_(linear.weight)
            linear.bias.data.fill_(0)
        
        
    def forward(self, x, edge_index, edge_attr, batch, li_ni_indices, k_neighbors, u_nb, v_nb, num_nodes):
        # 应用GAT卷积层
        x1 = self.conv1(x, edge_index, edge_attr) 
        x1 = F.relu(self.bn1(x1))
        #x1 = F.relu(x1)
        x2 = self.conv2(x1, edge_index, edge_attr)
        x2 = F.relu(self.bn2(x2))
        #x2 = F.relu(x2)
        x3 = self.conv3(x2, edge_index, edge_attr) + self.conv3(x1, edge_index, edge_attr)  
        x3 = F.relu(self.bn3(x3))
        #x3 = F.relu(x3)
        x4 = self.conv4(x3, edge_index, edge_attr)   
        x4 = F.relu(self.bn4(x4))
        x5 = self.conv5(x4, edge_index, edge_attr)
        x5 = F.relu(self.bn5(x5))
        x6 = self.conv6(x5, edge_index, edge_attr) + self.conv6(x4, edge_index, edge_attr)  
        x6 = F.relu(self.bn6(x6))
        x_f = self.conv_f(x4, edge_index, edge_attr)  
        # 计算每个图的起始索引
        batch_size = batch.max() + 1
        # 假设每个图的节点数量是固定的
        num_nodes_per_graph = num_nodes/batch_size  # 每个图的节点数量
        
        start_idx = torch.arange(0, batch_size * num_nodes_per_graph, num_nodes_per_graph, device=x.device)
        # 计算每条边属于哪个图
        graph_ids = torch.tensor([i for i in range(batch_size)])
        #print("Graph IDs for each edge:", graph_ids)

        # 调整li_ni_indices的索引
        adjusted_li_ni_indices = li_ni_indices + start_idx[graph_ids].unsqueeze(1)
        adjusted_u_nb = u_nb + start_idx[graph_ids].unsqueeze(1)
        adjusted_v_nb = v_nb + start_idx[graph_ids].unsqueeze(1)
        adjusted_k_neighbors = k_neighbors + start_idx[graph_ids].unsqueeze(1)
        # 计算每个li_ni_edge边两端的k近邻的平均特征
        k_hop_features = []
        for i, (li, ni) in enumerate(adjusted_li_ni_indices):
            # 获取li和ni的k跳邻居
            li_neighbors = adjusted_u_nb[i]
            ni_neighbors = adjusted_v_nb[i]
            k_neighbors_i = adjusted_k_neighbors[i]
            # 打印第一个li_ni对的邻居个数
            #if not self.first_pair_neighbors_printed:
            #    print("Li-Ni_indices:",li,ni)
            #    print("Number of neighbors for li:", len(li_neighbors),li_neighbors)
            #    print("Number of neighbors for ni:", len(ni_neighbors),ni_neighbors)
            #    print("K_neighbors:",len(k_neighbors_i),k_neighbors_i)
            #    self.first_pair_neighbors_printed = True

            # 获取邻居节点的特征并计算平均
            #li_neighbor_features = x_f[li_neighbors.long()].mean(dim=0)
            #ni_neighbor_features = x_f[ni_neighbors.long()].mean(dim=0)
            li_neighbor_features = x_f[li.long()]
            ni_neighbor_features = x_f[ni.long()]
            
            #聚合li和ni的邻居特征
            edge_features = (li_neighbor_features + ni_neighbor_features) / 2
            
            #li_tensor = li_neighbor_features.clone().detach()
            #ni_tensor = ni_neighbor_features.clone().detach()
            # Concatenate along a new dimension (e.g., dim=0)
            #edge_features = torch.cat((li_tensor, ni_tensor), dim=0)
 
            
            k_hop_features.append(edge_features)
        
        k_hop_features_tensor = torch.stack(k_hop_features, dim=0)
        # 使用预测器进行预测
        # 使用预测器进行预测
        # 预激活：先应用BN，然后是ReLU激活函数
        x_hidden1 = self.bn_hidden1(k_hop_features_tensor)
        x_hidden1 = F.relu(x_hidden1)
        x_hidden1 = self.hidden_layer1(x_hidden1)
        # 注意，这里没有在ReLU和第一个全连接层之间应用BN

        # 对于第二个隐藏层，我们也采用预激活的方式
        x_hidden2 = self.bn_hidden2(x_hidden1)
        x_hidden2 = F.relu(x_hidden2)
        x_hidden2 = self.hidden_layer2(x_hidden2) + x_hidden1
        # 注意，在x_hidden2的计算中加入了x_hidden1进行残差连接

        # 第三个隐藏层同样使用预激活方式
        x_hidden3 = self.bn_hidden3(x_hidden2)
        x_hidden3 = F.relu(x_hidden3)
        x_hidden3 = self.hidden_layer3(x_hidden3) + x_hidden2
        # 同样，在x_hidden3的计算中加入了x_hidden2进行残差连接

        x_hidden4 = self.bn_hidden4(x_hidden3)
        x_hidden4 = F.relu(x_hidden4)
        x_hidden4 = self.hidden_layer4(x_hidden4) + x_hidden3
        # 最后应用预测层得到最终的输出
        energy_prediction = self.predictor(x_hidden4)
        #energy_prediction = self.predictor(k_hop_features_tensor)
        return energy_prediction
        


def extract_params_from_path(model_path):
    # 使用正则表达式匹配所需的参数
    seed_match = re.search(r'seed_(\d+)', model_path)
    k_hop_attention_match = re.search(r'k(\d+)-FC', model_path)
    gat_heads_match = re.search(r'heads_(\d+)', model_path)
    r2_match = re.search(r'R2_([\d.]+)', model_path)
    opt_match = re.search(r'opt_(\w+).pth', model_path)
    
    # 提取匹配到的参数值，如果没有匹配到则返回None
    seed = int(seed_match.group(1)) if seed_match else None
    k_hop_attention = int(k_hop_attention_match.group(1)) if k_hop_attention_match else None
    gat_heads = int(gat_heads_match.group(1)) if gat_heads_match else None
    r2 = float(r2_match.group(1)) if r2_match else None
    opt = opt_match.group(1) if opt_match else None
    
    return seed, k_hop_attention, gat_heads, r2, opt


def process_subgraph(subgraph,k_hop_attention):
    pyg_graph = networkx_to_pyg_graph(subgraph,k_hop_attention)
    return pyg_graph

def print_progress(done, total):
    #假设trains(done, total):
    percent_done = done / total * 100
    bar_length = int(percent_done / 100 * 60)
    bar = "[" + "#" * bar_length + "-" * (60 - bar_length) + "]" + f"{percent_done:.2f}%" + f"   {done}/{total}"
    print(bar, "\r", end='')

def convert_graphs(graphs,k_hop_attention):
    with ProcessPoolExecutor() as executor:
        # 创建future到索引的映射
        futures = {executor.submit(process_subgraph, subgraph,k_hop_attention): i for i, subgraph in enumerate(graphs)}
        converted_graphs = [None] * len(graphs)  # 预先分配结果列表
        total_done = 0  # 已完成的任务数量

        while total_done < len(futures):
            done_futures = [f for f in futures if f.done()]  # 获取所有已完成的futures
            for future in done_futures:
                index = futures[future]  # 获取原始图的索引
                if converted_graphs[index] is None:  # 检查是否已更新进度
                    result = future.result()
                    converted_graphs[index] = result if result is not None else None
                    total_done += 1
                    print_progress(total_done, len(graphs))  # 打印进度
            sleep(0.1)  # 稍微等待以减少CPU使用率

    return [graph for graph in converted_graphs if graph is not None]

def predict_from_model(model_path, predict_graphs,batch_size):
    global init_info_print 
    start_time=time.time()
    print("#"*60)
    print(f"Predict with model:{model_path}") 
    seed,k_hop_attention,gat_heads,r2,opt=extract_params_from_path(model_path)
    print(f"R2:{r2},k_hop_attention:{k_hop_attention},gat_heads:{gat_heads},opt:{opt},seed:{seed}") 
    #if opt=="True":
    #    print("!!! Using Opimizing Model, Predict Energy From Opimized Structure Calculated by QE relax !!!")
    #    print("!!! Using Opimizing Model, Predict Energy From Opimized Structure Calculated by QE relax !!!")
    #    print("!!! Using Opimizing Model, Predict Energy From Opimized Structure Calculated by QE relax !!!")
    #elif opt=="False":
    #    print("!!! Using From Scratch Model, Predict Energy From R/3m LiNiO2 Lattice Site Structure !!!")
    #    print("!!! Using From Scratch Model, Predict Energy From R/3m LiNiO2 Lattice Site Structure !!!")
    #    print("!!! Using From Scratch Model, Predict Energy From R/3m LiNiO2 Lattice Site Structure !!!")

    #if None in [seed,k_hop_attention,gat_heads,r2,opt]:
    if None in [k_hop_attention,gat_heads,r2,opt]:
        print(f"{model_path} is not in standard format!!")
        return None
    set_seed(seed if seed is not None else 42)
    #_graphs和test_graphs已经被定义
    print("Converting networks predict graphs into pyg formats...")
    predict_dataset = convert_graphs(predict_graphs,k_hop_attention)
    print()
    # 假设你已经有了一个torch_geometric的Dataset对象dataset
    # 设置 DataLoader的worker_init_fn
    g_data = torch.Generator()
    g_data.manual_seed(0)  # 为了确保每次都能复现相同的数据加载顺序，设置一个固定的种子
    predict_loader = DataLoader(predict_dataset, batch_size=batch_size, shuffle=False, num_workers=2, worker_init_fn=seed_worker,generator=g_data)#num_workers是并行处理数据的子进程数量

    print("#"*60+"Dataset visualizing"+"#"*60)
    print(len(predict_graphs))
    print(predict_graphs[0])
    #init_info_print=True
    # 获取前三个样本
    first_three_samples = []
    for i, data in enumerate(predict_loader):
        if i >= 1: 
            break
        first_three_samples.append(data)
    
    # 检查每个样本
    for i, data in enumerate(first_three_samples):
        
        # 模拟模型的 forward 过程
        x, edge_index, edge_type, edge_attr, batch, li_ni_indices, k_neighbors, u_nb, v_nb, num_nodes = (
            data.x, data.edge_index, data.edge_type, data.edge_attr, data.batch, data.li_ni_pairs, 
            data.k_neighbors, data.u_nb, data.v_nb, data.num_nodes
        )
    
        batch_size = batch.max() + 1
        num_nodes_per_graph = num_nodes // batch_size  # 每个图的节点数量
        start_idx = torch.arange(0, batch_size * num_nodes_per_graph, num_nodes_per_graph, device=x.device)
        # 计算每条边属于哪个图
        graph_ids = torch.tensor([i for i in range(batch_size)])
    
    
        # 调整li_ni_indices的索引
        adjusted_li_ni_indices = li_ni_indices + start_idx[graph_ids].unsqueeze(1)
    
    print("Node X shape:", x.shape)
    print("Edge_index shape:", edge_index.shape)  # 打印edge_index的维度和内容
    print("Predict_dataset:",len(predict_dataset))
    print("Predict_loader:",len(predict_loader))
    print(predict_dataset[0])
    print(f"batch_size: {batch_size}")
    print("#"*60+"Dataset visualizing"+"#"*60)

    # 加载模型
    num_node_features = len(predict_dataset[0].x[0])
    model = GAT(num_node_features,gat_heads)  # 确保这里的模型架构与训练时相同
    model_state = torch.load(model_path)# 加载模型状态
    model.load_state_dict(model_state)# 应用状态字典
    model.eval()# 设置为评估模式
    
    predictions = []

    round_nn=0
    with torch.no_grad():
        for data in predict_loader:
            print(f"predict: {round_nn}/{len(predict_loader)}","\r",end='')
            out = model(data.x, data.edge_index, data.edge_attr, data.batch, data.li_ni_pairs, data.k_neighbors, data.u_nb, data.v_nb, data.num_nodes)
            predictions.extend(out.squeeze().tolist())
            round_nn+=1

    end_time=time.time()
    run_time = round(end_time-start_time)
    run_time_total = round(end_time - start_time_total)
    hour = run_time//3600
    minute = (run_time-3600*hour)//60
    second = run_time-3600*hour-60*minute
    hour_total = run_time_total//3600
    minute_total = (run_time_total-3600*hour_total)//60
    second_total = run_time_total-3600*hour_total-60*minute_total
    print(f"**Time cost on this Model:00:{minute}:{second}, Total:{hour_total}:{minute_total}:{second_total}")
    print("#"*60)

    return predictions

def find_center_atoms(subgraph):
    # 假设中心原子是连接li_ni_edge的两个原子
    for u, v, data in subgraph.edges(data=True):
        if data.get('edge_type') == 'li_ni_edge':
            return u, v
    return None, None
def ext_subgraph_infos(predict_graphs):
    print("Extracting dataset infos...")
    # 提取数据结构来源，Li和Ni索引，计算标准差
    results = []
    for i, subgraph in enumerate(predict_graphs):
        u, v = find_center_atoms(subgraph)
        if u is None or v is None:
            continue  # 忽略没有li_ni_edge的图
        #for subgraph in predict_graphs:
        # 获取子图中的第一个节点标识符
        first_node_id = next(iter(subgraph))
        #break

        graph_name = subgraph.nodes[first_node_id]['feature'][-1]
        results.append([i, graph_name, u+1, v+1])
    return results

def process_graph(G,k):
    subgraphs=extract_subgraphs_with_periodic_structure(G, k)
    return subgraphs

def ext_subgraphs(graphs,k):
    with ProcessPoolExecutor() as executor:
        # 创建future到索引的映射
        futures = {executor.submit(process_graph, graph,k): i for i, graph in enumerate(graphs)}
        converted_graphs = [None] * len(graphs)  # 预先分配结果列表
        total_done = 0  # 已完成的任务数量
        
        while total_done < len(futures):
            done_futures = [f for f in futures if f.done()]  # 获取所有已完成的futures
            for future in done_futures:
                index = futures[future]  # 获取原始图的索引
                if converted_graphs[index] is None:  # 检查是否已更新进度
                    result = future.result()
                    converted_graphs[index] = result if result is not None else None
                    total_done += 1
                    print_progress(total_done, len(graphs))  # 打印进度
            sleep(0.1)  # 稍微等待以减少CPU使用率
    

    return [subgraph for subgraphs in converted_graphs for subgraph in subgraphs if subgraph is not None]

def print_time(start_time,name):
    end_time=time.time()
    run_time = round(end_time-start_time)
    run_time_total = round(end_time - start_time_total)
    hour = run_time//3600
    minute = (run_time-3600*hour)//60
    second = run_time-3600*hour-60*minute
    hour_total = run_time_total//3600
    minute_total = (run_time_total-3600*hour_total)//60
    second_total = run_time_total-3600*hour_total-60*minute_total
    print(f"**Time cost on this {name}:{hour}:{minute}:{second}, Total:{hour_total}:{minute_total}:{second_total}")
    time_log_i=f"**Time cost on this {name}:{hour}:{minute}:{second}, Total:{hour_total}:{minute_total}:{second_total}"
    return end_time,time_log_i



#Load model and model-infos:
params = extract_params_from_path(model_path)
if params is None:
    print(f"ERROR: No parameters extracted for model_path: {model_path}")
    sys.exit(0)
# Unpack the parameters only if they are not None.
seed, k_hop_attention, gat_heads, r2, opt = params
print(f"Model:   R2:{r2},k_hop_attention:{k_hop_attention},gat_heads:{gat_heads},opt:{opt},seed:{seed}")
print(f"Dataset: K:{k_predict}")
if k_hop_attention>k_predict:
    print(f"ERROR: The K-th neighbor training the model({k_hop_attention}) is larger than the chosed subgraph dataset({k_predict})!")
    sys.exit(0)


time_log=[]
#Convert xsf structure 2 wholegraph
if os.path.isfile(whole_db_name):
    print("#"*60+f"{whole_db_name} found! Reading wholegraph...")
    all_graphs_dict = get_all_wholegraphs(whole_db_name)
    wholegraph=[graph for name,graph in all_graphs_dict.items()][0] # 0 for only one structure
    whole_graph_time,time_i=print_time(start_time_total,"Whole_graph")
    time_log.append(time_i)
else:
    print("#"*60+" Convert xsf into wholegraph...")
    wholegraph=create_whole_graphs(str_name)
    whole_graph_time,time_i=print_time(start_time_total,"Whole_graph")
    print(f"\nStoring whole graphs in {whole_db_name}...")
    store_wholegraphs_in_db([[str_name_base,wholegraph]],whole_db_name) 
    time_log.append(time_i)

print("Expanding whole graphs...")
wholegraph2=convert_subgraphs_expand([wholegraph],k_predict)
print("Expanding whole graphs Done!")
#Extract subgraphs from whole graph
print("#"*60+" Extracing subgraphs from wholegraph...")

for k in range(2,6):
    predict_graphs_i = ext_subgraphs(wholegraph2,k) 
    sub_db_name_i=sub_db_name.replace(".db",f"_k_{k}.db")
    print(f"\nStoring subgraphs in {sub_db_name_i}...")
    store_subgraphs_in_db(predict_graphs_i,sub_db_name,k)
    if k == k_predict:
        print(f"!!!Chosing k={k_predict} for predicting!!!")
        predict_graphs=predict_graphs_i
num_subgraphs=len(predict_graphs)
print(f"{num_subgraphs} subgraphs detected!")
ext_subgraph_time,time_i=print_time(whole_graph_time,"Ext_subgraph")
time_log.append(time_i)
#Extract some li ni infos and so on  
print("#"*60+" Extracing subgraphs li ni infos...")
dataset_infos=ext_subgraph_infos(predict_graphs)
ext_info_time,time_i=print_time(ext_subgraph_time,"Ext_info")
time_log.append(time_i)
#Make predictions using model
print("#"*60+" Making predictions...")
predictions=predict_from_model(model_path,predict_graphs,batch_size)
predict_time,time_i=print_time(ext_info_time,"Predicting")
time_log.append(time_i)
output_data = []
#add model infos
output_data.append(f"#Model:   R2:{r2},k_hop_attention:{k_hop_attention},gat_heads:{gat_heads},opt:{opt},seed:{seed}\n")
output_data.append('Index\tStructure\tatom1\tatom2\tPre_of_Model\n')
for i,v in enumerate(dataset_infos):
    output_data.append('\t'.join([str(x) for x in v]) + "\t" + f"{predictions[i]:.6f}" + "\n")

f1=open(f"{predict_path}/output_{str_name_base}_prd_num_{num_subgraphs}.csv","w+")
for i in output_data:
    f1.writelines(i)
f1.close()
print("$"*60)
print("#"*30+"Below Files Has Created"+"#"*30)
print(f"{predict_path}/output_{str_name_base}_prd_num_{num_subgraphs}.csv created!(Store all the predictions)")

all_time_cost,time_i=print_time(start_time_total,"This Struture Total Prediction")
time_log.append(time_i)

f1=open(f"{predict_path}/timelog.log","w+")
for i in time_log:
    f1.writelines(i+"\n")
f1.close()
print("$"*60)
print("#"*30+"Below Files Has Created"+"#"*30)
print(f"{predict_path}/timelog.log created!(Store all the times cost each step)")

# Threshold for the red line
threshold = 0.3

# Generating x values close to each other but not overlapping
x_values = np.random.normal(1, 0.01, len(predictions))
colors = ['red' if p < threshold else 'black' for p in predictions]
# Creating the plot
plt.figure(figsize=(10, 6))
plt.scatter(x_values, predictions, c=colors)
plt.axhline(y=threshold, color='red', linestyle='-')
# Annotating points below the threshold
for i, (x, y) in enumerate(zip(x_values, predictions)):
    if y < threshold:
        plt.text(x, y, str(i), color='red', fontsize=10)
#plt.ylim([0.0,1.4])
#plt.xlabel('Index')
plt.xticks([])
plt.yticks(fontsize=12)
plt.ylabel('Predictions (eV)',fontsize=14)
plt.title(f'Predict Li-Ni Exchange Energy in {str_name_base}\nBy R2:{r2},k_hop_attention:{k_hop_attention},gat_heads:{gat_heads},opt:{opt},seed:{seed}')
plt.savefig(f"{predict_path}/predict_pics_of_{str_name_base}_10_6.png",dpi=1200)
print(f"{predict_path}/predict_pics_of_{str_name_base}_10_6.png created!")
sb.Popen(["python",f"{code_dir}/show_fig.py",f"{predict_path}/predict_pics_of_{str_name_base}_10_6.png"])

# Outputting the indices of points below the threshold
#indices_below_threshold = [i for i, value in enumerate(predictions) if value < threshold]
#print(indices_below_threshold)


if opt=="True":
    print("!!! Using Opimizing Model, Predict Energy From Opimized Structure Calculated by QE relax !!!")
    print("!!! Using Opimizing Model, Predict Energy From Opimized Structure Calculated by QE relax !!!")
    print("!!! Using Opimizing Model, Predict Energy From Opimized Structure Calculated by QE relax !!!")
elif opt=="False":
    print("!!! Using From Scratch Model, Predict Energy From R/3m LiNiO2 Lattice Site Structure !!!")
    print("!!! Using From Scratch Model, Predict Energy From R/3m LiNiO2 Lattice Site Structure !!!")
    print("!!! Using From Scratch Model, Predict Energy From R/3m LiNiO2 Lattice Site Structure !!!")
