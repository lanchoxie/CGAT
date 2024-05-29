from collections import defaultdict
import sqlite3
import json
import networkx as nx
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import sleep
import argparse
from itertools import combinations
from functools import reduce
import copy
import sys

parser = argparse.ArgumentParser(description="Test")
parser.add_argument("--knn",'-k',type=int, default=5,help="k-th neighbor extraction")
parser.add_argument("--check",'-c',type=int, default=5,help="check neighbor extraction")
parser.add_argument("atom_number",type=int, nargs='?',default=1,help="check atom number, vesta index")
args=parser.parse_args()
atom=args.atom_number
k_nn=args.knn
print(k_nn)

db_file = 'gnn_data.save/wholegraphs_53d_features.db'

def get_graph_by_name(db_file, graph_name):
    # 连接到SQLite数据库
    conn = sqlite3.connect(db_file)
    c = conn.cursor()

    # 根据名称查询图数据
    c.execute("SELECT graph_data FROM wholegraphs WHERE name = ?", (graph_name,))
    row = c.fetchone()

    # 关闭数据库连接
    conn.close()

    # 如果找到了图，将JSON格式的图数据转换回图结构
    if row:
        graph_data = json.loads(row[0])
        graph = nx.node_link_graph(graph_data)
        return graph
    else:
        return None

def get_all_graphs(db_file):
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

# 示例：获取名称为'name1'的图
#graph_name = 'LiNiO2-331_NCMT_6111_1'
#graph = get_graph_by_name(db_file, graph_name)
#if graph:
#    print(f"Graph {graph_name} has been retrieved from the database.")
#else:
#    print(f"Graph {graph_name} was not found in the database.")
#print(graph.nodes[0]['feature'])
## 示例：获取包含所有图的字典
#all_graphs = get_all_graphs(db_file)
#print("All graphs have been retrieved from the database.")

def extract_subgraphs_with_periodic_structure(graph, k, label_extract=None):
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
    if label_extract:
        subgraphs = []
        for u, v, data in graph.edges(data=True):
            # 只考虑有能量信息的li_ni_edge边
            if data.get('edge_type') == 'li_ni_edge' and 'delta_E' in data:
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
                    subgraphs.append(subgraph)

    elif not label_extract:
        subgraphs = []
        for u, v, data in graph.edges(data=True):
            # 只考虑有能量信息的li_ni_edge边
            if data.get('edge_type') == 'li_ni_edge' and 'delta_E' not in data:
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
                    subgraphs.append(subgraph)
    #print(processed_edges)
    return subgraphs


def convert_sets_to_lists(obj):
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: convert_sets_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_sets_to_lists(i) for i in obj]
    else:
        return obj

def store_subgraphs_in_db(subgraphs,k,mode):
    # 连接到SQLite数据库
    conn = sqlite3.connect(f'gnn_data.save/subgraphs_k_neighbor_{k}_gnn_53d_feature_{mode}.db')
    c = conn.cursor()

    # 创建表格
    c.execute('''CREATE TABLE IF NOT EXISTS subgraphs
                 (id INTEGER PRIMARY KEY, graph_data TEXT)''')

    # 清空表中的现有记录
    c.execute("DELETE FROM subgraphs")
    # 存储每个子图
    for i, subgraph in enumerate(subgraphs):
        # 转换图中的所有 set 为 list
        subgraph_json_ready = nx.node_link_data(subgraph)
        subgraph_json_ready = convert_sets_to_lists(subgraph_json_ready)
        # 将图数据转换为JSON格式
        graph_data = json.dumps(subgraph_json_ready)
        c.execute("INSERT INTO subgraphs (graph_data) VALUES (?)", (graph_data,))

    # 提交事务并关闭连接
    conn.commit()
    conn.close()


def process_graph(G,k,data_model):
    label_ext=False if data_model=='predict' else True
    subgraphs=extract_subgraphs_with_periodic_structure(G, k,label_extract=label_ext)
    return subgraphs

def print_progress(done, total):
    percent_done = done / total * 100
    bar_length = int(percent_done / 100 * 60)
    bar = "[" + "#" * bar_length + "-" * (60 - bar_length) + "]" + f"{percent_done:.2f}%" + f"   {done}/{total}"
    print(bar, "\r", end='')

def convert_graphs(graphs,k,data_model):
    with ProcessPoolExecutor() as executor:
        # 创建future到索引的映射
        futures = {executor.submit(process_graph, graph,k,data_model): i for i, graph in enumerate(graphs)}
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



#all_graphs_dict = get_all_graphs(db_file)
#all_graphs=[graph for name,graph in all_graphs_dict.items()]


#print(len([[u,v,data] for u,v,data in all_graphs[0].edges(data=True) if 'delta_E' in data]))
#print(len([[u,v,data] for u,v,data in all_graphs[0].edges(data=True) if 'delta_E' not in data]))

#for debugging
#graph_test = get_graph_by_name(db_file, "LiNiO2-331_NCMT_6111_8")
#print(len([[u,v,data] for u,v,data in graph_test.edges(data=True) if 'delta_E' in data]))
#print(len([[u,v,data] for u,v,data in graph_test.edges(data=True) if 'delta_E' not in data]))
#a=([print([u+1,v+1,data]) for u,v,data in graph_test.edges(data=True) if 'delta_E' in data])
'''
print("Creating whole graphs...")
for k in range(2,6):
    print(f"\nextract subgraphs k neighbor:{k}")
    output_subgraphs_train = convert_graphs(all_graphs,k,'train') 
    print(f"\nLabeled_Number:{len(output_subgraphs_train)}")
    store_subgraphs_in_db(output_subgraphs_train,k,'train')
    output_subgraphs_predict = convert_graphs(all_graphs,k,'predict') 
    print(f"\nUnlabeled_Number:{len(output_subgraphs_predict)}")
    store_subgraphs_in_db(output_subgraphs_predict,k,'predict')
print()
'''

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
        boundary_info={x[1]:[x[2],x[0],x[3]] for x in directs}
         
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
                                atom_ind_dir_pair_in[(atom, neighbor)].add(f"-{axis_labels[i]}") #无视a，b顺序
        corrected_directions=list(set(directions))
        #print(atom,element,ind,corrected_directions,[f"{i:.8f}" for i in frac_coords],[f"{i:.8f}" for i in pos])

        boundary_directions.append([element, int(feature[-2])+1, corrected_directions, frac_coords])

    return boundary_directions,atom_ind_dir_pair_in


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
        if data['edge_type']=='li_ni_edge':
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

    if ka2>k:
        pass
    elif ka2<k and k<ka2*2:
        directions.append('+a')
    elif ka2*2<k and k<ka2*3:
        directions.append('+a')
        directions.append('-a')
    else:
        directions.append('+a')
        directions.append('-a')
        further_expand.append('a')

    if kb2>k:
        pass
    elif kb2<k and k<kb2*2:
        directions.append('+b')
    elif kb2<k*2 and k<kb2*3:
        directions.append('+b')
        directions.append('-b')
    else:
        directions.append('+b')
        directions.append('-b')
        further_expand.append('b')

    if kc2>k:
        pass
    elif kc2<k and k<kc2*2:
        directions.append('+c')
    elif kc2<k*2 and k<kc2*3:
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
    
    #write_xsf("boundary_atoms.xsf", graph_loc, lattice_vectors, selected_atoms=boundary_atoms)
    #write_xsf("boundary_atoms_marked.xsf", graph_loc, lattice_vectors, selected_atoms=boundary_atoms, directs=boundary_directions)
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



all_graphs_dict = get_all_graphs(db_file)
all_graphs=[graph for name,graph in all_graphs_dict.items()]
print("Expanding whole graphs...")
all_graphs2=convert_subgraphs_expand(all_graphs,k_nn)
print("Expanding whole graphs Done!")
#print(all_graphs2)
'''
graph_test1 = get_graph_by_name(db_file, "LiNiO2-331_NCMT_4311_6")
graph_test2 = get_graph_by_name(db_file, "LiNiO2-331_NCMT_4311_0")
all_graphs=[graph_test1,graph_test2]
all_graphs2=convert_subgraphs_expand(all_graphs,k_nn)
#print(all_graphs[0])
#for node,attr in all_graphs2[0].nodes(data=True):
#    print(node,len(attr['feature']))
#sys.exit(0)
#all_graphs2=[graph_loc2]

#for subgraph in all_graphs2:
#    for node,attr in subgraph.nodes(data=True):
##        nbs= [nb for nb in subgraph.neighbors(node) if subgraph.edges[nb,node]['edge_type']=='original']
#        if len(nbs)>6:
#             print(attr['feature'][-1],len(attr['feature']))
'''
'''
print("Creating whole subgraphs datas...")
for k in range(2,3):
    print(f"\nextract subgraphs k neighbor:{k}")
    output_subgraphs_train = convert_graphs(all_graphs2,k,'train') 
    for i,v in enumerate(output_subgraphs_train):
        if len(v.nodes)!=38:
            print(i,v.nodes[next(iter(v))]['feature'][-1],len(v.nodes))
    print(f"\nLabeled_Number:{len(output_subgraphs_train)}")
    store_subgraphs_in_db(output_subgraphs_train,k,'train')

sys.exit(0)
'''
print("Creating whole subgraphs datas...")
for k in range(2,6):
    print(f"\nextract subgraphs k neighbor:{k}")
    output_subgraphs_train = convert_graphs(all_graphs2,k,'train') 
    print(f"\nLabeled_Number:{len(output_subgraphs_train)}")
    store_subgraphs_in_db(output_subgraphs_train,k,'train')
    output_subgraphs_predict = convert_graphs(all_graphs2,k,'predict') 
    print(f"\nUnlabeled_Number:{len(output_subgraphs_predict)}")
    store_subgraphs_in_db(output_subgraphs_predict,k,'predict')
print()
