import os
import sys
import argparse
import sqlite3
import json
import networkx as nx
import numpy as np
from collections import Counter

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description="GNN Train Code")
# 添加参数
parser.add_argument('csv', type=str, help='The Top_xxx file stores the structure which the model cannot be well learned')
parser.add_argument('database', type=str, help='The mode training database')
parser.add_argument("out_type", choices=["xsf", "xsf_std", "vasp"], help="The format of the output structure.")
parser.add_argument("--original", "-o", action="store_true", help="Output the original structure.")
parser.add_argument("--perfect", "-p", action="store_true", help="Output the perfect (pre-optimization) structure.")

# 解析命令行参数
args = parser.parse_args()

csv_file=args.csv
dir_file=(os.popen("pwd").read())
current_dir=max(dir_file.split('\n'))
db_name=args.database.split("/")[-1].strip() if "/" in args.database else args.database.strip()
db_dir=current_dir+"/"+args.database.split("db_name")[0] if "/" in args.database else current_dir+"/"
db_file=args.database
output_type=args.out_type

output_dir_name=f"Exchange_of_{csv_file.split('/')[-1].split('.csv')[0] if '/' in csv_file else csv_file.strip().split('.csv')[0]}"
output_dir=current_dir+"/"+output_dir_name

if not output_dir:
    os.makedirs(output_dir,exist_ok=True)

print(f"Csv file:{csv_file}")
print(f"Database dir:{db_dir}")

def split_lines(infiles,split_syb,split_mode=None):
    f1=infiles
    read_data=[]
    for lines in f1:
        read_data_row=[]
        if "_Model_" in lines:
            continue
        if "#Model" in lines:
            continue
        if len(lines) <= 1:
            continue
        if "\t" in lines:
            a_bf=(lines.split("\n")[0]).split("\t")
        else:
            a_bf=(lines.split("\n")[0]).split(split_syb)
        for i in a_bf:
            if len(i)>0:
                if split_mode=="str" or not split_mode:
                    read_data_row.append(str(i))
                elif split_mode=="float":
                    read_data_row.append(float(i))
        read_data.append(read_data_row)
   # print(len(read_data),len(read_data[0]))
    return read_data


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

def rl_aw(element_i):
    element=["H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Uun","Uuu","Uub"]   #end in 112
    #return element[int(element_i-1)]
    return element.index(element_i)+1


def extract_lattice_and_coords(graph):
    # 晶格信息：'lattice_optimized'对应优化后的晶格矩阵，每个元素是一个包含3个值的列表（9维特征）
    # 节点信息：每个节点有一个'coordinates_optimized'特征，包含优化后的坐标（3维特征）
    elements_list = ["Li", "O", "Mg", "Al", "Ti", "V", "Mn", "Co", "Ni", "Zr"]
    
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
 
    #####not used down###
    ##### - bader价态：1维
    ##### - lowdin价态：1维
    ##### - 磁矩：1维
    #####not used up####

    # - 晶格-优化前：9维
    # - 坐标-优化前：3维
    # - 晶格-优化后：9维
    # - 坐标-优化后：3维
    # - 原子序号：1维
 
    # - 结构名称：1维
    # 总共：53维
    
    # 初始化列表来存储晶格和坐标信息
    perf_lat = []
    opt_lat = []
    perf_coords = []
    opt_coords = []
    perf_element = []
    opt_element = []
    index=[]
    for node, data in graph.nodes(data=True):
        features = data['feature']

        # 提取原子种类
        element_index = np.argmax(features[:10])  # 假设前10维是独热编码
        element = elements_list[element_index]  # 获取元素种类

        # 添加元素信息
        perf_element.append(element)
        opt_element.append(element)

        # 提取晶格信息
        if not perf_lat:  # 假设所有节点的晶格信息相同，只需提取一次
            perf_lat.extend(features[-26:-17])  # 提取优化前晶格信息
            opt_lat.extend(features[-14:-5])  # 提取优化后晶格信息
        
        # 提取坐标信息
        perf_coords.append(features[-17:-14])  # 提取优化前坐标信息
        opt_coords.append(features[-5:-2])  # 提取优化后坐标信息
        # 提取原子索引
        index.append(features[-2])

    # 打包，然后根据 index 排序
    packed_list = list(zip(index, perf_coords, opt_coords, perf_element, opt_element))
    packed_list.sort(key=lambda x: x[0])  # 假设index是可排序的
    
    # 解包
    index, perf_coords, opt_coords, perf_element, opt_element = zip(*packed_list)
    
    # 转换回所需的格式
    perf_coords = list(perf_coords)
    opt_coords = list(opt_coords)
    perf_element = list(perf_element)
    opt_element = list(opt_element)
    index = list(index)  # 如果你还需要使用排序后的索引列表

    # 将晶格信息转换为3x3的numpy数组
    perf_lat = np.array(perf_lat).reshape((3, 3))
    opt_lat = np.array(opt_lat).reshape((3, 3))

    # 组装返回的字典
    result = {
        'perf_lat': perf_lat,
        'opt_lat': opt_lat,
        'perf_coords': perf_coords,
        'opt_coords': opt_coords,
        'perf_element': perf_element,
        'opt_element': opt_element
    }

    return result



def elements_and_counts(elements):
    # Use Counter to get the counts of each element
    counts = Counter(elements)
    # Separate the elements and their counts into two lists
    elements_list = list(counts.keys())
    counts_list = list(counts.values())
    return elements_list, counts_list

def create_out_file(out_type,lat,coords,species,graph_name):
    output=[]

    if out_type=="xsf_std":
        output.append("DIM-GROUP\n")
        output.append("           3           1\n")
        output.append(" PRIMVEC\n")
        for i in lat:
            output.append(f"   {i[0]:.10f}    {i[1]:.10f}    {i[2]:.10f}\n") 
        output.append(" CONVVEC\n")
        for i in lat:
            output.append(f"   {i[0]:.10f}    {i[1]:.10f}    {i[2]:.10f}\n") 
        output.append(" PRIMCOORD\n")
        output.append(f"          {len(coords)}           1\n")
        for i,v in enumerate(coords):
            output.append(f" {rl_aw(species[i])}   {v[0]:.10f}    {v[1]:.10f}    {v[2]:.10f}\n") 

    elif out_type=="xsf":
        output.append("DIM-GROUP\n")
        output.append("           3           1\n")
        output.append(" PRIMVEC\n")
        for i in lat:
            output.append(f"   {i[0]:.10f}    {i[1]:.10f}    {i[2]:.10f}\n") 
        output.append(" PRIMCOORD\n")
        output.append(f"          {len(coords)}           1\n")
        for i,v in enumerate(coords):
            output.append(f" {species[i]}   {v[0]:.10f}    {v[1]:.10f}    {v[2]:.10f}\n") 

    elif out_type=="vasp":
        lattice_matrix=lat
        coords_frac=[]
        for coord in coords:
            car_coords_array = np.array(coord[:3])

            # 计算晶格矩阵的逆
            lattice_matrix_inv = np.linalg.inv(lattice_matrix)
            # 使用晶格矩阵的逆将笛卡尔坐标转换回分数坐标
            frac_coords_array = np.dot(car_coords_array,lattice_matrix_inv)
            coords_frac.append(frac_coords_array.tolist())
        elements, counts = elements_and_counts(species)
        output.append(f"VASP FILE {graph_name}\n")
        output.append("1.0\n")
        for i in lat:
            #print(i)
            output.append(f"   {i[0]:.10f}    {i[1]:.10f}    {i[2]:.10f}\n") 
        output.append(" "+"  ".join(elements)+"\n")
        output.append(" "+"  ".join([str(x) for x in counts])+"\n")
        output.append("Direct\n")
        for i,v in enumerate(coords_frac):
            output.append(f"   {v[0]:.10f}    {v[1]:.10f}    {v[2]:.10f}\n") 
        
    return output 

def create_output(out,name,subdir_i):
    subdir=output_dir+'/'+subdir_i
    if not os.path.exists(subdir):
        os.makedirs(subdir,exist_ok=True)
    f_out=open(f"{subdir}/{name}","w+")
    for i in out:
        f_out.writelines(i)
    f_out.close()
    print(f"{subdir}/{name} Created!")


def create_exchange_main(db_file,graph_name,li_index,ni_index):

    print("Executing on:",graph_name,"li-ni:",li_index," ",ni_index)
    graph = get_graph_by_name(db_file, graph_name)
    print(graph)
    #print([[u,v,data] for u,v,data in graph.edges(data=True)])
    str_info=extract_lattice_and_coords(graph)
    #print(str_info['orig_lat'])
    #for i,v in enumerate(orig_coords):
        #print(orig_ele[i],v)
    
    opt_lat=str_info['opt_lat']
    opt_coords_orig=str_info['opt_coords']
    opt_ele=str_info['opt_element']
    
    
    li_pos = li_index - 1
    ni_pos = ni_index - 1
    
    if output_type=="xsf":
        out_filename=f"out_relax_{graph_name}.xsf"
    elif output_type=="xsf_std":
        out_filename=f"out_relax_{graph_name}_standard.xsf"
    elif output_type=="vasp":
        out_filename=f"out_relax_{graph_name}.vasp"
    
    
    if args.perfect:
        perf_lat=str_info['perf_lat']
        perf_coords_orig=str_info['perf_coords']
        perf_ele=str_info['perf_element']
    
        perf_coords_new = perf_coords_orig.copy()
        perf_coords_new[li_pos], perf_coords_new[ni_pos] = perf_coords_orig[ni_pos], perf_coords_orig[li_pos]
        output_perf_new=create_out_file(output_type,perf_lat,perf_coords_new,perf_ele,graph_name)
        create_output(output_perf_new,out_filename.replace(f"{graph_name}",f"{graph_name}_Li{li_index}_Ni{ni_index}_perfect"),f'perfect_exchange_{output_type}')
        #print(out_filename.replace(f"{graph_name}",f"{graph_name}_Li{li_index}_Ni{ni_index}_perfect")+f" Created in {output_dir}!")
    
    
    opt_coords_new = opt_coords_orig.copy()
    opt_coords_new[li_pos], opt_coords_new[ni_pos] = opt_coords_orig[ni_pos], opt_coords_orig[li_pos]
    output_opt_new=create_out_file(output_type,opt_lat,opt_coords_new,opt_ele,graph_name)
    create_output(output_opt_new,out_filename.replace(f"{graph_name}",f"{graph_name}_Li{li_index}_Ni{ni_index}"),f'exchange_{output_type}') 
    #print(out_filename.replace(f"{graph_name}",f"{graph_name}_Li{li_index}_Ni{ni_index}")+f" Created in {output_dir}!")
    
    if args.original:
       output_opt_orig=create_out_file(output_type,opt_lat,opt_coords_orig,opt_ele,graph_name)
       create_output(output_opt_orig,out_filename,f'original_{output_type}') 
       #print(out_filename+f" Created in {output_dir}!")
       if args.perfect:
           output_perf_orig=create_out_file(output_type,perf_lat,perf_coords_orig,perf_ele,graph_name)
           create_output(output_perf_orig,out_filename.replace(f"{graph_name}",f"{graph_name}_perfect"),f'perfect_original_{output_type}') 
           #print(out_filename.replace(f"{graph_name}",f"{graph_name}_perfect")+f" Created in {output_dir}!")
    
datas=split_lines(open(csv_file).readlines(),'\t')
index=[int(i[0]) for i in datas]
str_name_list=[i[1] for i in datas]
li_indexs=[int(i[2]) for i in datas]
ni_indexs=[int(i[3]) for i in datas]
#print(index[:5],str_name_list[:5],li_indexs[:5],ni_indexs[:5])
#sys.exit(0)
for i,v in enumerate(str_name_list):
    create_exchange_main(db_file,v,li_indexs[i],ni_indexs[i])
