# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 17:26:16 2024

@author: xiety
"""
import numpy as np
import networkx as nx
from itertools import combinations
import copy

image_vector_list = ['-a', '+c', '+a', '-c', '-b', '+b']

def creat_image_graphs(G_original,image_vector_list):
    
    combinations_to_create = []

    # 生成所有可能的组合

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
    for combo in combinations_to_create:
        vector = combo_to_vector(combo)
        print(f"{combo}: {vector}")
    #print(combinations_to_create,len(combinations_to_create))  
    
    # 创建一个字典来存储所有生成的镜像图
    graphs = {}
    # 遍历所有组合以创建镜像图
    for idx, combo in enumerate(combinations_to_create):
        G_new = copy.deepcopy(G_original)
        translation_vector = np.zeros((3,))
    
        for vec in combo:
            sign = -1 if '-' in vec else 1
            axis = vec[-1]  # 'a', 'b', 或 'c'
    
            # 找到对应的晶格矢量
            if axis == 'a':
                vector = sign * np.array(G_new.nodes[0]['feature'][-14:-11]).reshape((3,))
            elif axis == 'b':
                vector = sign * np.array(G_new.nodes[0]['feature'][-11:-8]).reshape((3,))
            elif axis == 'c':
                vector = sign * np.array(G_new.nodes[0]['feature'][-8:-5]).reshape((3,))
    
            translation_vector += vector
    
        # 更新所有节点的坐标
        for node in G_new.nodes:
            original_coords = np.array(G_new.nodes[node]['feature'][-5:-2])
            G_new.nodes[node]['feature'][-5:-2] = original_coords + translation_vector
    
        # 保存平移后的图
        graphs["".join(combo)] = G_new

    # 此时 `graphs` 包含了所有生成的镜像图，每个镜像图都以其矢量组合的字符串标识为键（如 '-a+b-c'）
    
    
combinations_to_create = []

# 生成所有可能的组合

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
            
print(combinations_to_create,len(combinations_to_create))     
# 打印组合和对应的矢量表示
for combo in combinations_to_create:
    vector = combo_to_vector(combo)
    print(f"{combo}: {[int(i) for i in vector]}")
