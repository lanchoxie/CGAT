from collections import defaultdict


def group_directions_complex(directions, symbol):
    groups = defaultdict(dict)
    special_group = [('O',)]  # 始终包含'O'

    # 检查是否应该包括'-' + symbol和'+' + symbol
    has_minus_symbol = any('-' + symbol in dir_tuple for dir_tuple in directions)
    has_plus_symbol = any('+' + symbol in dir_tuple for dir_tuple in directions)

    if has_minus_symbol:
        special_group.append(('-' + symbol,))
    if has_plus_symbol:
        special_group.append(('+' + symbol,))

    # 如果没有'-' + symbol 和 '+' + symbol，直接返回原始的方向列表
    if not has_minus_symbol and not has_plus_symbol:
        #print(symbol)
        return {f"{v}":[v] for i,v in enumerate(directions)}

    # 特殊组处理
    groups['(\'O\',)'] = {
        'items': special_group,
        'by_direction': {}
    }
    groups['(\'O\',)'] = {
        'items': special_group,
        'by_direction': {
            'No ' + symbol: [('O',)],
            '-' + symbol: [('-' + symbol,)],
            '+' + symbol: [('+' + symbol,)]            
        }
    }

    # 处理其他方向
    other_directions_groups = defaultdict(list)

    # 分组，忽略'symbol'
    for direction_tuple in directions:
        if direction_tuple in special_group:
            continue  # 跳过特殊组
        # 剩余方向去掉'symbol'后进行分类
        remaining_directions = tuple(sorted(d for d in direction_tuple if symbol not in d))
        other_directions_groups[remaining_directions].append(direction_tuple)

    # 处理和组织其他组
    for key, items in other_directions_groups.items():
        
        #group_key = f"group_{''.join(key)}"  # 创建组的键
        group_key = f"{key}"  # 创建组的键
        groups[group_key] = {
            'items': items,
            'by_direction': {}
        }
        # 为每个条目创建方向标签
        for item in items:
            for dir in item:
                if symbol in dir:  # 仅创建与symbol相关的方向标签
                    if dir not in groups[group_key]['by_direction']:
                        groups[group_key]['by_direction'][dir] = []
                    groups[group_key]['by_direction'][dir].append(item)
            # 为没有包含'symbol'的组合创建一个"No symbol"的条目
            if not any('-' + symbol in s or '+' + symbol in s for s in item):
                groups[group_key]['by_direction']['No ' + symbol] = [item]  # 没有包含'symbol'的版本

    return groups


# 方向组合列表

directions = [
    ('O',), ('-a',), ('+c',), ('+a',), ('-c',), ('-b',), ('+b',), 
    ('-a', '+c'), ('-a', '-c'), ('-a', '-b'), ('-a', '+b'), 
    ('+c', '+a'), ('+c', '-b'), ('+c', '+b'), ('+a', '-c'), ('+a', '-b'), ('+a', '+b'), 
    ('-c', '-b'), ('-c', '+b'), 
    ('-a', '+c', '-b'), ('-a', '+c', '+b'), ('-a', '-c', '-b'), ('-a', '-c', '+b'), 
    ('+c', '+a', '-b'), ('+c', '+a', '+b'), ('+a', '-c', '-b'), ('+a', '-c', '+b')
]

#directions = [('-a',), ('+c',), ('+b',), ('-a', '+c'), ('-a', '+b'), ('+c', '+b'), ('-a', '+c', '+b')]
# = [('-a',), ('+c',), ('-c',), ('-a', '+c'), ('-a', '-c')]

# 调用函数，以'a'为例
groups1 = group_directions_complex(directions, 'a')
print([eval(name) for name,group in groups1.items()])
groups2 = group_directions_complex([eval(name) for name,group in groups1.items()], 'b')
print([eval(name) for name,group in groups2.items()])
groups3 = group_directions_complex([eval(name) for name,group in groups2.items()], 'c')
print([eval(name) for name,group in groups3.items()])
#groups = group_directions_complex([name for name,group in groups.items()], 'b')

# 打印合并的组结果和访问特定方向的方式
for name, group in groups1.items():
    print(f"Group {name},{len(group['by_direction'])}:")
    for dir_label, items in group['by_direction'].items():
        print(f"  {dir_label},{len(items)}:",items[0],"".join(items[0]))
