from pymatgen.io.vasp import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure
import numpy as np
import sys
import os
import random

# Parameters
create_str = 1
filename = "LiNiO2-331.vasp"
doping_elements = ["Ni","Co","Mn","Ti"]   # Doping atom elements
#This is doping by proportion
atom_part=int(27/9)   #没份3个原子
doping_propor = [3,1,3,2]
doping_numbers = [i*atom_part for i in doping_propor]      # Doping atom numbers relative to doping_elements

print(doping_numbers)
output_name_seg=filename.split(".vasp")[0]+"_"+"".join([i[0] for i in doping_elements])+"_"+"".join([str(int(i/atom_part)) for i in doping_numbers])

sample_number = 16                             # Monte-Carlo sample time
doped_ele = "Ni"                             # Doped atom element in original cell
base_directory = sys.path[0]+'/'                # Base directory for reading and writing files
output_dir = "Doping_out/"
# Load the original structure
original_structure_path = os.path.join(base_directory, filename)
original_structure = Poscar.from_file(original_structure_path).structure

if not os.path.exists(base_directory+output_dir):
    os.system(f"mkdir {base_directory+output_dir}")
# Define the function to perform doping (as before)
def perform_doping(structure, doping_elements, doping_numbers, doped_element):
    """
    Perform doping on a structure by replacing specified elements with doping elements.

    Parameters:
    - structure: The original pymatgen Structure object.
    - doping_elements: List of elements to dope into the structure.
    - doping_numbers: List of numbers of atoms for each element to be doped.
    - doped_element: The element in the structure to be replaced.
    
    Returns:
    - A list of doped Structure objects.
    """
    doped_structures = []
    
    for _ in range(sample_number):
        # Create a copy of the original structure to modify
        modified_structure = structure.copy()
        # Get all indices of the doped element
        doped_indices = [i for i, site in enumerate(modified_structure) if site.species_string == doped_element]
        # Check if we have enough atoms to replace
        if sum(doping_numbers) > len(doped_indices):
            raise ValueError("Not enough atoms of the doped element to replace.")
        
        # Perform the doping
        for element, number in zip(doping_elements, doping_numbers):
            # Randomly select indices to replace
            selected_indices = random.sample(doped_indices, number)
            # Replace selected indices with the doping element
            for idx in selected_indices:
                modified_structure.replace(idx, element)
                # Ensure the same index is not selected again
                doped_indices.remove(idx)
        
        # Add the modified structure to the list
        doped_structures.append(modified_structure)
    
    return doped_structures
# Define the function to check for duplicates
def check_for_duplicates1(structures, spacegroup_to_files):
    unique_structures = []
    duplicate_indices = []
    
    for i, structure in enumerate(structures):
        analyzer = SpacegroupAnalyzer(structure)
        spacegroup = analyzer.get_space_group_symbol()
        
        if spacegroup not in spacegroup_to_files:
            spacegroup_to_files[spacegroup] = [i]
            unique_structures.append(structure)
        else:
            existing_structure_indices = spacegroup_to_files[spacegroup]
            is_duplicate = False
            for idx in existing_structure_indices:
                if structures[idx] == structure:  # This simplistic comparison would be replaced by a proper equivalence check
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                spacegroup_to_files[spacegroup].append(i)
                unique_structures.append(structure)
            else:
                duplicate_indices.append(i)
                
    for spacegroup, file_list in spacegroup_to_files.items():
        if len(file_list) > 1:
            print(f"相同空间群的文件（空间群 {spacegroup}）：{file_list}")    
            
    return unique_structures, duplicate_indices

def check_for_duplicates(structures):
    unique_structures = []
    duplicate_indices = []
    matcher = StructureMatcher()

    for i, structure in enumerate(structures):
        # Assume the structure is unique until found otherwise
        is_duplicate = False
        for unique_structure in unique_structures:
            if matcher.fit(unique_structure, structure):
                is_duplicate = True
                duplicate_indices.append(i)
                break

        if not is_duplicate:
            unique_structures.append(structure)
    for spacegroup, file_list in spacegroup_to_files.items():
        if len(file_list) > 1:
            print(f"相同空间群的文件（空间群 {spacegroup}）：{file_list}")  
    return unique_structures, duplicate_indices


# Perform doping to generate structures (as before)
doped_structures = perform_doping(original_structure, doping_elements, doping_numbers, doped_ele)

# After generating structures, check for duplicates and print the results
spacegroup_to_files = {}
#unique_structures, duplicate_indices = check_for_duplicates(doped_structures, spacegroup_to_files)
unique_structures, duplicate_indices = check_for_duplicates(doped_structures)

# Print the number of unique and duplicate structures
print(f"Number of unique structures: {len(unique_structures)}")
print(f"Number of duplicate structures: {len(duplicate_indices)}")

if len([f for f in os.listdir(base_directory+output_dir) if output_name_seg in f and '.xsf' in f])>0:

    exist_file=[int(f.split(output_name_seg+'_')[-1].split(".xsf")[0]) for f in os.listdir(base_directory+output_dir) if output_name_seg in f and '.xsf' in f]
    max_file = max(exist_file)
else:
    max_file =0
print(f"Existing file number:",max_file)
# Save the unique structures to files if create_str is set to 1
if create_str == 1:
    for i, unique_structure in enumerate(unique_structures):
        doped_file_path = os.path.join(base_directory+output_dir, f"{output_name_seg}_{max_file+i+1}.xsf")
        #os.mknod(doped_file_path)
        unique_structure.to(fmt="xsf", filename=doped_file_path)
        #os.system(f"mv {output_name_seg}_{max_file+i}.xsf {output_dir}")
        # 读取文件内容
        with open(doped_file_path, 'r') as file1:
            lines = file1.readlines()
        
        # 检查第八行的内容，如果以#开头则删除该行
        #if '# Cartesian coordinates' in lines[7]:
        lines_out=[line for i,line in enumerate(lines) if i!=6]
        lines_out[1]="            3             1\n"
        #print(lines_out) 
        # 保存修改后的文件
        with open(doped_file_path, 'w') as file1:
            file1.writelines(lines_out)
