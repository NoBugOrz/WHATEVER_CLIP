import os
import json
import re
from utils.tools import prepare_frames


def write_file_labels(file_list, label_dict, output_path):
    """
    将文件列表中的文件名及其对应标签写入txt文件

    参数:
        file_list: 包含文件路径的列表，如['/root/.../bowing_to_students_45.mp4', ...]
        label_dict: 标签字典，如{'0': 'bowing to students', ...}
        output_path: 输出txt文件的路径
    """
    # 反转标签字典，便于通过动作名称查找数字标签
    reversed_label = {v: k for k, v in label_dict.items()}

    with open(output_path, 'w', encoding='utf-8') as f:
        for file_path in file_list:
            # 提取文件名（如"bowing_to_students_45.mp4"）
            file_name = os.path.basename(file_path)

            # 处理文件名以匹配标签（去掉扩展名和末尾数字）
            # 去掉.mp4扩展名
            name_without_ext = file_name.rsplit('.', 1)[0]
            # 去掉末尾的数字和下划线（如"_45"）
            action_part = re.sub(r'_\d+$', '', name_without_ext)
            # 将下划线转为空格，匹配标签字典中的格式
            action_name = action_part.replace('_', ' ')

            # 查找对应的标签
            if action_name in reversed_label:
                label = reversed_label[action_name]
                # 写入格式: 文件名 标签
                f.write(f"{file_name} {label}\n")
            else:
                # 处理未找到标签的情况
                print(f"警告: 未找到 '{action_name}' 对应的标签，已跳过文件 {file_name}")

def load_class_names(json_file):
    '''
    Returns: a list of class names
    '''
    with open(json_file, 'r', encoding='utf-8') as f:
        data_dict = json.load(f)
    return data_dict
    return [v for k,v in data_dict.items()]

def traverse_files(root_dir):
    # 遍历root_dir下的所有文件和子目录
    tmp = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            # 拼接文件的完整路径
            file_path = os.path.join(root, file)
            tmp.append(file_path)
            # 处理文件（这里仅打印路径，可替换为实际逻辑）
            print(file_path)
    return tmp

# root = '/root/autodl-tmp/TBAD_Dataset'
# json_file = '/root/autodl-tmp/WHATEVER_CLIP/dataset/TBAD/class_names.json'
#
# cls_names = load_class_names(json_file)
#
# file_names = traverse_files(root)
#
# write_file_labels(file_names, cls_names, '/root/autodl-tmp/WHATEVER_CLIP/dataset/TBAD/test_files/all_names')


