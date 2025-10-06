import os
import json
from collections import defaultdict

# 定义文件路径
train_file_path = '/root/autodl-tmp/WHATEVER_CLIP/dataset/TBAD/train_files/train.txt'
class_names_path = '/root/autodl-tmp/WHATEVER_CLIP/dataset/TBAD/class_names.json'
output_dir = '/root/autodl-tmp/WHATEVER_CLIP/dataset/TBAD/train_files/'

# 读取类别名称
with open(class_names_path, 'r') as f:
    class_names = json.load(f)

# 读取训练数据
with open(train_file_path, 'r') as f:
    lines = f.readlines()

# 按类别分组视频
class_videos = defaultdict(list)
for line in lines:
    video_name, class_id = line.strip().split()
    class_videos[class_id].append((video_name, class_id))

# 打印类别信息
print(f"总共有 {len(class_videos)} 个类别")
for class_id, videos in class_videos.items():
    print(f"类别 {class_id} ({class_names[class_id]}) 有 {len(videos)} 个视频")

# 生成不同shot的文件
for shot in [2, 4, 8]:
    output_file = os.path.join(output_dir, f'train_{shot}shot.txt')
    with open(output_file, 'w') as f:
        for class_id, videos in class_videos.items():
            # 为每个类别选择指定数量的视频
            selected_videos = videos[:shot]  # 选择前shot个视频
            for video_name, cid in selected_videos:
                f.write(f"{video_name} {cid}\n")
    
    print(f"已生成 {output_file}，包含 {shot}-shot 数据")
    # 验证生成的文件
    with open(output_file, 'r') as f:
        generated_lines = f.readlines()
    print(f"  {output_file} 包含 {len(generated_lines)} 行数据")

print("所有shot文件生成完成！")