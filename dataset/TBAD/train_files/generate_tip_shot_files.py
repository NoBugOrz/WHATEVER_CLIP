import os
import json
from collections import defaultdict

# 定义文件路径
train_file_path = '/root/autodl-tmp/WHATEVER_CLIP/dataset/TBAD/train_files/train.txt'
class_names_path = '/root/autodl-tmp/WHATEVER_CLIP/dataset/TBAD/class_names.json'
output_dir = '/root/autodl-tmp/WHATEVER_CLIP/dataset/TBAD/train_files/'

# 之前生成的文件路径
existing_files = [
    os.path.join(output_dir, 'train_2shot.txt'),
    os.path.join(output_dir, 'train_4shot.txt'),
    os.path.join(output_dir, 'train_8shot.txt')
]

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

# 收集所有已使用的视频
used_videos = set()
for file_path in existing_files:
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            existing_lines = f.readlines()
            for line in existing_lines:
                video_name, _ = line.strip().split()
                used_videos.add(video_name)

print(f"总共排除了 {len(used_videos)} 个已使用的视频")

# 为每个类别筛选未使用的视频
unused_class_videos = defaultdict(list)
for class_id, videos in class_videos.items():
    for video_name, cid in videos:
        if video_name not in used_videos:
            unused_class_videos[class_id].append((video_name, cid))

# 打印可用视频统计
print("筛选后可用视频数量：")
for class_id, videos in unused_class_videos.items():
    print(f"类别 {class_id} ({class_names[class_id]}) 有 {len(videos)} 个未使用的视频")

# 生成不重复的shot文件
for shot in [2, 4, 8]:
    output_file = os.path.join(output_dir, f'tip_{shot}shot.txt')
    
    # 检查每个类别的可用视频是否足够
    for class_id, videos in unused_class_videos.items():
        if len(videos) < shot:
            print(f"警告: 类别 {class_id} 只有 {len(videos)} 个未使用的视频，无法满足 {shot}-shot 的需求")
    
    with open(output_file, 'w') as f:
        for class_id, videos in unused_class_videos.items():
            # 为每个类别选择指定数量的未使用视频
            selected_videos = videos[:shot]  # 选择前shot个未使用的视频
            for video_name, cid in selected_videos:
                f.write(f"{video_name} {cid}\n")
    
    print(f"已生成 {output_file}，包含不重复的 {shot}-shot 数据")
    # 验证生成的文件
    with open(output_file, 'r') as f:
        generated_lines = f.readlines()
    print(f"  {output_file} 包含 {len(generated_lines)} 行数据")

print("所有tip-shot文件生成完成！")