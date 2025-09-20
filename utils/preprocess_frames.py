def preprocess_tensor(self, frames, target_size=1280):
    """
    将[num_frames, h, w, c] (uint8)的视频帧转换为 [num_frames, c, target_size, target_size] (float32)
    """
    num_frames, h, w, c = frames.shape
    output = torch.zeros(num_frames, c, target_size, target_size, dtype=torch.float32)

    # 计算缩放比例（保持宽高比）
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    for i in range(num_frames):
        # 1. 提取单帧并转换为numpy
        frame = frames[i].numpy()  # [h, w, c]
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 2. 缩放至目标尺寸
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 3. 创建填充背景并居中放置
        padded = np.full((target_size, target_size, c), 1, dtype=np.uint8)  # 黑色填充
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2

        padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w, :] = resized

        # 4. 转换回Tensor并调整维度
        tensor = torch.from_numpy(padded).permute(2, 0, 1)  # [c, target_size, target_size]
        output[i] = tensor

    return output