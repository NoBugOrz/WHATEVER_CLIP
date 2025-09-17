import PIL.Image as Image
import numpy as np
from matplotlib import pyplot as plt


def save_image(img, path='images/', normalize=True):
    '''
    Args:
        img: tensor shape of (*, c, h, w)
        path: path to save image
        normalize: whether to normalize the image
    '''
    # 确保张量在CPU上
    if img.device.type != 'cpu':
        imgs = img.cpu()

    num_frames = imgs.shape[0]

    for i in range(num_frames):
        tensor = imgs[i]
        # 确保是CHW格式
        assert tensor.dim() == 3, "张量必须是CHW格式（3维）"

        # 转换为numpy数组并调整为HWC格式
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # CHW -> HWC

        # 归一化到0-255范围
        if normalize:
            img_np = img_np - img_np.min()
            img_np = img_np / img_np.max() * 255

        # 转换为uint8类型
        img_np = img_np.astype(np.uint8)

        # 处理单通道图像（灰度图）
        if img_np.shape[2] == 1:
            img_np = img_np.squeeze(2)

        # 转换为PIL图像并保存
        img = Image.fromarray(img_np)
        img.save(path+str(i)+'.png')

    return img

def show_image(img, normalize=True):
    '''
    Args:
        img: tensor shape of (*, c, h, w)
        path: path to save image
        normalize: whether to normalize the image
    '''
    # 确保张量在CPU上
    if img.device.type != 'cpu':
        imgs = img.cpu()

    num_frames = imgs.shape[0]

    for i in range(1):
        tensor = imgs[i]
        # 确保是CHW格式
        assert tensor.dim() == 3, "张量必须是CHW格式（3维）"

        # 转换为numpy数组并调整为HWC格式
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # CHW -> HWC

        # 归一化到0-255范围
        if normalize:
            img_np = img_np - img_np.min()
            img_np = img_np / img_np.max() * 255

        # 转换为uint8类型
        img_np = img_np.astype(np.uint8)

        # 处理单通道图像（灰度图）
        if img_np.shape[2] == 1:
            img_np = img_np.squeeze(2)

        plt.imshow(img_np)


    return img



