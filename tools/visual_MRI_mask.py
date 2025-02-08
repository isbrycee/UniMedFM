import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import imageio

def visualize_labels_on_mri(image_folder, label_folder, output_folder):
    """
    可视化 MRI 图像上的标签，并保存为 GIF 格式。
    
    Args:
        image_folder (str): MRI 图像文件夹路径，图像为 `.nii.gz` 格式。
        label_folder (str): 标签文件夹路径，标签为 `.nii.gz` 格式。
        output_folder (str): 保存结果 GIF 文件的文件夹路径。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取所有图像文件和标签文件
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.nii.gz')])
    label_files = sorted([f for f in os.listdir(label_folder) if f.endswith('.nii.gz')])

    if len(image_files) != len(label_files):
        raise ValueError("图像文件和标签文件的数量不一致！")

    for image_file, label_file in zip(image_files, label_files):
        # 加载图像和标签
        image_path = os.path.join(image_folder, image_file)
        print(image_path)
        label_path = os.path.join(label_folder, label_file)
        image = nib.load(image_path).get_fdata()
        print(image.shape)
        label = nib.load(label_path).get_fdata()
        
        if image.shape != label.shape:
            raise ValueError(f"图像 {image_file} 和标签 {label_file} 的形状不匹配！")

        # 创建 GIF 保存路径
        gif_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}.gif")

        # 生成 GIF
        frames = []
        for i in range(image.shape[2]):  # 遍历每个切片
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(image[:, :, i], cmap="gray", interpolation="none")
            ax.imshow(label[:, :, i], cmap=ListedColormap(['none', 'red']), alpha=0.5, interpolation="none")
            ax.axis("off")
            
            # 将图像保存为帧
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8) # !!! matplotlib version needs 3.9.4
            # print(frame.size)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(frame)
            plt.close(fig)

        # 保存为 GIF
        imageio.mimsave(gif_path, frames, fps=5)
        print(f"保存 GIF 文件: {gif_path}")

if __name__ == "__main__":
    # 输入文件夹路径
    image_folder = '/home/jinghao/projects/MICCAI25/comparisons_seg_res/SAM-Med3D/train/imagesTr/'
    label_folder = '/home/jinghao/projects/MICCAI25/comparisons_seg_res/SAM-Med3D/train/labelsTr/'
    output_folder = '/home/jinghao/projects/MICCAI25/visualizations/'

    visualize_labels_on_mri(image_folder, label_folder, output_folder)
