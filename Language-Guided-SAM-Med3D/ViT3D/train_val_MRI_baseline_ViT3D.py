import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from  ViT3D.vit3d import DeiT_Transformer3D
import pandas as pd
from scipy.ndimage import zoom
from transformers import AutoTokenizer, AutoModel

def init_M3D_CLIP_model(model_path): # 
    device = torch.device("cuda") # or cpu

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=512,
        padding_side="right",
        use_fast=False
    )
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    model = model.to(device=device)

    return tokenizer, model

def resize_image(image, target_shape):
    """
    Resize a 3D image to the target shape using interpolation.
    """
    # 计算缩放因子
    zoom_factors = [
        target_shape[i] / image.shape[i] for i in range(len(target_shape))
    ]
    # 使用 scipy.ndimage.zoom 进行插值调整大小
    resized_image = zoom(image, zoom_factors, order=3)  # order=3 表示三次插值
    return resized_image

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 计算 CrossEntropyLoss
        # import pdb; pdb.set_trace()
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        # 获取预测概率
        probs = torch.exp(-ce_loss)
        # 计算 Focal Loss
        focal_loss = self.alpha * (1 - probs) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def read_excel_for_MRI_data(excel_path):
    # 读取Excel文件，header=0表示第一行是表头
    df = pd.read_excel(excel_path, header=0)

    # 初始化一个空字典来存储结果
    result_dict = {}
    
    # 获取第3、4、5、6列以及倒数第二列的列号
    columns_of_interest = [2, 3, 4, 5, len(df.columns) - 2]  # Python中列索引从0开始

    # 遍历第一列的每一行
    for index, row in df.iterrows():
        # 第一列作为key
        key = row[df.columns[0]].lower()
        
        # 创建一个字典来存储指定列的表头和对应的值
        value_dict = {}
        
        # 遍历感兴趣的列
        for col_index in columns_of_interest:
            # 将列的表头和对应的值添加到字典中
            value_dict[df.columns[col_index]] = row[col_index]
        
        # 将字典添加到结果字典中
        result_dict[key] = value_dict
        
    return result_dict

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# 自定义数据集类
class MRIDataset(Dataset):
    def __init__(self, data_dir, label_dict, transform=None, target_size=(128, 128, 128)):
        """
        Args:
            data_dir (str): MRI 数据文件夹路径
            label_dict (dict): 文件名到类别标签的映射
            transform (callable, optional): 图像预处理操作
        """
        self.data_dir = data_dir
        self.label_dict = label_dict
        self.transform = transform
        self.target_size = target_size
        self.file_names = os.listdir(data_dir)
        

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = os.path.join(self.data_dir, file_name)
        file_numbering = file_name.split('.')[0].lower()
        if '_resampled' in file_numbering:
            file_numbering = file_numbering.split('_resampled')[0]
        label = self.label_dict[file_numbering]
        
        # 加载 MRI 数据
        img = nib.load(file_path).get_fdata()
        mask_file_path = file_path.replace('imagesTr', 'labelsTr')
        mask_data = nib.load(mask_file_path).get_fdata()
        padding = 5
        
        # 检查图像和标注文件的形状是否一致
        if img.shape != mask_data.shape:
            raise ValueError("The shape of the MRI image and the mask must be the same.")
        
        
        ########################## Crop ROI ##############################
        # 找到标注为 1 的最小内接长方体的边界
        coords = np.array(np.where(mask_data == 1))
        min_coords = coords.min(axis=1)
        max_coords = coords.max(axis=1)
        
        # 扩展边界
        min_coords = np.maximum(min_coords - padding, 0)
        max_coords = np.minimum(max_coords + padding, np.array(img.shape) - 1)
        
        # 提取最小内接长方体区域
        img = img[min_coords[0]:max_coords[0]+1, 
                                    min_coords[1]:max_coords[1]+1, 
                                    min_coords[2]:max_coords[2]+1]
        #################################################################
        
        # # 转换为 float32 并归一化
        img = img.astype(np.float32)
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
        
        
        target_shape_for_M3D = (256, 256, 32)
        img_for_M3D_CLIP = resize_image(img, target_shape_for_M3D).reshape((1, 32, 256, 256))

        # 调整尺寸到 target_size
        img = self.resize(img, self.target_size)
        
        # 添加通道维度 (C, D, H, W)，适配 PyTorch 的 3D 卷积输入
        img = np.expand_dims(img, axis=0)
        
        if self.transform:
            img = self.transform(img)

        return torch.tensor(img, dtype=torch.float32), torch.tensor(img_for_M3D_CLIP, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def resize(self, img, target_size):
        """
        使用 scipy.ndimage.zoom 对 3D 图像进行重采样以调整大小。
        
        Args:
            img (numpy.ndarray): 原始图像 (D, H, W)
            target_size (tuple): 目标尺寸 (D, H, W)
        
        Returns:
            numpy.ndarray: 调整后的图像 (D, H, W)
        """
        factors = [t / s for t, s in zip(target_size, img.shape)]
        resized_img = zoom(img, factors, order=3)  # 使用三次插值 (order=3) 进行重采样
        return resized_img

# 数据加载函数
def create_dataloaders(train_data_dir, test_data_dir, label_dict, batch_size=4, test_size=0.2):
    """
    Args:
        data_dir (str): MRI 数据文件夹路径
        label_dict (dict): 文件名到类别标签的映射
        batch_size (int): 批量大小
        test_size (float): 测试集比例
    Returns:
        train_loader, val_loader: 训练和验证数据加载器
    """
    # file_names = list(label_dict.keys())
    # labels = list(label_dict.values())

    # 拆分训练集和测试集
    # train_files, val_files, train_labels, val_labels = train_test_split(
    #     file_names, labels, test_size=test_size, stratify=labels, random_state=42
    # )

    # train_dict = dict(zip(train_files, label_dict))
    # val_dict = dict(zip(val_files, label_dict))

    train_dataset = MRIDataset(train_data_dir, label_dict)
    val_dataset = MRIDataset(test_data_dir, label_dict)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader

# 训练函数
def train_model(model, M3D_CLIP_Model, train_loader, val_loader, device, num_epochs=20, learning_rate=1e-4, save_path="best_model.pth"):
    """
    Args:
        model (torch.nn.Module): 3D 图像分类模型
        train_loader, val_loader: 训练和验证数据加载器
        device (torch.device): 运行设备
        num_epochs (int): 训练轮数
        learning_rate (float): 学习率
    """
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    criterion = FocalLoss(alpha=0.25, gamma=2)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.Adam([
    #     {'params': model.mlp.parameters()},
    #     {'params': model.head.parameters()},
    #     ]
    #     , lr=learning_rate)
    best_val_acc = 0.0  # 保存最高验证准确率

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        for imgs, img_for_M3D_CLIP, labels in train_loader:
            imgs, img_for_M3D_CLIP, labels = imgs.to(device), img_for_M3D_CLIP.to(device), labels.to(device)
            M3D_CLIP_visual_features = M3D_CLIP_Model.encode_image(img_for_M3D_CLIP)[:, 0] # (bs, 768)

            # 前向传播
            outputs = model(imgs, M3D_CLIP_visual_features)

            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
        
        train_acc = accuracy_score(train_targets, train_preds)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for imgs, img_for_M3D_CLIP, labels in val_loader:
                imgs, img_for_M3D_CLIP, labels = imgs.to(device), img_for_M3D_CLIP.to(device), labels.to(device)
                
                M3D_CLIP_visual_features = M3D_CLIP_Model.encode_image(img_for_M3D_CLIP)[:, 0] # (bs, 768)
                outputs = model(imgs, M3D_CLIP_visual_features)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        print("val_preds: ", val_preds)
        print("val_targets: ", val_targets)
        val_acc = accuracy_score(val_targets, val_preds)

        # 如果验证集准确率更高，则保存模型权重
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved with val_acc: {best_val_acc:.4f}")

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss/len(train_loader):.4f} Train Acc: {train_acc:.4f} "
              f"Val Loss: {val_loss/len(val_loader):.4f} Val Acc: {val_acc:.4f}")

# 评估函数
def evaluate_model(model, M3D_CLIP_Model, data_loader, device):
    """
    Args:
        model (torch.nn.Module): 3D 图像分类模型
        data_loader: 数据加载器
        device (torch.device): 运行设备
    """
    model.eval()
    preds = []
    targets = []
    
    with torch.no_grad():
        for imgs, img_for_M3D_CLIP, labels in data_loader:
            imgs, img_for_M3D_CLIP, labels = imgs.to(device), img_for_M3D_CLIP.to(device), labels.to(device)
            M3D_CLIP_visual_features = M3D_CLIP_Model.encode_image(img_for_M3D_CLIP)[:, 0] # (bs, 768)

            # 前向传播
            outputs = model(imgs, M3D_CLIP_visual_features)

            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            targets.extend(labels.cpu().numpy())

    acc = accuracy_score(targets, preds)
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(targets, preds))

def load_ViT3D_pretrained_model(pre_trained_model_path, n_classes):
    # Load the pre-trained model checkpoint
    checkpoint = torch.load(pre_trained_model_path, map_location='cpu')
    print("Loaded pre-trained checkpoint from: %s" % pre_trained_model_path)

    # Extract the state dictionary from checkpoint
    checkpoint_model = checkpoint['net']
    
    # Load the state dict into your model
    model = DeiT_Transformer3D(img_size=(128, 128, 128), n_classes=n_classes, with_dist_token=False)  # replace YourModel with your actual model class
    msg = model.load_state_dict(checkpoint_model, strict=False)

    # Handling possible mismatches
    if msg.missing_keys:
        print("Warning: Missing keys in state dict: ", msg.missing_keys)
    if msg.unexpected_keys:
        print("Warning: Unexpected keys in state dict: ", msg.unexpected_keys)

    return model

# 主函数
def main():
    # 数据集路径和标签字典
    excel_path = '/home/haojing/workspace/MICCAI25/extract_explainable_feats/MRI_dataset_for_MICCAi25.xlsx'
    MRI_excel_info_dict = read_excel_for_MRI_data(excel_path)
    label_dict = {}
    for k, v in MRI_excel_info_dict.items():
        label_dict[k] = int(v['tumour classification']) - 1

    train_data_dir = "/home/haojing/workspace/MICCAI25/MRI_data/gland_train/imagesTr"  # 替换为你的 MRI 数据文件夹路径
    val_data_dir = "/home/haojing/workspace/MICCAI25/MRI_data/gland_test/imagesTr"

    pre_trained_model_path = '/home/haojing/workspace/MICCAI25/ViT3D_baseline/ViT_B_pretrained_noaug_mae75_BRATS2023_IXI_OASIS3_seed_8456_999_077000.pth.tar'
    
    model = load_ViT3D_pretrained_model(pre_trained_model_path, n_classes=2)
    
    M3D_tokenlizer, M3D_CLIP_Model = init_M3D_CLIP_model('/home/haojing/workspace/MICCAI25/M3D/M3D-CLIP')
    # 检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 创建数据加载器
    train_loader, val_loader = create_dataloaders(train_data_dir, val_data_dir, label_dict, batch_size=4)

    # 训练模型
    train_model(model, M3D_CLIP_Model, train_loader, val_loader, device, num_epochs=40, learning_rate=1e-4)

    # 测试模型
    print("Evaluating model on validation set:")
    evaluate_model(model, M3D_CLIP_Model, val_loader, device)

if __name__ == "__main__":
    main()

