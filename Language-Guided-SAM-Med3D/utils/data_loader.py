from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchio as tio
from torchio.data.io import sitk_to_nib
import torch
import numpy as np
import os
import torch
import SimpleITK as sitk
from prefetch_generator import BackgroundGenerator
import pandas as pd

def read_excel_for_MRI_data(excel_path):
    # 读取Excel文件，header=0表示第一行是表头
    df = pd.read_excel(excel_path, header=0)
    # 初始化一个空字典来存储结果
    result_dict = {}
    # 获取第3、4、5、6列以及倒数第二列的列号
    columns_of_interest = [2, 3, 4, 5, len(df.columns) - 2]  # Python中列索引从0开始
    # columns_of_interest = [1, 2, 5, 6, -1]  # for external dataset test
    # 遍历第一列的每一行
    for index, row in df.iterrows():
        # 第一列作为key
        if type(row[df.columns[0]]) == int:
            key = str(row[df.columns[0]]).zfill(3)
        else:
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

class Dataset_Union_ALL(Dataset):
    def __init__(
        self,
        paths,
        mode="train",
        data_type="Tr",
        image_size=128,
        transform=None,
        threshold=500,
        split_num=1,
        split_idx=0,
        pcc=False,
        get_all_meta_info=False,
        text_and_classification_anno_path='',
    ):
        self.paths = paths
        self.data_type = data_type
        self.split_num = split_num
        self.split_idx = split_idx

        self._set_file_paths(self.paths)
        self.image_size = image_size
        self.transform = transform
        self.threshold = threshold
        self.mode = mode
        self.pcc = pcc
        self.get_all_meta_info = get_all_meta_info
        MRI_excel_info_dict = read_excel_for_MRI_data(text_and_classification_anno_path)
        self.label_dict = {}
        self.text_info = {}
        encoding_for_text_info_dict = {
            'f': 'female', 'm': 'male',
            # 'sublingual': 0, 'parotid': 1, 'submandibular': 2,
            'l': 'left', 'r': 'right',
            # '1': 1, '2': -1
        }
        for k, v in MRI_excel_info_dict.items():
            self.label_dict[k] = int(v['tumour classification']) - 1
            sex = encoding_for_text_info_dict[v['Sex'].lower()]
            age = v['Age']
            tumour_site = v['tumour site']
            tumour_side = encoding_for_text_info_dict[v['tumour side'].lower()]
            input_txt = f"The patient is a {age}-year-old {sex}. MRI imaging reveals a tumor in the {tumour_site} region on the {tumour_side} side of the patient's head. This is a salivary gland tumour of the human." 
            self.text_info[k] = input_txt

    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, index):

        sitk_image = sitk.ReadImage(self.image_paths[index])
        sitk_label = sitk.ReadImage(self.label_paths[index])
        ###################### add by bryce ################
        file_numbering = self.image_paths[index].split('/')[-1].split('.')[0].lower()
        if '_resampled' in file_numbering:
            file_numbering = file_numbering.split('_resampled')[0]
        label = self.label_dict[file_numbering]
        text = self.text_info[file_numbering]
        ###################### END ################
        # import pdb; pdb.set_trace()
        if sitk_image.GetOrigin() != sitk_label.GetOrigin():
            sitk_image.SetOrigin(sitk_label.GetOrigin())
        if sitk_image.GetDirection() != sitk_label.GetDirection():
            sitk_image.SetDirection(sitk_label.GetDirection())

        sitk_image_arr, _ = sitk_to_nib(sitk_image)
        sitk_label_arr, _ = sitk_to_nib(sitk_label)

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=sitk_image_arr),
            label=tio.LabelMap(tensor=sitk_label_arr),
        )

        if "/ct_" in self.image_paths[index]:
            subject = tio.Clamp(-1000, 1000)(subject)
        if self.transform:
            try:
                subject = self.transform(subject)
            except:
                print(self.image_paths[index])

        if self.pcc:
            print("using pcc setting")
            # crop from random click point
            random_index = torch.argwhere(subject.label.data == 1)
            if len(random_index) >= 1:
                random_index = random_index[np.random.randint(0, len(random_index))]
                # print(random_index)
                crop_mask = torch.zeros_like(subject.label.data)
                # print(crop_mask.shape)
                crop_mask[random_index[0]][random_index[1]][random_index[2]][
                    random_index[3]
                ] = 1
                subject.add_image(
                    tio.LabelMap(tensor=crop_mask, affine=subject.label.affine),
                    image_name="crop_mask",
                )
                subject = tio.CropOrPad(
                    mask_name="crop_mask",
                    target_shape=(self.image_size, self.image_size, self.image_size),
                )(subject)

        if subject.label.data.sum() <= self.threshold:
            return self.__getitem__(np.random.randint(self.__len__()))

        if self.mode == "train" and self.data_type == "Tr":
            return (
                subject.image.data.clone().detach(),
                subject.label.data.clone().detach(),
                torch.tensor(label, dtype=torch.long),
                text,
            )
        elif self.mode == "Val" and self.data_type == "Ts":
            return (
                subject.image.data.clone().detach(),
                subject.label.data.clone().detach(),
                torch.tensor(label, dtype=torch.long),
                text,
            )
        elif self.get_all_meta_info:
            meta_info = {
                "image_path": self.image_paths[index],
                "origin": sitk_label.GetOrigin(),
                "direction": sitk_label.GetDirection(),
                "spacing": sitk_label.GetSpacing(),
            }
            return (
                subject.image.data.clone().detach(),
                subject.label.data.clone().detach(),
                torch.tensor(label, dtype=torch.long),
                text,
                meta_info,
            )
        else:
            return (
                subject.image.data.clone().detach(),
                subject.label.data.clone().detach(),
                self.image_paths[index],
                torch.tensor(label, dtype=torch.long), # add by bryce
                text,
            )

    def _set_file_paths(self, paths):
        self.image_paths = []
        self.label_paths = []

        # if ${path}/labelsTr exists, search all .nii.gz
        for path in paths:
            d = os.path.join(path, f"labels{self.data_type}")
            if os.path.exists(d):
                for name in os.listdir(d):
                    base = os.path.basename(name).split(".nii.gz")[0]
                    label_path = os.path.join(
                        path, f"labels{self.data_type}", f"{base}.nii.gz"
                    )
                    self.image_paths.append(label_path.replace("labels", "images"))
                    self.label_paths.append(label_path)


class Dataset_Union_ALL_Val(Dataset_Union_ALL):
    def _set_file_paths(self, paths):
        self.image_paths = []
        self.label_paths = []

        # if ${path}/labelsTr exists, search all .nii.gz
        for path in paths:
            for dt in ["Tr", "Val", "Ts"]:
                d = os.path.join(path, f"labels{dt}")
                if os.path.exists(d):
                    for name in os.listdir(d):
                        base = os.path.basename(name).split(".nii.gz")[0]
                        label_path = os.path.join(path, f"labels{dt}", f"{base}.nii.gz")
                        self.image_paths.append(label_path.replace("labels", "images"))
                        self.label_paths.append(label_path)
        self.image_paths = self.image_paths[self.split_idx :: self.split_num]
        self.label_paths = self.label_paths[self.split_idx :: self.split_num]


class Dataset_Union_ALL_Infer(Dataset):
    """Only for inference, no label is returned from __getitem__."""

    def __init__(
        self,
        paths,
        data_type="infer",
        image_size=128,
        transform=None,
        split_num=1,
        split_idx=0,
        pcc=False,
        get_all_meta_info=False,
    ):
        self.paths = paths
        self.data_type = data_type
        self.split_num = split_num
        self.split_idx = split_idx

        self._set_file_paths(self.paths)
        self.image_size = image_size
        self.transform = transform
        self.pcc = pcc
        self.get_all_meta_info = get_all_meta_info

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        sitk_image = sitk.ReadImage(self.image_paths[index])

        sitk_image_arr, _ = sitk_to_nib(sitk_image)

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=sitk_image_arr),
        )

        if "/ct_" in self.image_paths[index]:
            subject = tio.Clamp(-1000, 1000)(subject)

        if self.transform:
            try:
                subject = self.transform(subject)
            except:
                print("Could not transform", self.image_paths[index])

        if self.pcc:
            print("using pcc setting")
            # crop from random click point
            random_index = torch.argwhere(subject.label.data == 1)
            if len(random_index) >= 1:
                random_index = random_index[np.random.randint(0, len(random_index))]
                crop_mask = torch.zeros_like(subject.label.data)
                crop_mask[random_index[0]][random_index[1]][random_index[2]][
                    random_index[3]
                ] = 1
                subject.add_image(
                    tio.LabelMap(tensor=crop_mask, affine=subject.label.affine),
                    image_name="crop_mask",
                )
                subject = tio.CropOrPad(
                    mask_name="crop_mask",
                    target_shape=(self.image_size, self.image_size, self.image_size),
                )(subject)

        elif self.get_all_meta_info:
            meta_info = {
                "image_path": self.image_paths[index],
                "direction": sitk_image.GetDirection(),
                "origin": sitk_image.GetOrigin(),
                "spacing": sitk_image.GetSpacing(),
            }
            return subject.image.data.clone().detach(), meta_info
        else:
            return subject.image.data.clone().detach(), self.image_paths[index]

    def _set_file_paths(self, paths):
        self.image_paths = []

        # if ${path}/infer exists, search all .nii.gz
        for path in paths:
            d = os.path.join(path, f"{self.data_type}")
            if os.path.exists(d):
                for name in os.listdir(d):
                    base = os.path.basename(name).split(".nii.gz")[0]
                    image_path = os.path.join(
                        path, f"{self.data_type}", f"{base}.nii.gz"
                    )
                    self.image_paths.append(image_path)
                    
        self.image_paths = self.image_paths[self.split_idx :: self.split_num]


class Union_Dataloader(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class Test_Single(Dataset):
    def __init__(self, paths, image_size=128, transform=None, threshold=500):
        self.paths = paths

        self._set_file_paths(self.paths)
        self.image_size = image_size
        self.transform = transform
        self.threshold = threshold

    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, index):

        sitk_image = sitk.ReadImage(self.image_paths[index])
        sitk_label = sitk.ReadImage(self.label_paths[index])

        if sitk_image.GetOrigin() != sitk_label.GetOrigin():
            sitk_image.SetOrigin(sitk_label.GetOrigin())
        if sitk_image.GetDirection() != sitk_label.GetDirection():
            sitk_image.SetDirection(sitk_label.GetDirection())

        subject = tio.Subject(
            image=tio.ScalarImage.from_sitk(sitk_image),
            label=tio.LabelMap.from_sitk(sitk_label),
        )

        if "/ct_" in self.image_paths[index]:
            subject = tio.Clamp(-1000, 1000)(subject)

        if self.transform:
            try:
                subject = self.transform(subject)
            except:
                print(self.image_paths[index])

        if subject.label.data.sum() <= self.threshold:
            return self.__getitem__(np.random.randint(self.__len__()))

        return (
            subject.image.data.clone().detach(),
            subject.label.data.clone().detach(),
            self.image_paths[index],
        )

    def _set_file_paths(self, paths):
        self.image_paths = []
        self.label_paths = []

        self.image_paths.append(paths)
        self.label_paths.append(paths.replace("images", "labels"))


if __name__ == "__main__":
    test_dataset = Dataset_Union_ALL_Infer(
        paths=['./data/inference/heart/hearts/',],
        data_type='infer',
        transform=tio.Compose([
            tio.ToCanonical(),
            tio.CropOrPad(target_shape=(128,128,128)),
        ]),
        pcc=False,
        get_all_meta_info=True,
        split_idx = 0,
        split_num = 1,
        )

    # test_dataset = Dataset_Union_ALL_Val(
        # paths=["./data/validation/experimental/heart/hearts"],
        # mode="Val",
        # transform=tio.Compose(
            # [
                # tio.ToCanonical(),
                # tio.CropOrPad(target_shape=(128, 128, 128)),
            # ]
        # ),
        # threshold=0,
        # pcc=False,
        # get_all_meta_info=True,
    # )

    test_dataloader = DataLoader(
        dataset=test_dataset, sampler=None, batch_size=1, shuffle=True
    )

    print(len(test_dataset))
    
    # for i, j, n in test_dataloader:
    for i, j in test_dataloader:
        print(i.shape)
        # print(j.shape)
        # print(n)
        print(j)
