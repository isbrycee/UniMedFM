import SimpleITK as sitk
import os
import torchio as tio
import nibabel as nib
import numpy as np

def resample_nii(input_path: str,
                 output_path: str,
                 target_spacing: tuple = (1.5, 1.5, 1.5),
                 n=None,
                 reference_image=None,
                 mode="nearest"):
    """
    Resample a nii.gz file to a specified spacing using torchio.

    Parameters:
    - input_path: Path to the input .nii.gz file.
    - output_path: Path to save the resampled .nii.gz file.
    - target_spacing: Desired spacing for resampling. Default is (1.5, 1.5, 1.5).
    """
    # Load the nii.gz file using torchio
    subject = tio.Subject(img=tio.ScalarImage(input_path))
    resampler = tio.Resample(target=target_spacing, image_interpolation=mode)
    resampled_subject = resampler(subject)

    if (n != None):
        image = resampled_subject.img
        tensor_data = image.data
        if (isinstance(n, int)):
            n = [n]
        for ni in n:
            tensor_data[tensor_data == ni] = -1
        tensor_data[tensor_data != -1] = 0
        tensor_data[tensor_data != 0] = 1
        save_image = tio.ScalarImage(tensor=tensor_data, affine=image.affine)
        reference_size = reference_image.shape[
            1:]  # omitting the channel dimension
        cropper_or_padder = tio.CropOrPad(reference_size)
        save_image = cropper_or_padder(save_image)
    else:
        save_image = resampled_subject.img

    save_image.save(output_path)
    print(f"图像已保存为 {output_path}")


def read_image_folder(folder_path, output_path):
    # 获取文件夹中的所有文件
    for f in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, f)):
            read_folder_MRI = os.path.join(folder_path, f)

    mri_file_number = folder_path.split('/')[-1]
    
    files = [os.path.join(folder_path, f) for f in os.listdir(read_folder_MRI)]
    
    # 使用 SimpleITK 读取图像
    reader = sitk.ImageSeriesReader()
    
    # 如果是 DICOM 文件夹，获取系列文件名
    if files:
        series_IDs = reader.GetGDCMSeriesIDs(read_folder_MRI)
        if series_IDs:
            series_file_names = reader.GetGDCMSeriesFileNames(read_folder_MRI, series_IDs[0])
            reader.SetFileNames(series_file_names)
        else:
            # 如果不是 DICOM，假设是常规图像格式
            reader.SetFileNames(files)
    
    # 读取图像
    image = reader.Execute()
    print(f"Image size: {image.GetSize()}")
    print(f"Image spacing: {image.GetSpacing()}")
    print(f"Image origin: {image.GetOrigin()}")
    print(f"Image Direction: {image.GetDirection()}")

    sitk.WriteImage(image, os.path.join(output_path, mri_file_number+'.nii.gz'))
    print(f"图像已保存为 {os.path.join(output_path, mri_file_number+'.nii.gz')}")

def resample_image(image, target_spacing, is_mask=False):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    target_size = [
        int(np.round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
    ]
    interpolator = sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(target_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)
    resampled_image = resampler.Execute(image)
    if is_mask:
        resampled_image = sitk.Cast(resampled_image, sitk.sitkUInt8)
    return resampled_image

def process_mri_and_masks(mri_folder, mask_folder, output_mri_folder, output_mask_folder, target_spacing=(1.0, 1.0, 1.0)):
    """
    Resample MRI images and masks, and save them to output folders.

    Args:
        mri_folder (str): Path to the folder containing MRI images.
        mask_folder (str): Path to the folder containing mask images.
        output_mri_folder (str): Path to save the resampled MRI images.
        output_mask_folder (str): Path to save the resampled mask images.
        target_spacing (tuple): The desired spacing for resampled images (default is (1.0, 1.0, 1.0)).
    """
    # Create output folders if they don't exist
    os.makedirs(output_mri_folder, exist_ok=True)
    os.makedirs(output_mask_folder, exist_ok=True)
    

    # List all files in the MRI and mask folders
    mri_files = [f for f in os.listdir(mri_folder) if f.endswith('.nii.gz')]
    mask_files = [f for f in os.listdir(mask_folder) if f.endswith('.nii.gz')]
    
    # Ensure MRI and mask files match
    mri_files.sort()
    mask_files.sort()
    
    for mri_file, mask_file in zip(mri_files, mask_files):
        # Load MRI and mask
        mri_path = os.path.join(mri_folder, mri_file)
        mask_path = os.path.join(mask_folder, mask_file)
        
        print(f"Processing: {mri_file} and {mask_file}")
        
        mri_image = sitk.ReadImage(mri_path)
        mask_image = sitk.ReadImage(mask_path)
        
        # Resample MRI and mask
        resampled_mri = resample_image(mri_image, target_spacing, is_mask=False)
        mask_image.CopyInformation(mri_image)
        resampled_mask = resample_image(mask_image, target_spacing, is_mask=True)

        # Save resampled images
        output_mri_path = os.path.join(output_mri_folder, mri_file)
        output_mask_path = os.path.join(output_mask_folder, mask_file)
        
        sitk.WriteImage(resampled_mri, output_mri_path)
        sitk.WriteImage(resampled_mask, output_mask_path)
        
        print(f"Saved resampled MRI to: {output_mri_path}")
        print(f"Saved resampled mask to: {output_mask_path}")


################################################################################################################################
################ Start Execute ################
# read and load the Image size of MRI.nii.gz 
for i in os.listdir('/home/jinghao/projects/MICCAI25/External_data_Salivary_gland_tumours_Beijing/external_testset_gtmask_29_from_beijing_spacing1.5/labelsTr'):
    img = nib.load(os.path.join('/home/jinghao/projects/MICCAI25/External_data_Salivary_gland_tumours_Beijing/external_testset_gtmask_29_from_beijing_spacing1.5/labelsTr',i))
    print(i)
    print('Image size:', img.shape)

# convert to .nii.gz
# input_path = '/home/jinghao/projects/MICCAI25/External_data_Salivary_gland_tumours_Beijing/T2FS'
# output_path = '/home/jinghao/projects/MICCAI25/External_data_Salivary_gland_tumours_Beijing/imagesTr'

# for item in os.listdir(input_path):
#     folder_path = os.path.join(input_path, item)
#     image = read_image_folder(folder_path, output_path)


def resampled_nii_gz(input_path, output_path, target_spacing=(1.0, 1.0, 1.0)):
    for f in os.listdir(input_path):
        resample_nii(
            os.path.join(input_path, f),
            os.path.join(output_path, f),
            target_spacing = target_spacing
        )

# resampled_nii_gz('/home/jinghao/projects/MICCAI25/External_data_Salivary_gland_tumours_Beijing/labelsTr',
#                    '/home/jinghao/projects/MICCAI25/External_data_Salivary_gland_tumours_Beijing/external_testset_29_from_beijing/labelsTr', 
#                    target_spacing=(1.0, 1.0, 1.0))

# Example usage
mri_folder = "/home/jinghao/projects/MICCAI25/External_data_Salivary_gland_tumours_Beijing/imagesTr"  # Replace with the path to the MRI folder
mask_folder = "/home/jinghao/projects/MICCAI25/External_data_Salivary_gland_tumours_Beijing/labelsTr"  # Replace with the path to the mask folder
output_mri_folder = "/home/jinghao/projects/MICCAI25/External_data_Salivary_gland_tumours_Beijing/external_testset_gtmask_29_from_beijing_spacing1.5_new/imagesTr"  # Replace with the output folder for MRI
output_mask_folder = "/home/jinghao/projects/MICCAI25/External_data_Salivary_gland_tumours_Beijing/external_testset_gtmask_29_from_beijing_spacing1.5_new/labelsTr"  # Replace with the output folder for masks

# Process MRI and mask files
# process_mri_and_masks(mri_folder, mask_folder, output_mri_folder, output_mask_folder, target_spacing=(1.5, 1.5, 1.5))

################ End Execute ################