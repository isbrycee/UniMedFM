## if wo options '--use_text_features', the orignal SAM-Med3D will be used. 

python validation.py --seed 2025 \
 -vp ./results/vis_sam_med3d \
 -cp  /home/jinghao/projects/MICCAI25/UniMedFM/Language-Guided-SAM-Med3D/work_dir/union_train/sam_model_dice_best.pth \
 -tdp /home/jinghao/projects/MICCAI25/dataset_gland_MRI_10_shot/gland_test \
 -nc 5 \
 --save_name ./results/sam_med3d.py \
 --excel_path /home/jinghao/projects/MICCAI25/dataset_gland_MRI_10_shot/labels_gland_MRI_10_shot_for_MICCAI25.xlsx \
 --M3D_CLIP_model_path /home/jinghao/projects/MICCAI25/pretrained_models/M3D_CLIP \
 --use_text_features \

# -cp /home/haojing/workspace/MICCAI25/SAM-Med3D-with-ViT3D/work_dir/union_train/sam_model_latest.pth \
