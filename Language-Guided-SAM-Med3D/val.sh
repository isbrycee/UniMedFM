python validation.py --seed 2025 \
 -vp ./results/vis_sam_med3d_external_MRI_last \
 -cp  /home/jinghao/projects/MICCAI25/exp_trained_models/Language-Guided-SAM-Med3D_lora3_text_rank64_dice75.67_lr5e-4_param_img0.1_mask_prompt_trainRandom11Points.pth \
 -tdp /home/jinghao/projects/MICCAI25/External_data_Salivary_gland_tumours_Beijing/external_testset_gtmask_29_from_beijing_spacing1.5/ \
 -nc 5 \
 --save_name ./results/sam_med3d.py \
 --excel_path /home/jinghao/projects/MICCAI25/External_data_Salivary_gland_tumours_Beijing/MRI_exterenal_dataset_for_MICCAI25.xlsx \
 --M3D_CLIP_model_path /home/jinghao/projects/MICCAI25/pretrained_models/M3D_CLIP
# -cp /home/haojing/workspace/MICCAI25/SAM-Med3D-with-ViT3D/work_dir/union_train/sam_model_latest.pth \