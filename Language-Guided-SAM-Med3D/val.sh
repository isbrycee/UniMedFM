python validation.py --seed 2025 \
 -vp ./results/vis_sam_med3d \
 -cp  /home/haojing/workplace/MICCAI25/SAM-Med3D-with-ViT3D/exp_trained_models/lora3_text_rank64_dice75.67_lr5e-4_param_img0.1_mask_prompt_trainRandom11Points.pth \
 -tdp /home/haojing/workplace/MICCAI25/MRI_data/gland_10_shot/gland_test \
 -nc 5 \
 --save_name ./results/sam_med3d.py
# -cp /home/haojing/workspace/MICCAI25/SAM-Med3D-with-ViT3D/work_dir/union_train/sam_model_latest.pth \