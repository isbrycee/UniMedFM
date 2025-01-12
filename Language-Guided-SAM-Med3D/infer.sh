python inference.py --seed 42 \
 -cp /home/jinghao/projects/MICCAI25/exp_trained_models/Language-Guided-SAM-Med3D_lora3_text_rank64_dice75.67_lr5e-4_param_img0.1_mask_prompt_trainRandom11Points.pth \
 -tdp /home/jinghao/projects/MICCAI25/dataset_gland_MRI_10_shot/gland_train -nc 5 \
 -dt Tr \
 --output_dir /home/jinghao/projects/MICCAI25/comparisons_seg_res/UniMedFM  \
 --task_name infer_turbo \
 --excel_path /home/jinghao/projects/MICCAI25/dataset_gland_MRI_10_shot/labels_gland_MRI_10_shot_for_MICCAI25.xlsx \
 --M3D_CLIP_model_path /home/jinghao/projects/MICCAI25/pretrained_models/M3D_CLIP \
 --use_text_features \
#  --save_image_and_gt
 #--sliding_window
