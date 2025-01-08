# export CUDA_VISIBLE_DEVICES=1
#!/bin/bash
# for i in {0..9}; do
python train.py --batch_size 1 \
            --accumulation_steps 1 \
            --img_size 128 \
            --num_epochs 61 \
            --val_interval 1 \
            --lr 5e-4 \
            --checkpoint '/home/haojing/workplace/MICCAI25/SAM-Med3D-with-ViT3D/exp_trained_models/lora3_text_rank64_dice75.67_lr5e-4_param_img0.1_mask_prompt_trainRandom11Points.pth' # '/home/haojing/workplace/MICCAI25/sam_med3d_brain.pth'
# sleep 30s
# done