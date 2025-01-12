## if wo options '--use_text_features', the orignal SAM-Med3D will be used. 

export CUDA_VISIBLE_DEVICES=0
#!/bin/bash

python train.py --batch_size 1 \
            --accumulation_steps 1 \
            --img_size 128 \
            --num_epochs 61 \
            --val_interval 1 \
            --lr 5e-4 \
            --checkpoint /home/jinghao/projects/MICCAI25/pretrained_models/sam_med3d_brain.pth \
            --excel_path /home/jinghao/projects/MICCAI25/dataset_gland_MRI_10_shot/labels_gland_MRI_10_shot_for_MICCAI25.xlsx \
            --M3D_CLIP_model_path /home/jinghao/projects/MICCAI25/pretrained_models/M3D_CLIP \
            --use_text_features