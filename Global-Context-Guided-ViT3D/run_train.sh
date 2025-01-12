
export CUDA_VISIBLE_DEVICES=0

python train_val_MRI_baseline_ViT3D.py \
                    --excel_path /home/jinghao/projects/MICCAI25/dataset_gland_MRI_10_shot/labels_gland_MRI_10_shot_for_MICCAI25.xlsx \
                    --train_data_dir /home/jinghao/projects/MICCAI25/dataset_gland_MRI_10_shot/gland_train/ \
                    --val_data_dir /home/jinghao/projects/MICCAI25/dataset_gland_MRI_10_shot/gland_test/ \
                    --pre_trained_model_path /home/jinghao/projects/MICCAI25/pretrained_models/ViT_B_pretrained_noaug_mae75_BRATS2023_IXI_OASIS3_seed_8456_999_077000.pth.tar \
                    --M3D_CLIP_model_path /home/jinghao/projects/MICCAI25/pretrained_models/M3D_CLIP \
                    --batch_size 8 \
                    --learning_rate 5e-4 \
                    --updated_param head \
                    --is_crop \
                    --padding_size 0 \
                    --save_path UniMedFM_gtcrop_pad0_lr5e-4_test.pth \
                    --use_M3D_features \
                    --num_twoway_transformer_layers 2
                    
                    # --is_only_evaluate
                    # --use_M3D_features

