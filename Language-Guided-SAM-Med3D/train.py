# set up environment
import numpy as np
import random 
import datetime
import logging
import matplotlib.pyplot as plt
import os
join = os.path.join
from tqdm import tqdm
from torch.backends import cudnn
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchio as tio
from torch.utils.data.distributed import DistributedSampler
from segment_anything.build_sam3D import sam_model_registry3D
import argparse
from torch.cuda import amp
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from monai.losses import DiceCELoss
from contextlib import nullcontext
from utils.click_method import get_next_click3D_torch_2
from utils.data_loader import Dataset_Union_ALL, Union_Dataloader
from utils.data_paths import img_datas, img_datas_val
# add by bryce
from ViT3D.train_val_MRI_baseline_ViT3D import load_ViT3D_pretrained_model
from M3D.load_M3D_model import init_M3D_CLIP_model
from scipy.ndimage import zoom
import torch.nn as nn
from sklearn.metrics import accuracy_score
from glob import glob
from utils.data_loader import Dataset_Union_ALL_Val
from torch.utils.data import DataLoader


# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default='union_train')
parser.add_argument('--click_type', type=str, default='random')
parser.add_argument('--multi_click', action='store_true', default=False)
parser.add_argument('--model_type', type=str, default='vit_b_ori')
parser.add_argument('--checkpoint', type=str, default='ckpt/sam_med3d.pth')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--work_dir', type=str, default='work_dir')

# train
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[1])
parser.add_argument('--multi_gpu', action='store_true', default=False)
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--allow_partial_weight', action='store_true', default=False)
parser.add_argument('--is_train_ViT3D', type=bool, default=False)

# lr_scheduler
parser.add_argument('--lr_scheduler', type=str, default='multisteplr')
parser.add_argument('--step_size', type=list, default=[120, 180])
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--img_size', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--accumulation_steps', type=int, default=2)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=0.1)
parser.add_argument('--port', type=int, default=12361)

# val
parser.add_argument('-dt', '--data_type', type=str, default='Ts')
parser.add_argument('--threshold', type=int, default=0)
parser.add_argument('--val_interval', type=int, default=5)
parser.add_argument('--dim', type=int, default=3)
parser.add_argument('--split_idx', type=int, default=0)
parser.add_argument('--split_num', type=int, default=1)
parser.add_argument('--ft2d', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=2023)
args = parser.parse_args()

device = args.device
# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in args.gpu_ids])
logger = logging.getLogger(__name__)
LOG_OUT_DIR = join(args.work_dir, args.task_name)
click_methods = {
    'random': get_next_click3D_torch_2,
}
MODEL_SAVE_PATH = join(args.work_dir, args.task_name)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

def build_model(args):
    sam_model = sam_model_registry3D[args.model_type](checkpoint=None).to(device)
    if args.multi_gpu:
        sam_model = DDP(sam_model, device_ids=[args.rank], output_device=args.rank)
    return sam_model


def get_dataloaders(args):
    train_dataset = Dataset_Union_ALL(paths=img_datas, transform=tio.Compose([
        tio.ToCanonical(),
        tio.CropOrPad(mask_name='label', target_shape=(args.img_size,args.img_size,args.img_size)), # crop only object region
        tio.RandomFlip(axes=(0, 1, 2)),
    ]),
    threshold=1000)

    if args.multi_gpu:
        train_sampler = DistributedSampler(train_dataset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_dataloader = Union_Dataloader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size, 
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_dataloader

def get_dataloaders_val(args):

    all_dataset_paths = glob(join(img_datas_val[0]))
    all_dataset_paths = list(filter(os.path.isdir, all_dataset_paths))
    print("get", len(all_dataset_paths), "datasets")

    infer_transform = [
        tio.ToCanonical(),
        tio.CropOrPad(mask_name='label', target_shape=(args.img_size,args.img_size,args.img_size)),
    ]

    test_dataset = Dataset_Union_ALL_Val(
        paths=all_dataset_paths, 
        mode="Val", 
        data_type=args.data_type, 
        transform=tio.Compose(infer_transform),
        threshold=0,
        split_num=args.split_num,
        split_idx=args.split_idx,
        pcc=False,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        sampler=None,
        batch_size=args.batch_size, 
        shuffle=True
    )

    return test_dataloader


class BaseTrainer:
    def __init__(self, model, model_ViT3D, M3D_tokenlizer, M3D_CLIP_Model, dataloaders, dataloaders_val, args):

        self.model = model
        self.model_ViT3D = model_ViT3D
        self.M3D_tokenlizer = M3D_tokenlizer
        self.M3D_CLIP_Model = M3D_CLIP_Model
        self.dataloaders = dataloaders
        self.dataloaders_val = dataloaders_val
        self.args = args
        self.best_loss = np.inf
        self.best_dice = 0.0
        self.step_best_loss = np.inf
        self.step_best_dice = 0.0
        self.losses = []
        self.dices = []
        self.ious = []
        self.set_loss_fn()
        self.set_optimizer()
        self.set_lr_scheduler()
        if(args.resume):
            self.init_checkpoint(join(self.args.work_dir, self.args.task_name, 'sam_model_latest.pth'))
        else:
            self.init_checkpoint(self.args.checkpoint)

        self.norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
        
    def set_loss_fn(self):
        self.seg_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    
    def set_optimizer(self):
        if self.args.multi_gpu:
            sam_model = self.model.module
        else:
            sam_model = self.model

        self.optimizer = torch.optim.AdamW([
            {'params': sam_model.image_encoder.parameters(), 'lr': self.args.lr * 0.1}, # , 'lr': self.args.lr * 0.1},
            {'params': sam_model.prompt_encoder.parameters() , 'lr': self.args.lr * 0.1},
            {'params': sam_model.mask_decoder.parameters(), 'lr': self.args.lr * 0.1},
            # {'params': self.model_ViT3D.parameters(), 'lr': self.args.lr * 1},
            # {'params': self.model_ViT3D.head.parameters(), 'lr': self.args.lr * 1},
            # {'params': self.model_ViT3D.mlp.parameters(), 'lr': self.args.lr * 1},
        ], lr=self.args.lr, betas=(0.9,0.999), weight_decay=self.args.weight_decay)

    def set_lr_scheduler(self):
        if self.args.lr_scheduler == "multisteplr":
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                self.args.step_size,
                                                                self.args.gamma)
        elif self.args.lr_scheduler == "steplr":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                self.args.step_size[0],
                                                                self.args.gamma)
        elif self.args.lr_scheduler == 'coswarm':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer)
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, 0.1)

    def init_checkpoint(self, ckp_path):
        last_ckpt = None
        if os.path.exists(ckp_path):
            if self.args.multi_gpu:
                dist.barrier()
                last_ckpt = torch.load(ckp_path, map_location=self.args.device)
            else:
                last_ckpt = torch.load(ckp_path, map_location=self.args.device)
        
        if last_ckpt:
            if(self.args.allow_partial_weight):
                if self.args.multi_gpu:
                    self.model.module.load_state_dict(last_ckpt['model_state_dict'], strict=False)
                else:
                    self.model.load_state_dict(last_ckpt['model_state_dict'], strict=False)
            else:
                if self.args.multi_gpu:
                    self.model.module.load_state_dict(last_ckpt['model_state_dict'])
                else:
                    self.model.load_state_dict(last_ckpt['model_state_dict'], strict=False)
            if not self.args.resume:
                self.start_epoch = 0 
            else:
                self.start_epoch = last_ckpt['epoch']
                self.optimizer.load_state_dict(last_ckpt['optimizer_state_dict'])
                self.lr_scheduler.load_state_dict(last_ckpt['lr_scheduler_state_dict'])
                self.losses = last_ckpt['losses']
                self.dices = last_ckpt['dices']
                self.best_loss = last_ckpt['best_loss']
                self.best_dice = last_ckpt['best_dice']
            print(f"Loaded checkpoint from {ckp_path} (epoch {self.start_epoch})")
        else:
            self.start_epoch = 0
            print(f"No checkpoint found at {ckp_path}, start training from scratch")

    def save_checkpoint(self, epoch, state_dict, describe="last"):
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "losses": self.losses,
            "dices": self.dices,
            "best_loss": self.best_loss,
            "best_dice": self.best_dice,
            "args": self.args,
            "used_datas": img_datas,
        }, join(MODEL_SAVE_PATH, f"sam_model_{describe}.pth"))
    
    def batch_forward(self, sam_model, image_embedding, gt3D, low_res_masks, text_embed, points=None):
        
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=points,
            boxes=None,
            masks=low_res_masks,
        )
        low_res_masks, iou_predictions = sam_model.mask_decoder(
            image_embeddings=image_embedding.to(device), # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            text_embed=text_embed,
            multimask_output=False,
        )
        prev_masks = F.interpolate(low_res_masks, size=gt3D.shape[-3:], mode='trilinear', align_corners=False)
        return low_res_masks, prev_masks

    def get_points(self, prev_masks, gt3D):
        batch_points, batch_labels = click_methods[self.args.click_type](prev_masks, gt3D)

        points_co = torch.cat(batch_points, dim=0).to(device)
        points_la = torch.cat(batch_labels, dim=0).to(device)

        self.click_points.append(points_co)
        self.click_labels.append(points_la)

        points_multi = torch.cat(self.click_points, dim=1).to(device)
        labels_multi = torch.cat(self.click_labels, dim=1).to(device)

        if self.args.multi_click:
            points_input = points_multi
            labels_input = labels_multi
        else:
            points_input = points_co
            labels_input = points_la
        return points_input, labels_input

    def interaction(self, sam_model, image_embedding, gt3D, text_embed, num_clicks):
        return_loss = 0
        prev_masks = torch.zeros_like(gt3D).to(gt3D.device)
        low_res_masks = F.interpolate(prev_masks.float(), size=(args.img_size//4,args.img_size//4,args.img_size//4))
        random_insert = np.random.randint(2, 9)
        for num_click in range(num_clicks):
            points_input, labels_input = self.get_points(prev_masks, gt3D)

            if num_click == random_insert or num_click == num_clicks - 1:
                low_res_masks, prev_masks = self.batch_forward(sam_model, image_embedding, gt3D, low_res_masks, text_embed, points=None)
            else:
                low_res_masks, prev_masks = self.batch_forward(sam_model, image_embedding, gt3D, low_res_masks, text_embed, points=[points_input, labels_input])
            loss = self.seg_loss(prev_masks, gt3D)
            return_loss += loss
        return prev_masks, return_loss
    
    def get_dice_score(self, prev_masks, gt3D):
        def compute_dice(mask_pred, mask_gt):
            mask_threshold = 0.5

            mask_pred = (mask_pred > mask_threshold)
            mask_gt = (mask_gt > 0)
            
            volume_sum = mask_gt.sum() + mask_pred.sum()
            if volume_sum == 0:
                return np.NaN
            volume_intersect = (mask_gt & mask_pred).sum()
            return 2*volume_intersect / volume_sum
    
        pred_masks = (prev_masks > 0.5)
        true_masks = (gt3D > 0)
        dice_list = []
        for i in range(true_masks.shape[0]):
            dice_list.append(compute_dice(pred_masks[i], true_masks[i]))
        return (sum(dice_list)/len(dice_list)).item() 
    
    # add by bryce
    def get_RoI_img_from_pred_mask(self, imgs, gt3D, pred_masks, target_size=(128,128,128), padding=5):
        
        prev_masks = F.interpolate(pred_masks, size=gt3D.shape[-3:], mode='trilinear', align_corners=False)
        medsam_seg_prob = torch.sigmoid(prev_masks)  # (B, 1, 64, 64, 64)
        # convert prob to mask
        medsam_seg_prob = medsam_seg_prob.cpu().detach().numpy().squeeze()

        if len(medsam_seg_prob.shape) == 3:
            medsam_seg_prob = medsam_seg_prob[None, :, :, :]

        medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)

        ########################## Crop ROI ##############################
        # 找到标注为 1 的最小内接长方体的边界
        RoI_img_list = []
        # import pdb; pdb.set_trace()
        for bs_id in range(medsam_seg.shape[0]):
            coords = np.array(np.where(medsam_seg[bs_id] == 1))
            min_coords = coords.min(axis=1)
            max_coords = coords.max(axis=1)
            
            # 扩展边界
            min_coords = np.maximum(min_coords - padding, 0)
            max_coords = np.minimum(max_coords + padding, np.array(medsam_seg[bs_id].shape) - 1)
            
            # 提取最小内接长方体区域
            img = imgs[bs_id, :, min_coords[0]:max_coords[0]+1, 
                                        min_coords[1]:max_coords[1]+1, 
                                        min_coords[2]:max_coords[2]+1]
            
            img = np.expand_dims(self.resize_for_ViT3D(img.cpu().numpy().squeeze(0), target_size), axis=0)
            RoI_img_list.append(img)
            
        #################################################################
        # 调整尺寸到 target_size
        img = np.concatenate(RoI_img_list, axis=0)
        # import pdb; pdb.set_trace()

        # 添加通道维度 (C, D, H, W)，适配 PyTorch 的 3D 卷积输入
        img = np.expand_dims(img, axis=1)

        return torch.tensor(img, dtype=torch.float32)
    
    def resize_for_ViT3D(self, img, target_size=(128, 128, 128)):

        factors = [t / s for t, s in zip(target_size, img.shape)]
        
        resized_img = zoom(img, factors, order=3)  # 使用三次插值 (order=3) 进行重采样
    
        return resized_img

    def train_epoch(self, epoch, num_clicks):
        epoch_loss = 0
        epoch_iou = 0
        self.model.train()
        if self.args.multi_gpu:
            sam_model = self.model.module
        else:
            sam_model = self.model
            self.args.rank = -1
        
        if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
            tbar = tqdm(self.dataloaders)
        else:
            tbar = self.dataloaders

        self.optimizer.zero_grad()

        step_loss = 0
        epoch_dice = 0
        
        # add by bryce
        criterion = nn.CrossEntropyLoss()
        train_preds = []
        train_targets = []

        for step, (image3D, gt3D, binary_label, text) in enumerate(tbar):

            my_context = self.model.no_sync if self.args.rank != -1 and step % self.args.accumulation_steps != 0 else nullcontext

            with my_context():

                image3D = self.norm_transform(image3D.squeeze(dim=1)) # (N, C, W, H, D)
                image3D = image3D.unsqueeze(dim=1)
                
                image3D = image3D.to(device) # torch.Size([4, 1, 128, 128, 128])
                gt3D = gt3D.to(device).type(torch.long) # torch.Size([4, 1, 128, 128, 128])
                with amp.autocast():
                    image_embedding = sam_model.image_encoder(image3D)
                    # add by bryce; for text embedding
                    text_embeddings = self.M3D_tokenlizer(text, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
                    text_input_id = text_embeddings["input_ids"].to(device=device)
                    attention_mask_benign = text_embeddings["attention_mask"].to(device=device)
                    text_benign_features = self.M3D_CLIP_Model.encode_text(text_input_id, attention_mask_benign)[:, 0]

                    self.click_points = []
                    self.click_labels = []

                    pred_list = []

                    prev_masks, loss = self.interaction(sam_model, image_embedding, gt3D, text_benign_features, num_clicks=num_clicks)  # torch.Size([4, 1, 128, 128, 128])

                    # add by bryce; for feed into ViT3D
                    if epoch > 10 and self.args.is_train_ViT3D:
                        RoI_img = self.get_RoI_img_from_pred_mask(image3D, gt3D, prev_masks, target_size=(128,128,128), padding=5)
                        outputs = self.model_ViT3D(RoI_img)

                        # print(outputs)
                        loss_for_classification = criterion(outputs, binary_label)

                        train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                        train_targets.extend(binary_label.cpu().numpy())

                epoch_loss += loss.item()
                epoch_dice += self.get_dice_score(prev_masks,gt3D) 
                # add by bryce
                if epoch > 10 and self.args.is_train_ViT3D:
                    cur_loss = loss.item() + loss_for_classification.item() # changed by bryce
                    epoch_loss += 10 * loss_for_classification.item()
                else:
                    cur_loss = loss.item() 
                
                if epoch > 10 and self.args.is_train_ViT3D:
                    loss = (loss + 10 * loss_for_classification) / self.args.accumulation_steps
                else:
                    loss /= self.args.accumulation_steps # changed by bryce
                self.scaler.scale(loss).backward()    
            
            if step % self.args.accumulation_steps == 0 and step != 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                print_loss = step_loss / self.args.accumulation_steps
                step_loss = 0
                print_dice = self.get_dice_score(prev_masks, gt3D)
            else:
                step_loss += cur_loss

            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                if step % self.args.accumulation_steps == 0 and step != 0:
                    print(f'Epoch: {epoch}, Step: {step}, Loss: {print_loss}, Dice: {print_dice}')
                    if print_dice > self.step_best_dice:
                        self.step_best_dice = print_dice
                        if print_dice > 0.95:
                            self.save_checkpoint(
                                epoch,
                                sam_model.state_dict(),
                                describe=f'{epoch}_step_dice:{print_dice}_best'
                            )
                    if print_loss < self.step_best_loss:
                        self.step_best_loss = print_loss
            
        epoch_loss /= step+1
        epoch_dice /= step+1
        # add by bryce
        epoch_acc = accuracy_score(train_targets, train_preds)
        # print(self.model_ViT3D.head.parameters())
        return epoch_loss, epoch_iou, epoch_dice, pred_list, epoch_acc

    def eval_epoch(self, epoch, num_clicks):
        epoch_loss = 0
        epoch_iou = 0
        self.model.eval()
        self.model_ViT3D.eval()
        if self.args.multi_gpu:
            sam_model = self.model.module
        else:
            sam_model = self.model
            self.args.rank = -1
        
        if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
            tbar = tqdm(self.dataloaders_val)
        else:
            tbar = self.dataloaders_val

        step_loss = 0
        epoch_dice = 0
        
        # add by bryce
        criterion = nn.CrossEntropyLoss()
        train_preds = []
        train_targets = []

        for step, (image3D, gt3D, binary_label, text) in enumerate(tbar):

            my_context = self.model.no_sync if self.args.rank != -1 and step % self.args.accumulation_steps != 0 else nullcontext

            with my_context():

                image3D = self.norm_transform(image3D.squeeze(dim=1)) # (N, C, W, H, D)
                image3D = image3D.unsqueeze(dim=1)
                # add by bryce; for text embedding
                text_embeddings = self.M3D_tokenlizer(text, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
                text_input_id = text_embeddings["input_ids"].to(device=device)
                attention_mask_benign = text_embeddings["attention_mask"].to(device=device)
                text_benign_features = self.M3D_CLIP_Model.encode_text(text_input_id, attention_mask_benign)[:, 0]
                image3D = image3D.to(device) # torch.Size([4, 1, 128, 128, 128])
                gt3D = gt3D.to(device).type(torch.long) # torch.Size([4, 1, 128, 128, 128])
                with torch.no_grad():
                    image_embedding = sam_model.image_encoder(image3D)

                    self.click_points = []
                    self.click_labels = []

                    pred_list = []

                    prev_masks, loss = self.interaction(sam_model, image_embedding, gt3D, text_benign_features, num_clicks=num_clicks)  # torch.Size([4, 1, 128, 128, 128])

                    # add by bryce; for feed into ViT3D
                    if epoch > 10 and self.args.is_train_ViT3D:
                        RoI_img = self.get_RoI_img_from_pred_mask(image3D, gt3D, prev_masks, target_size=(128,128,128), padding=5)
                        outputs = self.model_ViT3D(RoI_img)
                        loss_for_classification = criterion(outputs, binary_label)
                        
                        train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                        train_targets.extend(binary_label.cpu().numpy())

                epoch_loss += loss.item()
                epoch_dice += self.get_dice_score(prev_masks,gt3D) 
                # add by bryce
                if epoch > 10 and self.args.is_train_ViT3D:
                    cur_loss = loss.item() + loss_for_classification.item() # changed by bryce
                    epoch_loss += 10 * loss_for_classification.item()
                else:
                    cur_loss = loss.item() 

            if step % self.args.accumulation_steps == 0 and step != 0:
                print_loss = step_loss / self.args.accumulation_steps
                step_loss = 0
                print_dice = self.get_dice_score(prev_masks, gt3D)
            else:
                step_loss += cur_loss

            # if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
            #     if step % self.args.accumulation_steps == 0 and step != 0:
            #         print(f'Epoch: {epoch}, Step: {step}, Loss: {print_loss}, Dice: {print_dice}')
            #         if print_dice > self.step_best_dice:
            #             self.step_best_dice = print_dice
            #             if print_dice > 0.9:
            #                 self.save_checkpoint(
            #                     epoch,
            #                     sam_model.state_dict(),
            #                     describe=f'{epoch}_step_dice:{print_dice}_best'
            #                 )
            #         if print_loss < self.step_best_loss:
            #             self.step_best_loss = print_loss
            
        epoch_loss /= step+1
        epoch_dice /= step+1
        # add by bryce
        if self.args.is_train_ViT3D:
            epoch_acc = accuracy_score(train_targets, train_preds)
            print(train_preds)
            print(train_targets)
        else:
            epoch_acc=None
        # print(self.model_ViT3D.head.parameters())
        return epoch_loss, epoch_iou, epoch_dice, pred_list, epoch_acc
    
    def plot_result(self, plot_data, description, save_name):
        plt.plot(plot_data)
        plt.title(description)
        plt.xlabel('Epoch')
        plt.ylabel(f'{save_name}')
        plt.savefig(join(MODEL_SAVE_PATH, f'{save_name}.png'))
        plt.close()


    def train(self):
        self.scaler = amp.GradScaler()
        for epoch in range(self.start_epoch, self.args.num_epochs):
            print(f'Epoch: {epoch}/{self.args.num_epochs - 1}')

            if self.args.multi_gpu:
                dist.barrier()
                self.dataloaders.sampler.set_epoch(epoch)
            num_clicks = np.random.randint(1, 11)
            epoch_loss, epoch_iou, epoch_dice, pred_list, epoch_acc = self.train_epoch(epoch, num_clicks)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if self.args.multi_gpu:
                dist.barrier()
        
            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                self.losses.append(epoch_loss)
                self.dices.append(epoch_dice)
                print(f'EPOCH: {epoch}, Loss: {epoch_loss}')
                print(f'EPOCH: {epoch}, Dice: {epoch_dice}')
                print(f'EPOCH: {epoch}, Acc: {epoch_acc}')
                logger.info(f'Epoch\t {epoch}\t : loss: {epoch_loss}, dice: {epoch_dice}, Acc: {epoch_acc}')

                if self.args.multi_gpu:
                    state_dict = self.model.module.state_dict()
                else:
                    state_dict = self.model.state_dict()
                
                # save latest checkpoint
                self.save_checkpoint(
                    epoch, 
                    state_dict, 
                    describe='latest'
                )

                self.plot_result(self.losses, 'Dice + Cross Entropy Loss', 'Loss')
                self.plot_result(self.dices, 'Dice', 'Dice')

                # for validation stage
                if epoch % self.args.val_interval == 0:
                    epoch_loss, epoch_iou, epoch_dice, pred_list, epoch_acc = self.eval_epoch(epoch, 5)

                    # save train loss best checkpoint
                    if epoch_loss < self.best_loss: 
                        self.best_loss = epoch_loss
                        self.save_checkpoint(
                            epoch,
                            state_dict,
                            describe='loss_best'
                        )
                    
                    # save train dice best checkpoint
                    if epoch_dice > self.best_dice: 
                        self.best_dice = epoch_dice
                        self.save_checkpoint(
                            epoch,
                            state_dict,
                            describe='dice_best'
                        )

                    print("=============== valid ===============")
                    print(f'EPOCH: {epoch}, Loss: {epoch_loss}')
                    print(f'EPOCH: {epoch}, Dice: {epoch_dice}')
                    print(f'EPOCH: {epoch}, Acc: {epoch_acc}')
                    print(f'Best dice: {self.best_dice}')
                    print("============= valid  End==============")

        # print(f'Best dice: {self.best_dice}')
        logger.info('=====================================================================')
        logger.info(f'Best loss: {self.best_loss}')
        logger.info(f'Best dice: {self.best_dice}')
        logger.info(f'Total loss: {self.losses}')
        logger.info(f'Total dice: {self.dices}')
        logger.info('=====================================================================')
        logger.info(f'args : {self.args}')
        logger.info(f'Used datasets : {img_datas}')
        logger.info('=====================================================================')

def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True
        
def device_config(args):
    try:
        if not args.multi_gpu:
            # Single GPU
            if args.device == 'mps':
                args.device = torch.device('mps')
            else:
                args.device = torch.device(f"cuda:{args.gpu_ids[0]}")
        else:
            args.nodes = 1
            args.ngpus_per_node = len(args.gpu_ids)
            args.world_size = args.nodes * args.ngpus_per_node

    except RuntimeError as e:
        print(e)


def main():
    mp.set_sharing_strategy('file_system')
    device_config(args)
    if args.multi_gpu:
        mp.spawn(
            main_worker,
            nprocs=args.world_size,
            args=(args, )
        )
    else:
        random.seed(2025)
        np.random.seed(2025)
        torch.manual_seed(2025)
        # Load datasets
        dataloaders = get_dataloaders(args)
        dataloaders_val = get_dataloaders_val(args)
        # Build model
        model = build_model(args)

        # add by bryce 
        ViT3D_pre_trained_model_path = '/home/haojing/workplace/MICCAI25/ViT3D_baseline/ViT_B_pretrained_noaug_mae75_BRATS2023_IXI_OASIS3_seed_8456_999_077000.pth.tar'
        model_ViT3D = load_ViT3D_pretrained_model(ViT3D_pre_trained_model_path, n_classes=2)
        M3D_tokenlizer, M3D_CLIP_Model = init_M3D_CLIP_model("/home/haojing/workplace/MICCAI25/SAM-Med3D-with-ViT3D/M3D/M3D-CLIP")
        
        
        # Create trainer
        trainer = BaseTrainer(model, model_ViT3D, M3D_tokenlizer, M3D_CLIP_Model, dataloaders, dataloaders_val, args)
        # Train
        trainer.train()

def main_worker(rank, args):
    setup(rank, args.world_size)

    torch.cuda.set_device(rank)
    args.num_workers = int(args.num_workers / args.ngpus_per_node)
    args.device = torch.device(f"cuda:{rank}")
    args.rank = rank

    init_seeds(2023 + rank)

    cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO if rank in [-1, 0] else logging.WARN,
        filemode='w',
        filename=os.path.join(LOG_OUT_DIR, f'output_{cur_time}.log'))
    
    dataloaders = get_dataloaders(args)
    model = build_model(args)
    trainer = BaseTrainer(model, dataloaders, args)
    trainer.train()
    cleanup()


def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://127.0.0.1:{args.port}',
        world_size=world_size,
        rank=rank
    )

def cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
