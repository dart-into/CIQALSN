import os
import random
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import clip


class IQADataset(Dataset):
    """
    IQA data
    """
    def __init__(self, df: pd.DataFrame, image_dir: str, preprocess):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.preprocess = preprocess

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name   = str(row.iloc[0]).strip()
        text_desc  = str(row.iloc[1]).strip()
        quality    = float(row.iloc[2])
        attributes = [float(row.iloc[i]) for i in range(3, 8)]

        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        image = self.preprocess(image)

        text_token = clip.tokenize([text_desc], truncate=True).squeeze(0)
        quality_tensor = torch.tensor(quality, dtype=torch.float32)
        attr_tensor    = torch.tensor(attributes, dtype=torch.float32)

        return image, text_token, quality_tensor, attr_tensor



class FusionRegressorNet(nn.Module):
    """image-text-fusion，input (img_norm, txt_norm, attr_pred, quality_pred)"""
    def __init__(self, clip_model_name: str = 'ViT-B/32', device: str = 'cuda'):
        super(FusionRegressorNet, self).__init__()

        self.clip_model, _ = clip.load(clip_model_name, device=device)
        self.clip_model = self.clip_model.float()

        # frozen
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # train the latest 4 layers（for the student model）
        self._unfreeze_last_n_visual_layers(n=4)

        img_dim = self.clip_model.visual.output_dim
        txt_dim = self.clip_model.text_projection.shape[1]
        fuse_dim = img_dim + txt_dim

        self.fusion_proj = nn.Sequential(
            nn.Linear(fuse_dim, 1000),
            nn.LayerNorm(1000),
            nn.GELU(),
        )

        self.backbone = nn.Sequential(
            nn.Linear(1000, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.attr_head = nn.Linear(64, 6)          # the former 5 are attributes
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1.0 / 0.07))

        self._print_trainable_params()

    def _unfreeze_last_n_visual_layers(self, n: int = 4):
        visual = self.clip_model.visual
        if hasattr(visual, 'transformer'):
            blocks = visual.transformer.resblocks
            total = len(blocks)
            for i in range(total - n, total):
                for param in blocks[i].parameters():
                    param.requires_grad = True
            for param in visual.ln_post.parameters():
                param.requires_grad = True
            if visual.proj is not None:
                visual.proj.requires_grad = True
        elif hasattr(visual, 'layer4'):
            for param in visual.layer4.parameters():
                param.requires_grad = True
            if hasattr(visual, 'attnpool'):
                for param in visual.attnpool.parameters():
                    param.requires_grad = True
        else:
            print("[Warning] undentified Visual Encoder")

    def _print_trainable_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Info] total parameters: {total:,} | state: {trainable:,} "
              f"({100.0 * trainable / total:.2f}%)")

    def forward(self, images, texts):
        img_feat = self.clip_model.encode_image(images)
        txt_feat = self.clip_model.encode_text(texts)
        img_feat_norm = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-8)
        txt_feat_norm = txt_feat / (txt_feat.norm(dim=-1, keepdim=True) + 1e-8)

        fusion = torch.cat([img_feat, txt_feat], dim=-1)
        fusion = self.fusion_proj(fusion)
        feat_64 = self.backbone(fusion)
        attr_p = self.attr_head(feat_64)          # [B, 6]
        attr_pred = attr_p[:, :5]                 # attributes（index 0~4）
        quality_pred = attr_p[:, 5]               # quality（index 5）
        return img_feat_norm, txt_feat_norm, attr_pred, quality_pred



# evaluation

def compute_metrics(pred, target):
    """calculate SRCC 和 PLCC"""
    pred = np.array(pred).flatten()
    target = np.array(target).flatten()
    srcc, _ = stats.spearmanr(pred, target)
    try:
        coeffs = np.polyfit(pred, target, deg=5)
        fitted = np.polyval(coeffs, pred)
        plcc, _ = stats.pearsonr(fitted, target)
    except:
        plcc, _ = stats.pearsonr(pred, target)
    return srcc, plcc


def evaluate_student(model, teacher_model, loader, device):
    """
           'quality_srcc', 'quality_plcc' : student vs mos
        'attr_vs_teacher_srcc', 'attr_vs_teacher_plcc' : list
        'attr_vs_label_srcc', 'attr_vs_label_plcc' : list
    """
    model.eval()
    teacher_model.eval()

    all_q_pred = []
    all_q_label = []
    all_attr_pred = []   # student attributes [N,5]
    all_attr_label = []  # LLM attributes [N,5]
    all_attr_teacher = [] # teacher attributes [N,5]

    with torch.no_grad():
        for images, texts, quality_labels, attr_labels in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device)
            texts = texts.to(device)
            quality_labels = quality_labels.numpy()
            attr_labels = attr_labels.numpy()

            _, _, attr_pred, q_pred = model(images, texts)
            q_pred = q_pred.cpu().numpy()
            attr_pred = attr_pred.cpu().numpy()

            _, _, attr_teacher, _ = teacher_model(images, texts)
            attr_teacher = attr_teacher.cpu().numpy()

            all_q_pred.extend(q_pred)
            all_q_label.extend(quality_labels)
            all_attr_pred.append(attr_pred)
            all_attr_label.append(attr_labels)
            all_attr_teacher.append(attr_teacher)

    all_attr_pred = np.concatenate(all_attr_pred, axis=0)   # [N,5]
    all_attr_label = np.concatenate(all_attr_label, axis=0)
    all_attr_teacher = np.concatenate(all_attr_teacher, axis=0)

    # quality
    q_srcc, q_plcc = compute_metrics(all_q_pred, all_q_label)

    # attributes student vs teacher
    attr_vs_teacher_srcc = []
    attr_vs_teacher_plcc = []
    for k in range(5):
        srcc, plcc = compute_metrics(all_attr_pred[:, k], all_attr_teacher[:, k])
        attr_vs_teacher_srcc.append(srcc)
        attr_vs_teacher_plcc.append(plcc)

    # 3. attributes student vs LLM
    attr_vs_label_srcc = []
    attr_vs_label_plcc = []
    for k in range(5):
        srcc, plcc = compute_metrics(all_attr_pred[:, k], all_attr_label[:, k])
        attr_vs_label_srcc.append(srcc)
        attr_vs_label_plcc.append(plcc)

    return {
        'quality_srcc': q_srcc,
        'quality_plcc': q_plcc,
        'attr_vs_teacher_srcc': attr_vs_teacher_srcc,
        'attr_vs_teacher_plcc': attr_vs_teacher_plcc,
        'attr_vs_label_srcc': attr_vs_label_srcc,
        'attr_vs_label_plcc': attr_vs_label_plcc,
    }



# atrributes weight gating based on meta-learning

def train_distill(args):

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Info] device: {device}")

    # ---- download CLIP pre ----
    _, preprocess = clip.load(args.clip_model, device=device)

    # ---- write data ----
    df = pd.read_csv(args.label_file, header=None)
    print(f"[Info] total samples: {len(df)}")
    assert df.shape[1] >= 8, "CSV  at least 8 lines"

    train_df, test_df = train_test_split(
        df, test_size=args.test_size, random_state=args.seed, shuffle=True
    )
    print(f"[Info] train: {len(train_df)} | test: {len(test_df)}")

    train_dataset = IQADataset(train_df, args.image_dir, preprocess)
    test_dataset  = IQADataset(test_df,  args.image_dir, preprocess)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # ----download teacher pre ----
    teacher = FusionRegressorNet(clip_model_name=args.clip_model, device=str(device)).to(device)
    ckpt = torch.load(args.teacher_checkpoint, map_location=device)
    teacher.load_state_dict(ckpt)       teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    print(f"[Info] teacher model load: {args.teacher_checkpoint}")

    # ---- conduct student ----
    student = FusionRegressorNet(clip_model_name=args.clip_model, device=str(device)).to(device)

    # ---- optimizer with differentiated learning rates ----
    clip_params = []
    head_params = []
    for name, param in student.named_parameters():
        if not param.requires_grad:
            continue
        if 'clip_model' in name:
            clip_params.append(param)
        else:
            head_params.append(param)

    optimizer = optim.AdamW([
        {'params': clip_params, 'lr': args.lr},
        {'params': head_params, 'lr': args.lr_head},
    ], weight_decay=args.weight_decay)

    # cosine annealing + warmup
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=args.num_epochs - args.warmup_epochs, eta_min=args.min_lr)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs])

    # ===== quality gradient =====
    shared_params = list(student.fusion_proj.parameters()) + list(student.backbone.parameters())

    # ---- distillation loss (MSE) ----
    mse_criterion = nn.MSELoss(reduction='none')   

    # ---- train----
    best_q_srcc = -1.0
    best_epoch = 0

    print("\n" + "="*60)
    print(" distillation strat（weight gating） ".center(60))
    print("="*60)

    for epoch in range(1, args.num_epochs + 1):
        student.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} Train", leave=False)

        for images, texts, _, _ in pbar:
            images = images.to(device)
            texts = texts.to(device)

            _, _, attr_stu, q_stu = student(images, texts)   # attr_stu: [B,5], q_stu: [B]
            with torch.no_grad():
                _, _, attr_tea, q_tea = teacher(images, texts)  # attr_tea: [B,5], q_tea: [B]

            temp = args.temperature
            # quality: logits
            prob_tea = F.softmax(q_tea / temp, dim=0)          # [B]
            log_prob_stu = F.log_softmax(q_stu / temp, dim=0)  # [B]
            L_rank = F.kl_div(log_prob_stu, prob_tea, reduction='batchmean')

            attr_mse = mse_criterion(attr_stu, attr_tea)       # [B,5]
            L_attr_vec = attr_mse.mean(dim=0)                  # [5] 

            optimizer.zero_grad()

            # ---- grad_rank ----
            grad_rank = torch.autograd.grad(
                L_rank,
                shared_params,
                retain_graph=True,          
                allow_unused=True
            )

            flat_grad_rank = torch.cat([g.view(-1) for g in grad_rank if g is not None])

            # ----consine similarity ----
            meta_gates = []
            for k in range(5):
                L_attr_k = L_attr_vec[k]    
                grad_attr_k = torch.autograd.grad(
                    L_attr_k,
                    shared_params,
                    retain_graph=True,      
                    allow_unused=True
                )
                flat_grad_attr = torch.cat([g.view(-1) for g in grad_attr_k if g is not None])

                cos_sim = torch.dot(flat_grad_rank, flat_grad_attr) / (
                    torch.norm(flat_grad_rank) * torch.norm(flat_grad_attr) + 1e-8
                )
                #  sigmoid 
                gate = torch.sigmoid(args.meta_tau * cos_sim)
                # detach() gradient
                meta_gates.append(gate.detach())

            meta_gates = torch.stack(meta_gates)   # [5]


            optimizer.zero_grad()

            # ===== second meta-learning training =====
            _, _, attr_stu_new, q_stu_new = student(images, texts)
            
            with torch.no_grad():
                _, _, attr_tea_new, q_tea_new = teacher(images, texts)

            attr_mse_new = mse_criterion(attr_stu_new, attr_tea_new)   # [B,5]
            L_attr_vec_new = attr_mse_new.mean(dim=0)                  # [5]

            L_attr_weighted = torch.sum(meta_gates * L_attr_vec_new)   

            prob_tea_new = F.softmax(q_tea_new / temp, dim=0)
            log_prob_stu_new = F.log_softmax(q_stu_new / temp, dim=0)
            L_rank_new = F.kl_div(log_prob_stu_new, prob_tea_new, reduction='batchmean')


            loss = L_rank_new + L_attr_weighted

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in student.parameters() if p.requires_grad],
                max_norm=args.grad_clip
            )
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(train_loader)
        scheduler.step()
        cur_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch:3d}/{args.num_epochs}] LR={cur_lr:.2e} | Total Loss={avg_loss:.4f}")

        metrics = evaluate_student(student, teacher, test_loader, device)

        print(f"  [quality] SRCC={metrics['quality_srcc']:.4f}  PLCC={metrics['quality_plcc']:.4f}")

        print("  [attributes vs teacher] SRCC: " + " ".join([f"{x:.4f}" for x in metrics['attr_vs_teacher_srcc']]))
        print("                  PLCC: " + " ".join([f"{x:.4f}" for x in metrics['attr_vs_teacher_plcc']]))

        print("  [attributes vs LLM] SRCC: " + " ".join([f"{x:.4f}" for x in metrics['attr_vs_label_srcc']]))
        print("                  PLCC: " + " ".join([f"{x:.4f}" for x in metrics['attr_vs_label_plcc']]))
        print()

        if metrics['quality_srcc'] > best_q_srcc:
            best_q_srcc = metrics['quality_srcc']
            best_epoch = epoch
            save_path = os.path.join(args.save_dir, 'best_student.pth')
            torch.save({
                'epoch': epoch,
                'state_dict': student.state_dict(),
                'optimizer': optimizer.state_dict(),
                'quality_srcc': metrics['quality_srcc'],
                'quality_plcc': metrics['quality_plcc'],
            }, save_path)
            print(f"  ┗━ save best model ( SRCC={best_q_srcc:.4f}) → {save_path}")

    print("\n" + "="*60)
    print(" final test ".center(60))
    print("="*60)

    best_ckpt = torch.load(os.path.join(args.save_dir, 'best_student.pth'), map_location=device)
    student.load_state_dict(best_ckpt['state_dict'])
    final_metrics = evaluate_student(student, teacher, test_loader, device)

    print(f"\n best Epoch : {best_epoch}")
    print(f"quality SRCC  : {final_metrics['quality_srcc']:.4f}")
    print(f"quality PLCC  : {final_metrics['quality_plcc']:.4f}\n")

    print("attributes vs teacher（SRCC）:", "  ".join([f"{x:.4f}" for x in final_metrics['attr_vs_teacher_srcc']]))
    print("attributes vs teacher（PLCC）:", "  ".join([f"{x:.4f}" for x in final_metrics['attr_vs_teacher_plcc']]))
    print("attributes vs LLM（SRCC）:", "  ".join([f"{x:.4f}" for x in final_metrics['attr_vs_label_srcc']]))
    print("attributes vs LLM（PLCC）:", "  ".join([f"{x:.4f}" for x in final_metrics['attr_vs_label_plcc']]))
    print("="*60 + "\n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='meta-learning gating')

    parser.add_argument('--image_dir',   type=str,   default='/home/user/data/AGIQA3k/', help='图像目录')
    parser.add_argument('--label_file',  type=str,   default='./AGIQA_merged.csv', help='标签文件')
    parser.add_argument('--save_dir', type=str, default='./distill_checkpoints', help='保存路径')

    # 模型相关
    parser.add_argument('--clip_model', type=str, default='ViT-B/32')
    parser.add_argument('--teacher_checkpoint', type=str, default='./pre_model.pth', help='教师模型权重路径')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-5, help='CLIP 层学习率')
    parser.add_argument('--lr_head', type=float, default=1e-4, help='自定义层学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--min_lr', type=float, default=1e-7)

    parser.add_argument('--temperature', type=float, default=1.0, help='排序蒸馏温度')
    parser.add_argument('--meta_tau', type=float, default=2.0, help='元门控 sigmoid 温度系数')

    args = parser.parse_args()
    train_distill(args)