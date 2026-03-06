import os
import json
import argparse
import random
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Sampler
from torchvision import datasets, transforms
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt

classes = [
    "sedan", "SUV", "pickup_truck", "van", "box_truck",
    "motorcycle", "flatbed_truck", "bus",
    "pickup_truck_w_trailer", "semi_w_trailer"
]


def get_label_smoothing_loss(logits, targets, smoothing=0.1):
    n_classes = logits.size(1)
    log_probs = F.log_softmax(logits, dim=1)
    with torch.no_grad():
        smooth_targets = torch.full_like(log_probs, smoothing / (n_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
    return torch.mean(-torch.sum(smooth_targets * log_probs, dim=1))


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class PairedEoSarDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, split='train', eo_size=32, sar_size=56,
                 classes=classes, use_augmentation=False, aug_strength='medium'):
        assert split in ['train', 'val']
        self.split = split
        self.eo_size = eo_size
        self.sar_size = sar_size
        self.use_augmentation = use_augmentation and (split == 'train')
        self.aug_strength = aug_strength

        df = pd.read_csv(csv_path)
        df = df[df['split'] == split].reset_index(drop=True)
        if len(df) == 0:
            raise RuntimeError(f"No samples for split={split}")
        self.df = df

        if classes is None:
            print("use new lacsses mapping!")
            classes = sorted(df['label'].unique().tolist())
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        self.targets = []
        self.entities = []
        for _, row in df.iterrows():
            label = row['label']
            cluster_id = row['cluster_id']
            self.targets.append(self.class_to_idx[label])
            self.entities.append(cluster_id)

        print(f"✓ Dataset loaded: {split}, {len(self.df)} samples, {len(self.classes)} classes")
        if self.use_augmentation:
            print(f"   Augmentation: enabled (strength={aug_strength})")

    def __len__(self):
        return len(self.df)

    def _transform(self, eo_img, sar_img):
        eo_img = TF.resize(eo_img, (self.eo_size, self.eo_size))
        sar_img = TF.resize(sar_img, (self.sar_size, self.sar_size))
        eo_img = TF.to_grayscale(eo_img)
        sar_img = TF.to_grayscale(sar_img)
        return eo_img, sar_img

    def _augment(self, eo_img, sar_img):
        if random.random() > 0.5:
            eo_img = TF.hflip(eo_img)
            sar_img = TF.hflip(sar_img)
        if random.random() > 0.5:
            eo_img = TF.vflip(eo_img)
            sar_img = TF.vflip(sar_img)
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)  
            eo_img = TF.rotate(eo_img, angle, fill=0)
            sar_img = TF.rotate(sar_img, angle, fill=0)
        if random.random() > 0.5 and self.aug_strength != 'light':
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            eo_img = TF.adjust_brightness(eo_img, brightness)
            eo_img = TF.adjust_contrast(eo_img, contrast)
        return eo_img, sar_img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        eo_img = datasets.folder.default_loader(row['eo_path'])
        sar_img = datasets.folder.default_loader(row['sar_path'])
        eo_img, sar_img = self._transform(eo_img, sar_img)

        if self.use_augmentation:
            eo_img, sar_img = self._augment(eo_img, sar_img)

        # ToTensor & Normalize
        eo_tensor = TF.to_tensor(eo_img)
        sar_tensor = TF.to_tensor(sar_img)
        eo_tensor = TF.normalize(eo_tensor, [0.5], [0.5])
        sar_tensor = TF.normalize(sar_tensor, [0.5], [0.5])

        if self.use_augmentation and random.random() > 0.5:
            eo_tensor = self._random_erase(eo_tensor)
            sar_tensor = self._random_erase(sar_tensor)

        label = self.class_to_idx[row['label']]
        cluster_id = int(row['cluster_id'])

        return (sar_tensor, label, cluster_id), (eo_tensor, label, cluster_id)

    def _random_erase(self, tensor, p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)):
        if random.random() > p:
            return tensor
        _, h, w = tensor.shape
        area = h * w
        target_area = random.uniform(*scale) * area
        aspect_ratio = random.uniform(*ratio)
        er_h = int(round((target_area * aspect_ratio) ** 0.5))
        er_w = int(round((target_area / aspect_ratio) ** 0.5))
        if er_h < h and er_w < w:
            x1 = random.randint(0, h - er_h)
            y1 = random.randint(0, w - er_w)
            tensor[:, x1:x1+er_h, y1:y1+er_w] = torch.randn_like(tensor[:, x1:x1+er_h, y1:y1+er_w]) * 0.5
        return tensor

class EntityBalancedSampler(Sampler):
    def __init__(self, labels, entities, batch_size):
        self.labels = np.array(labels)
        self.entities = np.array(entities)
        self.batch_size = batch_size
        self.map = defaultdict(lambda: defaultdict(list))
        for idx, (c, e) in enumerate(zip(self.labels, self.entities)):
            self.map[c][e].append(idx)
        self.classes = list(self.map.keys())
        self.length = len(labels)

    def __iter__(self):
        indices = []
        while len(indices) < self.length:
            batch = []
            while len(batch) < self.batch_size:
                c = random.choice(self.classes)
                e = random.choice(list(self.map[c].keys()))
                idx = random.choice(self.map[c][e])
                batch.append(idx)
            indices.extend(batch)
        return iter(indices[:self.length])

    def __len__(self):
        return self.length

# ----------------------------- MixStyle -----------------------------
class MixStyle(nn.Module):
    def __init__(self, p=0.5, alpha=0.2, eps=1e-4):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x):
        if not self.training or random.random() > self.p:
            return x
        B = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        x_norm = (x - mu) / sig
        perm = torch.randperm(B).to(x.device)
        mu_perm, sig_perm = mu[perm], sig[perm]
        lmda = torch.distributions.Beta(self.alpha, self.alpha).sample((B, 1, 1, 1)).to(x.device)
        mu_mix = mu * lmda + mu_perm * (1 - lmda)
        sig_mix = sig * lmda + sig_perm * (1 - lmda)
        return x_norm * sig_mix + mu_mix

class SimpleEncoder(nn.Module):
    def __init__(self, in_channels=1, out_dim=64, input_size=32,
                 mixstyle_p=0.5, dropout=0.3, channels=(8, 16, 32)):
        super().__init__()
        c1, c2, c3 = channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True)
        )
        self.mixstyle = MixStyle(p=mixstyle_p) if mixstyle_p > 0 else nn.Identity()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        fusion_channels = c1 + c2 + c3
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_channels, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout*0.5)  # 额外dropout
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x2_up = F.interpolate(x2, size=x3.shape[2:], mode='bilinear', align_corners=False)
        x1_up = F.interpolate(x1, size=x3.shape[2:], mode='bilinear', align_corners=False)
        fused = torch.cat([x3, x2_up, x1_up], dim=1)
        fused = self.mixstyle(fused)
        pooled = self.global_pool(fused).flatten(1)
        features = self.head(pooled)
        return features

class DistillationNet(nn.Module):
    def __init__(self, num_classes, encoder_dim=64, mixstyle_p=0.5,
                 sar_channels=(8,16,32), eo_channels=(8,16,32)):
        super().__init__()
        self.sar_encoder = SimpleEncoder(in_channels=1, out_dim=encoder_dim,
                                         input_size=56, mixstyle_p=mixstyle_p,
                                         channels=sar_channels)
        self.eo_encoder = SimpleEncoder(in_channels=1, out_dim=encoder_dim,
                                        input_size=32, mixstyle_p=mixstyle_p,
                                        channels=eo_channels)
        self.sar_classifier = nn.Linear(encoder_dim, num_classes)
        self.eo_classifier = nn.Linear(encoder_dim, num_classes)

    def forward(self, x, mode='sar'):
        if mode == 'sar':
            features = self.sar_encoder(x)
            logits = self.sar_classifier(features)
        elif mode == 'eo':
            features = self.eo_encoder(x)
            logits = self.eo_classifier(features)
        else:
            raise ValueError
        return logits

    def get_features(self, x, mode='sar'):
        if mode == 'sar':
            return self.sar_encoder(x)
        else:
            return self.eo_encoder(x)

class CachedValDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, sar_root, class_to_idx, device='cpu'):
        self.device = device
        df = pd.read_csv(csv_path)
        self.images = []
        self.labels = []
        self.ood_flags = []
        self.class_to_idx = class_to_idx
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((56, 56)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        for _, row in df.iterrows():
            img_path = os.path.join(sar_root, str(row['image_id']) + '.png')
            cls_name = str(row['class']).strip()
            ood = int(row['OOD_flag'])
            if os.path.isfile(img_path):
                self.images.append(img_path)
                self.ood_flags.append(ood)
                self.labels.append(class_to_idx[cls_name] if ood == 0 and cls_name in class_to_idx else -1)
        self.cached_tensors = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.cached_tensors is None:
            self.cached_tensors = []
            for path in tqdm(self.images, desc="Caching val", leave=False):
                img = datasets.folder.default_loader(path)
                self.cached_tensors.append(self.transform(img))
        return self.cached_tensors[idx], self.labels[idx], self.ood_flags[idx]

def validate_on_test_set(model, val_dataset, device, batch_size=256):
    model.eval()
    loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    all_logits, all_labels, all_ood = [], [], []
    with torch.no_grad():
        for images, labels, ood in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images, mode='sar')
            all_logits.append(logits.cpu())
            all_labels.append(labels)
            all_ood.append(ood)
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels).numpy()
    all_ood = torch.cat(all_ood).numpy()
    preds = torch.argmax(all_logits, dim=1).numpy()
    probs = torch.softmax(all_logits, dim=1).numpy()
    max_probs = probs.max(axis=1)

    id_mask = (all_ood == 0)
    id_acc, id_f1 = 0.0, 0.0
    if np.any(id_mask):
        valid_id = id_mask & (all_labels != -1)
        if np.any(valid_id):
            id_true = all_labels[valid_id]
            id_pred = preds[valid_id]
            id_acc = accuracy_score(id_true, id_pred)
            id_f1 = f1_score(id_true, id_pred, average='macro')

    auc = 0.5
    if len(np.unique(all_ood)) > 1:
        auc = roc_auc_score(all_ood, -max_probs)

    score = 0.75 * id_acc + 0.25 * auc
    return id_acc, id_f1, auc, score, (all_labels, preds, all_ood)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, 'train_log.txt')

    train_dataset = PairedEoSarDataset(
        csv_path=args.train_csv,
        split='train',
        eo_size=args.eo_size,
        sar_size=args.sar_size,
        use_augmentation=args.use_augmentation,
        aug_strength=args.aug_strength
    )
    num_classes = len(train_dataset.classes)
    with open(os.path.join(args.output_dir, 'classes.json'), 'w') as f:
        json.dump(train_dataset.class_to_idx, f, indent=4)

    val_dataset = CachedValDataset(args.test_csv, args.test_sar_root, train_dataset.class_to_idx)

    sampler = EntityBalancedSampler(train_dataset.targets, train_dataset.entities, args.batch_size)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=8, pin_memory=True, drop_last=True
    )

    sar_channels = tuple(int(x) for x in args.sar_channels.split(','))
    eo_channels = tuple(int(x) for x in args.eo_channels.split(','))
    model = DistillationNet(
        num_classes, encoder_dim=args.encoder_dim,
        mixstyle_p=args.mixstyle_p,
        sar_channels=sar_channels,
        eo_channels=eo_channels
    ).to(device)

    if args.loss_type == 'focal':
        if args.focal_alpha:
            try:
                alpha_tensor = torch.tensor(eval(args.focal_alpha), dtype=torch.float32, device=device)
            except:
                alpha_tensor = float(args.focal_alpha)
        else:
            alpha_tensor = None
        criterion = FocalLoss(gamma=args.focal_gamma, alpha=alpha_tensor)
    else:  # label_smoothing
        criterion = lambda logits, targets: get_label_smoothing_loss(logits, targets, smoothing=args.label_smoothing)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    elif args.lr_scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    else:
        scheduler = None

    scaler = torch.cuda.amp.GradScaler()

    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    best_model_path = os.path.join(args.output_dir, 'best_model.pth')

    if args.use_swa:
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        swa_start = args.epochs // 2  

    for epoch in range(args.epochs):
        model.train()
        train_stats = {'sar_preds': [], 'sar_targets': [], 'eo_preds': []}
        loss_sar_sum, loss_eo_sum, loss_consist_sum, loss_total_sum = 0, 0, 0, 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}', leave=False)
        for (sar, y, _), (eo, _, _) in pbar:
            sar, eo, y = sar.to(device), eo.to(device), y.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                feat_sar = model.get_features(sar, 'sar')
                logits_sar = model.sar_classifier(feat_sar)
                feat_eo = model.get_features(eo, 'eo')
                logits_eo = model.eo_classifier(feat_eo)

                loss_sar = criterion(logits_sar, y)
                loss_eo = criterion(logits_eo, y)

                loss_consist = (F.kl_div(F.log_softmax(logits_sar, dim=1),
                                          F.softmax(logits_eo.detach(), dim=1), reduction='batchmean') +
                                F.kl_div(F.log_softmax(logits_eo, dim=1),
                                          F.softmax(logits_sar.detach(), dim=1), reduction='batchmean')) / 2

                loss = loss_sar + loss_eo + args.consist_weight * loss_consist

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_sar_sum += loss_sar.item()
            loss_eo_sum += loss_eo.item()
            loss_consist_sum += loss_consist.item()
            loss_total_sum += loss.item()

            with torch.no_grad():
                train_stats['sar_preds'].append(logits_sar.argmax(1).cpu().numpy())
                train_stats['sar_targets'].append(y.cpu().numpy())
                train_stats['eo_preds'].append(logits_eo.argmax(1).cpu().numpy())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        if args.use_swa and epoch >= swa_start:
            swa_model.update_parameters(model)

        sar_preds = np.concatenate(train_stats['sar_preds'])
        sar_targets = np.concatenate(train_stats['sar_targets'])
        eo_preds = np.concatenate(train_stats['eo_preds'])
        sar_acc, sar_f1 = accuracy_score(sar_targets, sar_preds), f1_score(sar_targets, sar_preds, average='macro')
        eo_acc, eo_f1 = accuracy_score(sar_targets, eo_preds), f1_score(sar_targets, eo_preds, average='macro')

        val_acc, val_f1, val_auc, val_score, (true_labels, preds, ood_flags) = validate_on_test_set(
            model, val_dataset, device, args.batch_size
        )

        id_mask = (ood_flags == 0) & (true_labels != -1)
        if np.any(id_mask):
            id_true = true_labels[id_mask]
            id_pred = preds[id_mask]
            class_f1 = f1_score(id_true, id_pred, average=None, labels=list(range(num_classes)))
            class_f1_str = ' '.join([f'c{i}:{f1:.3f}' for i, f1 in enumerate(class_f1)])
        else:
            class_f1_str = 'No ID samples'

        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_score)  
            else:
                scheduler.step()

        log_str = (f"[{epoch+1:03d}] Loss: {loss_total_sum/len(train_loader):.4f} "
                   f"(SAR:{loss_sar_sum/len(train_loader):.4f} EO:{loss_eo_sum/len(train_loader):.4f} "
                   f"Consist:{loss_consist_sum/len(train_loader):.4f}) | "
                   f"Train EO [Acc:{eo_acc:.3f} F1:{eo_f1:.3f}] | "
                   f"Train SAR [Acc:{sar_acc:.3f} F1:{sar_f1:.3f}] | "
                   f"Val [Acc(ID):{val_acc:.3f} F1:{val_f1:.3f} AUC:{val_auc:.3f} Score:{val_score:.3f}]\n"
                   f"   Class F1: {class_f1_str}")
        print(log_str)
        with open(log_file, 'a') as f:
            f.write(log_str + '\n')

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"✓ Best model saved with F1: {val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"Training finished. Best F1: {best_val_f1:.4f} at epoch {best_epoch+1}")
    model.load_state_dict(torch.load(best_model_path))

    final_acc, final_f1, final_auc, final_score, (true_labels, preds, ood_flags) = validate_on_test_set(
        model, val_dataset, device, args.batch_size
    )

    print(f"\n{'='*60}")
    print(f"=== Final Validation Results ===")
    print(f"Acc@1 (ID): {final_acc:.4f}")
    print(f"Macro F1:   {final_f1:.4f}")
    print(f"AUC (OOD):  {final_auc:.4f}")
    print(f"Score:      {final_score:.4f}")
    print(f"{'='*60}\n")

    id_mask = (np.array(ood_flags) == 0) & (np.array(true_labels) != -1)
    if np.any(id_mask):
        cm = confusion_matrix(true_labels[id_mask], preds[id_mask])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(train_dataset.class_to_idx.keys()))
        fig, ax = plt.subplots(figsize=(10, 8))
        disp.plot(ax=ax, xticks_rotation='vertical')
        plt.title("Confusion Matrix (ID samples)")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'), dpi=150)

    with open(os.path.join(args.output_dir, 'final_results.json'), 'w') as f:
        json.dump({
            'final_acc_id': float(final_acc),
            'final_f1': float(final_f1),
            'final_auc': float(final_auc),
            'final_score': float(final_score),
            'best_val_f1': float(best_val_f1),
            'best_epoch': int(best_epoch)
        }, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--train_csv', type=str, default='./entity_split1.csv')
    parser.add_argument('--test_csv', type=str, default='./validation_reference.csv')
    parser.add_argument('--test_sar_root', type=str, default='/root/autodl-tmp/MAVOC_boilerplate/val/')
    parser.add_argument('--output_dir', type=str, default='./')
    
    parser.add_argument('--eo_size', type=int, default=32)
    parser.add_argument('--sar_size', type=int, default=56)

    parser.add_argument('--sar_channels', type=str, default='8,16,32', help='Comma-separated channels for SAR encoder')
    parser.add_argument('--eo_channels', type=str, default='8,16,32', help='Comma-separated channels for EO encoder')
    parser.add_argument('--encoder_dim', type=int, default=64, help='Feature dimension after fusion')
    parser.add_argument('--mixstyle_p', type=float, default=0.5, help='MixStyle probability')

    parser.add_argument('--use_augmentation', action='store_true', default=True, help='Enable data augmentation')
    parser.add_argument('--aug_strength', type=str, default='medium', choices=['light', 'medium', 'strong'])

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=512)  
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-2)  
    parser.add_argument('--patience', type=int, default=7)

    parser.add_argument('--loss_type', type=str, default='label_smoothing', choices=['focal', 'label_smoothing'])
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Smoothing factor')
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--focal_alpha', type=str, default=None, help='e.g., "[0.5,0.5,...]" or float')

    parser.add_argument('--consist_weight', type=float, default=0.2)  

    parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['none', 'cosine', 'plateau'])

    parser.add_argument('--use_swa', action='store_true', default=True, help='Use Stochastic Weight Averaging')

    args = parser.parse_args()
    train(args)
