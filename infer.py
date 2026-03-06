import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import cv2
import json
import re
from PIL import Image
import random
from sklearn.metrics import roc_curve

# =========================
# 配置区域（请根据实际路径修改）
# =========================
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

test_img_root = "/root/autodl-tmp/MAVOC_boilerplate/test"

model1_1_path = "/root/autodl-tmp/MAVOC_boilerplate/MAVOC_boilerplate/PBVS2025_SAR_Classification_WangLab/cross_model_2resnet.pth"
model1_2_path = "/root/autodl-tmp/MAVOC_boilerplate/MAVOC_boilerplate/PBVS2025_SAR_Classification_WangLab/cross_model_1.pth"
model2_path = "model_complete.pth"

mapping_path = "./classes.json"

# 验证集F1分数
f1_model1 = [0.3, 0.12, 0.2149, 0.3158, 0.2451, 0.055, 0.4795, 0.5926, 0.7836, 0.0]
f1_model2 = [0.0172, 0.1341, 0.1769, 0.338, 0.0462, 0.4805, 0.1622, 0.5867, 0.8603, 0.0435]

fusion_method = "weighted_f1"
global_alpha = 0.3

output_csv = "results.csv"

def normalize_output(output):
    mean = output.mean(dim=0, keepdim=True)
    std = output.std(dim=0, keepdim=True)
    return (output - mean) / (std + 1e-5)

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
            nn.Dropout(dropout*0.5)
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

class TestDatasetModel1(data.Dataset):
    def __init__(self, img_root, transform=None):
        self.img_root = img_root
        self.transform = transform
        self.image_files = [f for f in os.listdir(img_root) if f.endswith('.png')]
        self.image_files.sort()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_root, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        match = re.search(r'\d+', img_name)
        image_id = int(match.group()) if match else idx
        return img, image_id

class TestDatasetModel2(data.Dataset):
    def __init__(self, img_root, transform=None):
        self.img_root = img_root
        self.transform = transform
        self.image_files = [f for f in os.listdir(img_root) if f.endswith('.png')]
        self.image_files.sort()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_root, img_name)
        img = Image.open(img_path).convert('L')
        if self.transform:
            img = self.transform(img)
        match = re.search(r'\d+', img_name)
        image_id = int(match.group()) if match else idx
        return img, image_id

# =========================
# 数据预处理
# =========================
inf_transform1 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

inf_transform2 = transforms.Compose([
    transforms.Resize((56, 56)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# =========================
# 融合权重计算函数
# =========================
def compute_weights_from_f1(f1_1, f1_2, method='weighted_f1', global_alpha=0.3):
    f1_1 = np.array(f1_1, dtype=float)
    f1_2 = np.array(f1_2, dtype=float)
    eps = 1e-8

    if method == 'weighted_f1':
        total = f1_1 + f1_2 + eps
        w1 = f1_1 / total
        w2 = f1_2 / total
    elif method == 'best_selector':
        w1 = (f1_1 > f1_2).astype(float)
        w2 = 1 - w1
    elif method == 'global_weight':
        w1 = np.full_like(f1_1, global_alpha)
        w2 = np.full_like(f1_2, 1 - global_alpha)
    elif method == 'softmax_f1':
        stacked = np.vstack([f1_1, f1_2]).T
        from scipy.special import softmax
        weights = softmax(stacked, axis=1)
        w1 = weights[:, 0]
        w2 = weights[:, 1]
    else:
        raise ValueError(f"Unknown method: {method}")
    return w1, w2

def main():
    print(f"Using device: {device}")

    with open(mapping_path, 'r') as f:
        class2id = json.load(f)
    num_classes = len(class2id)
    id2class = {v: k for k, v in class2id.items()}
    print(f"Number of classes: {num_classes}")

    model1_1 = torch.load(model1_1_path, map_location=device)
    model1_2 = torch.load(model1_2_path, map_location=device)
    model1_1.to(device).eval()
    model1_2.to(device).eval()
    print("Model1 (ensemble) loaded.")

    model2 = torch.load(model2_path, map_location=device)
    model2.to(device).eval()
    print("Model2 loaded.")

    dataset1 = TestDatasetModel1(test_img_root, transform=inf_transform1)
    loader1 = data.DataLoader(dataset1, batch_size=64, shuffle=True, num_workers=4)

    dataset2 = TestDatasetModel2(test_img_root, transform=inf_transform2)
    loader2 = data.DataLoader(dataset2, batch_size=512, shuffle=False, num_workers=4)

    results1 = {}
    print("Inferencing model1...")
    with torch.no_grad():
        for imgs, ids in tqdm(loader1):
            imgs = imgs.to(device)
            out1 = normalize_output(model1_1(imgs))
            out2 = normalize_output(model1_2(imgs))
            logits = 0.82 * out1 + 0.18 * out2
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            scores, preds = torch.max(logits, dim=1)
            scores = scores.cpu().numpy()
            for img_id, prob, score in zip(ids, probs, scores):
                if torch.is_tensor(img_id):
                    img_id = img_id.item()
                results1[img_id] = {'probs': prob, 'score': score}

    results2 = {}
    print("Inferencing model2...")
    with torch.no_grad():
        for imgs, ids in tqdm(loader2):
            imgs = imgs.to(device)
            logits = model2(imgs, mode='sar')
            probs_tensor = torch.softmax(logits, dim=1)
            scores, preds = torch.max(probs_tensor, dim=1)
            probs = probs_tensor.cpu().numpy()
            scores = scores.cpu().numpy()
            # preds = preds.cpu().numpy()
            for img_id, prob, score in zip(ids, probs, scores):
                if torch.is_tensor(img_id):
                    img_id = img_id.item()
                results2[img_id] = {'probs': prob, 'score': score}

    common_ids = sorted(set(results1.keys()) & set(results2.keys()))
    print(f"Found {len(common_ids)} common images.")

    w1, w2 = compute_weights_from_f1(f1_model1, f1_model2, fusion_method, global_alpha)
    print("Fusion weights computed.")

    final_records = []
    for img_id in common_ids:
        prob1 = results1[img_id]['probs']
        prob2 = results2[img_id]['probs']
        fused_probs = w1 * prob1 + w2 * prob2
        new_pred = int(np.argmax(fused_probs))
        final_score = results1[img_id]['score']
        final_records.append({
            'image_id': img_id,
            'class_id': new_pred,
            'score': final_score
        })

    final_df = pd.DataFrame(final_records)
    final_df = final_df.sort_values('image_id').reset_index(drop=True)
    final_df.to_csv(output_csv, index=False)
    print(f"Fusion completed. Results saved to {output_csv}")

if __name__ == "__main__":
    main()