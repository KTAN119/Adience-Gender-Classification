import os

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader, Dataset
import torchvision
from torchvision import transforms
from sklearn.model_selection import train_test_split
from pytorchcv.model_provider import get_model

from PIL import Image
from tqdm import tqdm

import pandas as pd

data_0 = pd.read_csv('fold_0_data.txt', delimiter='\t')
data_1 = pd.read_csv('fold_1_data.txt', delimiter='\t')
data_2 = pd.read_csv('fold_2_data.txt', delimiter='\t')
data_3 = pd.read_csv('fold_3_data.txt', delimiter='\t')
data_4 = pd.read_csv('fold_4_data.txt', delimiter='\t')
data = [data_0, data_1, data_2, data_3, data_4]
data = pd.concat(data)

train_df, valid_df = train_test_split(data, test_size=0.2, random_state=42)


class GenderDataset(Dataset):
    def __init__(self, root_dir, df, transform=None):
        self.root_dir = root_dir
        self.subdir = df['user_id'].tolist()
        self.face_id = df['face_id'].tolist()
        self.filename = df['original_image'].tolist()
        self.label = df['gender'].tolist()
        self.transform = transform

    def __getitem__(self, idx):
        filename = 'landmark_aligned_face.' + \
            str(self.face_id[idx]) + '.' + self.filename[idx]
        img_path = os.path.join(self.root_dir, self.subdir[idx], filename)
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        label = self.label[idx]
        if label == 'm':
            gender = 0
        else:
            gender = 1

        return img, gender

    def __len__(self):
        return len(self.label)


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Load pretrained network as backbone
        pretrained = get_model('efficientnet_b5b', pretrained=True)
        self.backbone = pretrained.features
        self.output = pretrained.output
        self.classifier = nn.Linear(1000, 2)

        del pretrained

    def forward(self, x):
        x = self.backbone(x)
        x = x.reshape(x.size(0), -1)
        x = self.output(x)
        x = self.classifier(x)

        return x

    def freeze_backbone(self):
        """Freeze the backbone network weight"""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        """Freeze the backbone network weight"""
        for p in self.backbone.parameters():
            p.requires_grad = True


def accuracy(prediction, ground_truth):
    num_correct = (np.array(prediction) == np.array(ground_truth)).sum()
    return num_correct / len(prediction)


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.1, contrast=0.1,
                           saturation=0.1, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_ds = GenderDataset(
    '/home/students/acct2014_04/age-gender-data/aligned', train_df, transform=train_transform)
valid_ds = GenderDataset(
    '/home/students/acct2014_04/age-gender-data/aligned', valid_df, transform=valid_transform)

EPOCHS = 2
BATCH_SIZE = 64
LR = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                      shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE,
                      num_workers=4, pin_memory=True)

model = CNNModel()
model.freeze_backbone()
model = nn.DataParallel(model).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, len(train_dl), T_mult=EPOCHS*len(train_dl))
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()

    for img, label in tqdm(train_dl):
        img = img.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        logits = model(img)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

    model.eval()

    predictions = []
    ground_truths = []

    for img, label in tqdm(valid_dl):
        img = img.to(device)
        with torch.no_grad():
            logits = model(img)
            prediction = torch.argmax(logits, dim=1)

            predictions.extend(prediction.tolist())
            ground_truths.extend(label.tolist())

    acc = accuracy(predictions, ground_truths)
    print('valid_acc = {}'.format(acc))

EPOCHS = 40
BATCH_SIZE = 64
LR = 25e-6

model = CNNModel()
model.unfreeze_backbone()
model = nn.DataParallel(model).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-7)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, len(train_dl), T_mult=len(train_dl)*EPOCHS)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    for img, label in tqdm(train_dl):
        img = img.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        logits = model(img)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        scheduler.step()

    model.eval()

    predictions = []
    ground_truths = []
    for img, label in tqdm(valid_dl):
        img = img.to(device)
        with torch.no_grad():
            logits = model(img)
            prediction = torch.argmax(logits, dim=1)

            predictions.extend(prediction.tolist())
            ground_truths.extend(label.tolist())

    acc = accuracy(predictions, ground_truths)
    print('valid_acc = {}'.format(acc))

    if acc > 0.95:
        torch.save(model.state_dict(),
                   'CNN_Weight/weightsb5b_epoch_{}_acc_{}.pth'.format(epoch, acc))
