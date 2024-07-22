import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from feature_multiprocess import get_feature
from net1 import AudioClassifier
from data_loader import get_dataloader
import utils

import warnings
warnings.filterwarnings('ignore')

class Config:
    SR = 32000
    
    # Dataset
    ROOT_META = "train_final.csv"
    # weight 저장용 폴더의 path(없으면 만드셈)
    WEIGHT_DIR = "model_weight"

    # Training
    BATCH_SIZE = 32
    N_EPOCHS = 50
    LR = 1e-4
    
    # Others
    SEED = 42
    SERIAL = utils.get_serial()
    USE_RATIO = 1 # 사용 할 데이터 양
    
CONFIG = Config()

utils.seed_everything(CONFIG.SEED) # Seed 고정
utils.ensure_directory_exists(f"{CONFIG.WEIGHT_DIR}/{CONFIG.SERIAL}")
utils.save_config(CONFIG)

whole_df = pd.read_csv(CONFIG.ROOT_META)
whole_df = whole_df.sample(frac=CONFIG.USE_RATIO, random_state=CONFIG.SEED)
print(f"use {whole_df.shape[0]}")

# train_data := list of (cnn_feature, rnn_feature)
# cnn_feature := image_sequence, each image is of shape (224, 224, 6)
# rnn_feature := feature of shape (Value depending on the sound file, 67)
train_data_df = whole_df[whole_df['path'].str.contains('train')]
validate_data_df = whole_df[whole_df['path'].str.contains('validate')]
print(f"The number of train_data_df   : {train_data_df.shape[0]}")
print(f"The number of validate_data_df: {validate_data_df.shape[0]}")

train_data, train_labels = get_feature(df=train_data_df, train_mode=True)
validate_data, val_labels = get_feature(df=validate_data_df, train_mode=True)
cnn_channel_num = train_data[0][0][0].shape[-1]
rnn_feature_num = train_data[0][1].shape[-1]

train_loader = get_dataloader(dataset=train_data, labels=train_labels, batch_size=CONFIG.BATCH_SIZE, shuffle=True)
val_loader = get_dataloader(dataset=validate_data, labels=val_labels, batch_size=CONFIG.BATCH_SIZE, shuffle=False)

def train(model, optimizer, scheduler, train_loader, val_loader, device):
    model.to(device)
    criterion = nn.BCELoss().to(device)
    
    result = pd.DataFrame(columns=['epoch', 'Train_Loss', 'Val_Loss', 'Val_AUC', 'lr'])
    best_val_score = 0
    best_model = None
    
    for epoch in range(1, CONFIG.N_EPOCHS+1):
        model.train()
        train_loss = []
        for x_cnn, x_len_cnn, x_rnn, x_len_rnn, labels in tqdm(train_loader):
            # x_cnn.shape     := (CONFIG.BATCH_SIZE, mini batch에서 길이가 가장 긴 image_sequence의 길이, 6, 224, 224)
            # x_len_cnn.shape := (CONFIG.BATCH_SIZE)
            x_cnn = x_cnn.float().to(device)
            x_rnn = x_rnn.float().to(device)
            labels = labels.float().to(device)
            
            optimizer.zero_grad()
            
            output = model(x_cnn, x_len_cnn, x_rnn, x_len_rnn)
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss.append(loss.item())

        _val_loss, _val_score = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val AUC : [{_val_score:.5f}]')
        result.loc[epoch] =[f"{epoch}", f"{_train_loss:.5f}", f"{_val_loss:.5f}", f"{_val_score:.5f}", f"{scheduler.get_last_lr()[0]}"]

        if best_val_score < _val_score:
            best_val_score = _val_score
            best_model = model
        
        # save weight
        tmp = CONFIG.WEIGHT_DIR+f"/{CONFIG.SERIAL}/{epoch}.pth"
        print(f"saving {tmp}")
        torch.save(model.state_dict(), tmp)
    
    print(f"saving best model and result")
    torch.save(best_model.state_dict(), CONFIG.WEIGHT_DIR+f"/{CONFIG.SERIAL}/best.pth")
    result.to_csv(CONFIG.WEIGHT_DIR+f"/{CONFIG.SERIAL}/result.csv", index=False)
    return best_model

def multiLabel_AUC(y_true, y_scores):
    auc_scores = []
    for i in range(y_true.shape[1]):
        auc = roc_auc_score(y_true[:, i], y_scores[:, i])
        auc_scores.append(auc)
    mean_auc_score = np.mean(auc_scores)
    return mean_auc_score
    
def validation(model, criterion, val_loader, device):
    print("validating")
    model.eval()
    val_loss, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for x_cnn, x_len_cnn, x_rnn, x_len_rnn, labels in tqdm(val_loader):
            # x_cnn.shape     := (CONFIG.BATCH_SIZE, mini batch에서 길이가 가장 긴 image_sequence의 길이, 6, 224, 224)
            # x_len_cnn.shape := (CONFIG.BATCH_SIZE)
            x_cnn = x_cnn.float().to(device)
            x_rnn = x_rnn.float().to(device)
            labels = labels.float().to(device)
            
            probs = model(x_cnn, x_len_cnn, x_rnn, x_len_rnn)
            loss = criterion(probs, labels)
            
            val_loss.append(loss.item())

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
        
        _val_loss = np.mean(val_loss)

        all_labels = np.concatenate(all_labels, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)
        
        # Calculate AUC score
        auc_score = multiLabel_AUC(all_labels, all_probs)
    
    return _val_loss, auc_score

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = AudioClassifier(cnn_channel_num=cnn_channel_num, rnn_feature_num=rnn_feature_num)
optimizer = torch.optim.Adam(params = model.parameters(), lr = CONFIG.LR)

# 데이터 병렬 처리
if torch.cuda.device_count() > 1:
    print(f"using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

model.to(device)
print(f"Using device: {device}")

criterion = nn.BCELoss().to(device)
optimizer = optim.AdamW(model.parameters(), lr=CONFIG.LR)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CONFIG.LR, steps_per_epoch=len(train_loader), epochs=CONFIG.N_EPOCHS)

best_model = train(model, optimizer, scheduler, train_loader, val_loader, device)