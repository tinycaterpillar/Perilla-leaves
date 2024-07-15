import pandas as pd
import torch
from tqdm import tqdm
import numpy as np

import torch.nn as nn
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import roc_auc_score

from feature import get_feature
from net import AudioClassifier
from data_loader import get_dataloader
from utils import get_serial, seed_everything

import warnings
warnings.filterwarnings('ignore')

class Config:
    SR = 32000
    
    # Dataset
    ROOT_META = "train_sample.csv"
    # weight 저장용 폴더의 path(없으면 만드셈)
    WEIGHT_DIR = "model_weight"

    # Training
    BATCH_SIZE = 32
    N_EPOCHS = 50
    LR = 3e-4
    
    # Others
    SEED = 42
    SERIAL = get_serial()
    
CONFIG = Config()

seed_everything(CONFIG.SEED) # Seed 고정

whole_df = pd.read_csv(CONFIG.ROOT_META)

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

train_loader = get_dataloader(train_data, train_labels, batch_size=CONFIG.BATCH_SIZE, shuffle=True)
val_loader = get_dataloader(validate_data, val_labels, batch_size=CONFIG.BATCH_SIZE, shuffle=True)

def train(model, optimizer, train_loader, val_loader, device):
    model.to(device)
    criterion = nn.BCELoss().to(device)
    
    best_val_score = 0
    best_model = None
    
    for epoch in range(1, CONFIG.N_EPOCHS+1):
        model.train()
        train_loss = []
        for x_cnn, x_len_cnn, packed_rnn_input, labels in tqdm(train_loader):
            # x_cnn.shape     := (CONFIG.BATCH_SIZE, mini batch에서 길이가 가장 긴 image_sequence의 길이, 6, 224, 224)
            # x_len_cnn.shape := (CONFIG.BATCH_SIZE)
            x_cnn = x_cnn.float().to(device)
            packed_rnn_input = packed_rnn_input.float().to(device)
            labels = labels.float().to(device)
            
            optimizer.zero_grad()
            
            output = model(x_cnn, x_len_cnn, packed_rnn_input)
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss.append(loss.item())

        _val_loss, _val_score = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val AUC : [{_val_score:.5f}]')
            
        if best_val_score < _val_score:
            best_val_score = _val_score
            best_model = model
        
        # save weight
        tmp = CONFIG.WEIGHT_DIR+f"/{CONFIG.SERIAL}/{epoch}.pth"
        print(f"saving {tmp}")
        torch.save(model.state_dict(), tmp)
    
    return best_model

def multiLabel_AUC(y_true, y_scores):
    auc_scores = []
    for i in range(y_true.shape[1]):
        auc = roc_auc_score(y_true[:, i], y_scores[:, i])
        auc_scores.append(auc)
    mean_auc_score = np.mean(auc_scores)
    return mean_auc_score
    
def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for x_cnn, x_len_cnn, packed_rnn_input, labels in tqdm(train_loader):
            # x_cnn.shape     := (CONFIG.BATCH_SIZE, mini batch에서 길이가 가장 긴 image_sequence의 길이, 6, 224, 224)
            # x_len_cnn.shape := (CONFIG.BATCH_SIZE)
            x_cnn = x_cnn.float().to(device)
            packed_rnn_input = packed_rnn_input.float().to(device)
            labels = labels.float().to(device)
            
            probs = model(x_cnn, x_len_cnn, packed_rnn_input)
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

model.to(device)
print(f"Using device: {device}")

criterion = nn.BCELoss().to(device)
optimizer = optim.AdamW(model.parameters(), lr=CONFIG.LR)
scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=int(len(train_loader) * CONFIG.N_EPOCHS * 0.1),
    num_training_steps=(len(train_loader) * CONFIG.N_EPOCHS)
)

infer_model = train(model, optimizer, train_loader, val_loader, device)