import torch
import os
import random
import numpy as np
import datetime
import json

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_serial():
    # 현재 시간을 가져와 형식에 맞게 변환
    now = datetime.datetime.now()
    serial_number = now.strftime("%Y%m%d%H%M%S")  # 연도, 월, 일, 시, 분, 초로 구성된 시리얼 번호
    return int(serial_number)


def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory {directory_path} created.")
    else:
        print(f"Directory {directory_path} already exists.")

def save_config(CONFIG):
    # Config 객체를 딕셔너리로 변환
    config_dict = {
        'SR': CONFIG.SR,
        'ROOT_META': CONFIG.ROOT_META,
        'WEIGHT_DIR': CONFIG.WEIGHT_DIR,
        'BATCH_SIZE': CONFIG.BATCH_SIZE,
        'N_EPOCHS': CONFIG.N_EPOCHS,
        'LR': CONFIG.LR,
        'SEED': CONFIG.SEED,
        'SERIAL': CONFIG.SERIAL,
        'USE_RATIO': CONFIG.USE_RATIO
    }

    # 딕셔너리를 JSON 파일로 저장
    with open(f"{CONFIG.WEIGHT_DIR}/{CONFIG.SERIAL}/config.json", 'w') as f:
        json.dump(config_dict, f, indent=4)