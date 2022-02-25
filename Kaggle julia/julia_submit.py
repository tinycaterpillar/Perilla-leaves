import glob
from pathlib import Path
import csv
import numpy as np
from PIL import Image
import string
from torch.utils import data
from torch import from_numpy 
from torch import nn, optim, cuda, zeros, load
from torchvision import models
from torch.utils import data
from pathlib import Path

path = "/home/inhamath/inhamath/kaggle_FSWJ/jinik"
    
ind = 0
meta_str2int = dict()
meta_int2str = [-1]*62
for i in string.ascii_letters:
    meta_str2int[i] = ind
    meta_int2str[ind] = i
    ind += 1

for i in "0123456789":
    meta_str2int[i] = ind
    meta_int2str[ind] = i
    ind += 1

device = "cuda" if cuda.is_available() else "cpu" # device가 cuda 면 행렬연산에서 cpu 보다 빠른연산을 수행할 수 있는 gpu(그래픽카드)를 사용한다. 그렇지 않으면 그냥 cpu 를 사용한다
print(f"Training Julia on '{device}'\n{'='*44}") # cpu 를 사용하는지 gpu(cuda는 nvidia)를 사용하는지 출력해줌

weight_path = "/home/inhamath/inhamath/kaggle_FSWJ/jinik/backup/weight_2022-02-25 12:47:15/49 8"
model = models.resnet18(pretrained=True)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
model.fc = nn.Linear(in_features=512, out_features=62)
model.load_state_dict(load(weight_path))
model.to(device)


lossfunction = nn.CrossEntropyLoss()
optimiser = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.5) # stochastic gradient descent

class CustomDataset():
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, ind):
        return self.images[ind], self.labels[ind]

image_path = sorted(glob.glob(path+'/testResized_32/*'), key=lambda path: int(Path(path).stem))
image_data = np.array([np.asarray(Image.open(path)).reshape((1, 28, 28)) for path in image_path])
image_data = from_numpy(image_data).float()
test_dataset = CustomDataset(image_data, image_path)

batch_size = 8
test_loader = data.DataLoader(
    dataset = test_dataset,
    batch_size = batch_size,
    shuffle = False
)

rows = [["ID", "Class"]]

model.eval() # nn.Module.eval method, train 과 test 를 구분, train 할 때는 model.train 을 호출한다
for data, target in test_loader:
    data = data.to(device)
    output = model(data)

    # get the index of the max
    pred = output.data.max(1, keepdim=True)[1]
    for id, label in zip([Path(path).stem for path in target], pred):
        rows.append([id, meta_int2str[label.item()]])

filename = path+"/julia.csv"

# writing to csv file
with open(filename, 'w') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)
     
    # writing the data rows
    csvwriter.writerows(rows)