from __future__ import print_function
from torch import nn, optim, cuda, save
from torch.utils import data
from torchvision import datasets, transforms
import torch.nn.functional as F
import time
from torch import nn, optim, cuda, zeros
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
import datetime
import os

path = "/home/inhamath/inhamath/kaggle_FSWJ/jinik"
serial = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
writer = SummaryWriter(path+f"/backup/MNIST_pretrain_graph_{serial}")
weight_path = path+f"/backup/MNIST_pretrain_weight_{serial}"
os.makedirs(weight_path)

#MNIST dataset
train_dataset = datasets.MNIST(root=path+"/mnist_data/", 
                              train=True, 
                              transform=transforms.ToTensor(),
                              download=True)
test_dataset = datasets.MNIST(root=path+"/mnist_data/",
                             train=False,
                             transform=transforms.ToTensor())

# Data loader
batch_size = 64
train_loader = data.DataLoader(dataset=train_dataset,
                               batch_size=batch_size,
                               shuffle=True)
test_loader = data.DataLoader(dataset=test_dataset,
                              batch_size=batch_size,
                              shuffle=False)

device = "cuda" if cuda.is_available() else "cpu" # device가 cuda 면 행렬연산에서 cpu 보다 빠른연산을 수행할 수 있는 gpu(그래픽카드)를 사용한다. 그렇지 않으면 그냥 cpu 를 사용한다
print(f"Training Julia on '{device}'\n{'='*44}") # cpu 를 사용하는지 gpu(cuda는 nvidia)를 사용하는지 출력해줌

model = models.resnet18(pretrained=True)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
model.fc = nn.Linear(in_features=512, out_features=62)
model.to(device)

lossfunction = nn.CrossEntropyLoss()
optimiser = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.5) # stochastic gradient descent

def train(epoch):
    model.train() # nn.Module.train method, train 과 test 를 구분, test 할 때는 modle.eval 을 호출한다
    for batch_ind, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimiser.zero_grad() # gradients 값을 0으로 초기화
        output = model(data)
        loss = lossfunction(output, target)
        loss.backward()
        optimiser.step()
        if batch_ind % 100 == 0:
            print('Train Epoch : {} | Batch Status : {}/{} ({:.0f}%) | Loss : {:.6f}'.format(
                epoch, 
                batch_ind*len(data), len(train_loader.dataset), 100. * batch_ind / len(train_loader),
                loss.item()
                ))

def test(epoch):
    model.eval() # nn.Module.eval method, train 과 test 를 구분, train 할 때는 model.train 을 호출한다
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device) # gpu 에 tensor 들을 할당함(cpu가 아닌 gpu 로 계산하기 위함), 여기서 data, target 은 tensor 이다
        output = model(data)

        # sum up batch loss
        test_loss += lossfunction(output, target).item()

        # get the index of the max
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    writer.add_scalar("Loss", test_loss, epoch)
    writer.add_scalar("accuracy", correct/len(test_loader.dataset), epoch)
    print("="*44)
    print(f"Test set: Average loss : {test_loss:.4f}, Accuracy : {correct}/{len(test_loader.dataset)}({100. * correct / len(test_loader.dataset):.0f}%)")

print(f"save files under {path}")
epoch = 50
for epoch in range(1, epoch+1):
    train(epoch)
    save(model.state_dict(), weight_path+f"/{epoch} {batch_size}")
    test(epoch)