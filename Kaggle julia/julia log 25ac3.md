# julia log

# 결과

![해냈다!!.png](julia%20log%2025ac3/%ED%95%B4%EB%83%88%EB%8B%A4!!.png)

![요정도에 있을듯.png](julia%20log%2025ac3/%EC%9A%94%EC%A0%95%EB%8F%84%EC%97%90_%EC%9E%88%EC%9D%84%EB%93%AF.png)

리더보드 17 등 정도의 성능이 나왔다

# directory 구조

```
Project_Julia
│   julia_main.py
│   julia_submit.py
│   julia_resize_data.py
│   trainLabels.csv
│
└───test
│   │   1.bmp
│   │   2.bmp
│   │   ...   
│   
└───train
│   │   6284.bmp
│   │   6285.bmp
│   │   ...
│   
└───trainResize_32
│   │   1.bmp
│   │   2.bmp
│   │   ...   
│   
└───testResize_32
│   │   6284.bmp
│   │   6285.bmp
│   │   ...
│
└───backup*
    └───graph_2022-02-25 00:17:08
    │       events.out.blabla  
    │
    └───weight_2022-02-25 00:17:08
        │   1 8
        │   2 8
        │   ...
```

# 설명

## julia_main.py

torchvision.models 의 resnet18 을 약간 변형해서(finetuning) 사용했다

```python
from torchvision import models
!pip3 install pytorch-model-summary
import pytorch_model_summary
import torch

model = models.resnet18(pretrained=True)
model.to(device)
print(pytorch_model_summary.summary(model, torch.zeros(1, 3, 28, 28).to(device), show_input=True))
```

```
----------------------------------------------------------------------------
Layer (type)         Input Shape         Param #     Tr. Param #
============================================================================
               Conv2d-1      [1, 3, 28, 28]           9,408           9,408
          BatchNorm2d-2     [1, 64, 14, 14]             128             128
                 ReLU-3     [1, 64, 14, 14]               0               0
            MaxPool2d-4     [1, 64, 14, 14]               0               0
           BasicBlock-5       [1, 64, 7, 7]          73,984          73,984
           BasicBlock-6       [1, 64, 7, 7]          73,984          73,984
           BasicBlock-7       [1, 64, 7, 7]         230,144         230,144
           BasicBlock-8      [1, 128, 4, 4]         295,424         295,424
           BasicBlock-9      [1, 128, 4, 4]         919,040         919,040
          BasicBlock-10      [1, 256, 2, 2]       1,180,672       1,180,672
          BasicBlock-11      [1, 256, 2, 2]       3,673,088       3,673,088
          BasicBlock-12      [1, 512, 1, 1]       4,720,640       4,720,640
   AdaptiveAvgPool2d-13      [1, 512, 1, 1]               0               0
              Linear-14            [1, 512]         513,000         513,000
============================================================================
Total params: 11,689,512
Trainable params: 11,689,512
Non-trainable params: 0
```

원래 이렇게 생겼는데 다음 코드를 추가해서 처음과 끝의 layer 만 바꿨다
처음에 있는 layer 인 conv1 의 kernel size 는 (5, 5) 에서 (3, 3) 으로 바꿨다
training data 의 사이즈가 (28, 28) 로 작은편이라서 큰 filter 를 사용할 필요가 없어보였다
그에따라 stride 는 1, padding 은 (1, 1) 로 맞춰주었다.

```python
model.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
model.fc = nn.Linear(in_features=512, out_features=62)
```

```
----------------------------------------------------------------------------
           Layer (type)         Input Shape         Param #     Tr. Param #
============================================================================
               Conv2d-1      [1, 1, 28, 28]             576             576
          BatchNorm2d-2     [1, 64, 28, 28]             128             128
                 ReLU-3     [1, 64, 28, 28]               0               0
            MaxPool2d-4     [1, 64, 28, 28]               0               0
           BasicBlock-5     [1, 64, 14, 14]          73,984          73,984
           BasicBlock-6     [1, 64, 14, 14]          73,984          73,984
           BasicBlock-7     [1, 64, 14, 14]         230,144         230,144
           BasicBlock-8      [1, 128, 7, 7]         295,424         295,424
           BasicBlock-9      [1, 128, 7, 7]         919,040         919,040
          BasicBlock-10      [1, 256, 4, 4]       1,180,672       1,180,672
          BasicBlock-11      [1, 256, 4, 4]       3,673,088       3,673,088
          BasicBlock-12      [1, 512, 2, 2]       4,720,640       4,720,640
   AdaptiveAvgPool2d-13      [1, 512, 2, 2]               0               0
              Linear-14            [1, 512]          31,806          31,806
============================================================================
Total params: 11,199,486
Trainable params: 11,199,486
Non-trainable params: 0
```

train data에서 train, validation 의 비율은 4 : 1 이고
batch size 는 8 로 설정했다.
a-zA-Z0-9 까지의 labels 를 0~62 까지 integer 에 매칭하여 meta_str2int, meta_int2str 에 저장하였다

Tmi

처음에는 EfficientNetb0 를 사용했었는데 학습이 되지 않았다
EfficientNetb0 도 위와 같이 구조를 출력해봤더니 input Shape 이 너무 줄어드는것이 보였다
그래서 torchvision.models 에 있는 모든 모델중에서 가장 layer 이 적은 resnet18 으로 골랐다.

네트워크를 직접 코딩해보고 싶었지만 아직은 모델을 만들기에 지식이 너무 부족하고 
pretrain data 를 사용할 수 있는 장점도 있어 잘하는 사람들이 만들어놓은 코드(torchvision)를 그냥 사용하기로 했다.

왜그런지는 모르겠으나 label 을 string 형태로 그냥 사용했더니 오류가 나왔다
그래서 a-zA-Z0-9 까지의 labels 를 0~62 까지 integer 에 매칭하여 meta_str2int, meta_int2str 에 저장하였다(이 부분은 허건혁 선배님께서 도와주셨다)

# julia_resize_data.py

kaggle 에서 준 trainResize 는 (20, 20, 3), (20, 20) 데이터가 섞여있었다
성능이 좋은 MNIST 모델을 그대로 사용하려는 의도로 모든 데이터를 (28, 28) 로 변형했다