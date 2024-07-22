from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch import nn
import torchvision.models as models
import torch

class CRNN(nn.Module):
    def __init__(self, cnn_channel_num):
        super(CRNN, self).__init__()
        self.cnn = models.resnet18(pretrained=True)

        # 새로운 컨볼루션 레이어 생성 (채널 수만 변경)
        self.cnn.conv1 = nn.Conv2d(cnn_channel_num, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 마지막 fully connected 층 제거
        self.cnn.fc = nn.Identity()  
    
        self.rnn = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)
    
    def forward(self, x_cnn, x_len_cnn):
        # x의 크기: (batch_size, seq_len, c, h, w)
        batch_size, seq_len, c, h, w = x_cnn.size()

        # CNN을 적용하기 위해 크기를 변환: (batch_size * seq_len, c, h, w)
        c_in = x_cnn.view(batch_size * seq_len, c, h, w)

        # CNN 적용: (batch_size * seq_len, 512, 1, 1)
        c_out = self.cnn(c_in)

        # RNN의 입력으로 사용하기 위해 크기를 변환: (batch_size, seq_len, 512*7*7)
        r_in = c_out.view(batch_size, seq_len, -1)

        # 시퀀스 길이에 따라 패킹
        x_len_cnn = x_len_cnn.cpu().long()
        packed_input = pack_padded_sequence(r_in, x_len_cnn, batch_first=True, enforce_sorted=False)

        # LSTM 적용
        packed_output, (h_n, c_n) = self.rnn(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        output = output[range(output.shape[0]), x_len_cnn - 1, :]
        
        return output

    
class RNN(nn.Module):
    def __init__(self, rnn_feature_num):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(input_size=rnn_feature_num, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)
        
    def forward(self, x_rnn, x_len_rnn):
        x_len_rnn = x_len_rnn.cpu().long()
        packed_rnn_input = pack_padded_sequence(x_rnn, x_len_rnn, batch_first=True)
        packed_output, (h_n, c_n) = self.rnn(packed_rnn_input)
        output, lengths = pad_packed_sequence(packed_output, batch_first=True)
        output = output[range(output.shape[0]), lengths - 1, :]

        return output
    
class AudioClassifier(nn.Module):
    def __init__(self, cnn_channel_num, rnn_feature_num):
        super(AudioClassifier, self).__init__()
        self.crnn = CRNN(cnn_channel_num)
        self.rnn = RNN(rnn_feature_num)
        self.fc_combine = nn.Linear(512 + 512, 512)  # 512(crnn) + 512(rnn)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2),
            nn.Sigmoid()
        )

    def forward(self, x_cnn, x_len_cnn, x_rnn, x_len_rnn):
        # CRNN을 통한 이미지 처리
        crnn_features = self.crnn(x_cnn, x_len_cnn)  # (batch_size, 512)

        # LSTM을 통한 시퀀셜 데이터 처리
        rnn_features = self.rnn(x_rnn, x_len_rnn)  # (batch_size, 512)

        # CNN과 LSTM 특징 결합
        combined_features = torch.cat((crnn_features, rnn_features), dim=1)  # (batch_size, 1024)
        combined_features = self.fc_combine(combined_features)  # (batch_size, 512)
        combined_features = torch.relu(combined_features)
        
        # 최종 분류
        output = self.classifier(combined_features)  
              
        return output