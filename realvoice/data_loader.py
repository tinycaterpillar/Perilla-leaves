import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence, pad_sequence

class CustomDataset(Dataset):
    def __init__(self, x: list, y=None, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        cnn_feature, rnn_feature = self.x[idx]
        cnn_feature = torch.stack([self.transform(tmp) for tmp in cnn_feature])
        rnn_feature = torch.from_numpy(rnn_feature)
        
        if self.y is not None:
            return cnn_feature, rnn_feature, torch.from_numpy(self.y[idx])
        return cnn_feature, rnn_feature

def collate_fn(batch):
    cnn_features, rnn_features, labels = zip(*batch)
    
    lengths = torch.tensor([len(rnn) for rnn in rnn_features])
    sorted_indices = lengths.argsort(descending=True)
    
    sorted_cnn_features = [cnn_features[i] for i in sorted_indices]
    sorted_rnn_features = [rnn_features[i] for i in sorted_indices]
    sorted_labels = torch.stack([labels[i] for i in sorted_indices])

    x_cnn = pad_sequence([torch.tensor(f) for f in sorted_cnn_features], batch_first=True)
    packed_rnn_input = pack_sequence(sorted_rnn_features, enforce_sorted=False)
    x_len_cnn = torch.tensor([len(cnn) for cnn in sorted_cnn_features])

    return x_cnn, x_len_cnn, packed_rnn_input, sorted_labels

def get_dataloader(dataset: list, labels: list, batch_size: int, shuffle: bool):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225]),
    ])

    dataset = CustomDataset(dataset, labels, transform=transform)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle
    )

    return data_loader