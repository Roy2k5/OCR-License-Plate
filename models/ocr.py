import torch
from torch import nn
import torch.nn.functional as F
from data.ocr_dataset import OCRDataset


class RCNNBasedOCRModel(nn.Module):
    def __init__(self, vocal_size):
        super().__init__()
        # 32, 64
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding="same", bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.Conv2d(32, 64, 3, padding="same", bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.Conv2d(64, 128, 3, padding="same", bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(128, 256, 3, padding="same", bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, padding="same"),
            nn.BatchNorm2d(512),
            nn.MaxPool2d((2, 1), (2, 1)),
        )
        self.rnn = nn.LSTM(512, 256, 2, bidirectional=True)
        self.linear = nn.Linear(512, vocal_size)

    def forward(self, x):
        x = self.cnn(x).squeeze(2).permute(2, 0, 1)
        x, _ = self.rnn(x)
        x = self.linear(x)  # (16, B, vocab_size)
        return F.log_softmax(x, dim=-1)


class CNNBasedOCRModel(nn.Module):
    def __init__(self, vocabsize):
        super().__init__()
        # 32, 64
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding="same", bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.Dropout(0.2, inplace=True),
            nn.Conv2d(32, 64, 3, padding="same", bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.Conv2d(64, 128, 3, padding="same", bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.Dropout(0.2, True),
            nn.Conv2d(128, 256, 3, padding="same", bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, padding="same"),
            nn.BatchNorm2d(512),
            nn.MaxPool2d((2, 1), (2, 1)),
        )
        self.mha = nn.MultiheadAttention(512, 8, batch_first=True)
        self.linear = nn.Linear(512, vocabsize)

    def forward(self, x):
        x = self.cnn(x).squeeze().permute(0, 2, 1)
        x, _ = self.mha(x, x, x, need_weights=False)
        x = F.dropout(F.relu(x), 0.1, self.training)
        x = self.linear(x)
        return x


def inference(input):
    """input (T, C): T là timestep, C: số class"""
    sequence = input
    if isinstance(input, torch.Tensor):
        sequence = torch.argmax(input, dim=-1).tolist()
    id2label = OCRDataset.prepare_label_dict()
    label_sequence = [id2label[item] for item in sequence]
    result = []
    prev = None
    for item in label_sequence:
        if item != prev and item != "-":
            result.append(item)
        prev = item
    return "".join(result)


def inference_no_ctc(input):
    """input (T, C): T là timestep, C: số class"""
    sequence = input
    if isinstance(input, torch.Tensor):
        sequence = torch.argmax(input, dim=-1).tolist()
    id2label = OCRDataset.prepare_label_dict()
    label_sequence = [id2label[item] for item in sequence]
    result = list(filter(lambda x: x != "-", label_sequence))
    return "".join(result)


if __name__ == "__main__":
    print(inference([0, 1, 2, 3, 0, 1, 1, 0, 2, 2, 0]))
