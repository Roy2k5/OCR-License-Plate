import torch
from torch import nn
import torch.nn.functional as F
from data.ocr_dataset import OCRDataset


class MHA(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim phải chia hết cho num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.requant = torch.quantization.QuantStub()
        self.dropout = nn.Dropout(dropout)
        self.scaling = self.head_dim**-0.5

    def forward(self, x):
        tgt_len, bsz, embed_dim = x.size(1), x.size(0), x.size(2)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        if q.is_quantized:
            q = q.dequantize()
            k = k.dequantize()
            v = v.dequantize()

        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, embed_dim)
        )
        attn_output = self.requant(attn_output)
        return self.out_proj(attn_output)


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
        self.quant = torch.quantization.QuantStub()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.Dropout(0.2, inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.Dropout(0.2, True),
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.MaxPool2d((2, 1), (2, 1)),
        )
        self.mha = MHA(512, 8)
        self.linear = nn.Linear(512, vocabsize)
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.cnn(x).squeeze(2).permute(0, 2, 1)
        x = self.mha(x)
        x = F.dropout(F.relu(x), 0.1, self.training)
        x = self.linear(x)
        x = self.dequant(x)
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
