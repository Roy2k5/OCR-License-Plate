import argparse
import os
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torch
from torch import nn
from tqdm.autonotebook import tqdm
from PIL import ImageDraw, Image
import numpy as np
from data.ocr_dataset import get_loader, OCRDataset
from models.ocr import CNNBasedOCRModel, inference_no_ctc
from utils import load_checkpoint, count_quantized_params
import matplotlib.pyplot as plt

torch.manual_seed(42)


def get_args():
    parser = argparse.ArgumentParser(description="Test OCR task")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def test(args):
    batch_size = args.batch_size
    device = "cpu"
    id2label = OCRDataset.prepare_label_dict()
    transform = Compose([Resize((32, 64)), ToTensor()])
    test_dataloader = get_loader(
        os.path.join("data", "train", "Scenario-A", "Brazilian"),
        transform,
        batch_size,
        False,
        4,
        True,
    )
    model = CNNBasedOCRModel(37).to(device)
    model.eval()
    torch.quantization.fuse_modules(
        model.cnn,
        modules_to_fuse=[
            ["0", "1", "2"],  # Conv + BN + ReLU
            ["5", "6", "7"],
            ["9", "10", "11"],
            ["14", "15", "16"],
        ],
        inplace=True,
    )
    model.train()
    model.qconfig = torch.quantization.default_qat_qconfig
    torch.quantization.prepare_qat(model, inplace=True)
    load_checkpoint(model, None, device, "NoCTC_quantize", True)
    model.eval()
    cpu_model = model.to("cpu")
    torch.quantization.convert(cpu_model, inplace=True)
    print(f"Kích thước mô hình: {count_quantized_params(model)}")
    progress_bar = tqdm(test_dataloader, colour="cyan", leave=True)
    cpu_model.eval()
    total_accuracy = 0
    num = 0
    accuracy_list = []
    with torch.inference_mode():
        for images, labels in progress_bar:
            images = images.to(device)
            logits = cpu_model(images)
            predicts = [inference_no_ctc(logit) for logit in logits]
            labels = [
                "".join(id2label[item.item()] for item in label if item.item() != 0)
                for label in labels
            ]
            accuracy = sum(predict == label for predict, label in zip(predicts, labels))
            total_accuracy += accuracy
            num += len(labels)
            accuracy_list.append(accuracy / len(labels))
        with open("result_non_ctc_quan.txt", "w") as file:
            file.write(f"Accuracy: {total_accuracy/num}")
        plt.plot(range(1, len(progress_bar) + 1), accuracy_list)
        plt.savefig("accuracy_non_ctc_quan.png")


if __name__ == "__main__":
    test(get_args())
