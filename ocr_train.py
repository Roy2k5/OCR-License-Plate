import argparse
import os
from torchvision.transforms import Compose, Resize, ToTensor
import torch
from torch import nn
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import ImageDraw, Image
import numpy as np
from data.ocr_dataset import get_loader, OCRDataset
from models.ocr import CNNBasedOCRModel, inference_no_ctc
from utils import load_checkpoint, save_checkpoint

torch.manual_seed(42)


def get_args():
    parser = argparse.ArgumentParser(description="Train OCR task")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--continue-train", action="store_true")
    return parser.parse_args()


def train(args):
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    id2label = OCRDataset.prepare_label_dict()
    transform = Compose([Resize((32, 64)), ToTensor()])
    train_dataloader = get_loader(
        os.path.join("data", "train", "Scenario-B", "Brazilian"),
        transform,
        batch_size,
        True,
        4,
        True,
    )
    test_dataloader = get_loader(
        os.path.join("data", "train", "Scenario-A", "Brazilian"),
        transform,
        batch_size,
        False,
        4,
        False,
    )
    model = CNNBasedOCRModel(37)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=0.001)
    best_accuracy = 0
    start = 0
    model = model.to(device)
    if args.continue_train:
        best_accuracy, start = load_checkpoint(model, optimizer, device, "no_ctc_v2")

    writer = SummaryWriter("logs/ocr_no_ctc_v2", flush_secs=30)
    for epoch in range(start, epochs):
        progress_bar = tqdm(train_dataloader, colour="cyan", leave=False)
        losses = []
        model.train()
        for image, label in progress_bar:
            # input_lengths = torch.full((label.shape[0],), 16, dtype=torch.long).to(
            #     device
            # )
            image = image.to(device)
            label = label.to(device)
            # target_len = target_len.to(device)
            logits = model(image).permute(0, 2, 1)
            loss = criterion(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().item())
            progress_bar.set_description(f"Loss: {loss.item()}")
        mean_loss = np.sum(losses) / len(losses)
        writer.add_scalar("Loss/MSE", mean_loss, epoch)
        progress_bar = tqdm(test_dataloader, colour="cyan", leave=False)
        model.eval()
        with torch.inference_mode():
            for images, labels in progress_bar:
                images = images.to(device)
                logits = model(images)
                predicts = [inference_no_ctc(logit) for logit in logits]
                labels = [
                    "".join(id2label[item.item()] for item in label if item.item() != 0)
                    for label in labels
                ]
                accuracy = sum(
                    predict == label for predict, label in zip(predicts, labels)
                )
                writer.add_scalar("Val/Accuracy", accuracy, epoch)
                img_tensorboards = []
                for img, predict, label in zip(images, predicts, labels):
                    img = img.detach().cpu()
                    pil = to_pil_image(img)
                    canvas = Image.new("RGB", (64, 32 + 20), (255, 255, 255))
                    canvas.paste(pil, (0, 20))
                    draw = ImageDraw.Draw(canvas)
                    draw.text((5, 5), f"GT:{label}", fill=(0, 255, 0))
                    draw.text((5, 20), f"PR:{predict}", fill=(255, 0, 0))
                    tensor_img = to_tensor(
                        canvas
                    )  # đưa pil sang tensor và về [0, 1] dtype float
                    img_tensorboards.append(tensor_img)
                img_tensorboards = torch.stack(img_tensorboards)
                img_tensorboard = make_grid(img_tensorboards, nrow=5)
                writer.add_image("Val/Predict", img_tensorboard, epoch)
                break

        if best_accuracy <= accuracy:
            best_accuracy = accuracy
            save_checkpoint(
                model, optimizer, best_accuracy, epoch + 1, "no_ctc_v2", True
            )
        save_checkpoint(model, optimizer, best_accuracy, epoch + 1, "no_ctc_v2", False)
    writer.close()


if __name__ == "__main__":
    train(get_args())
