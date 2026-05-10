import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import classification_report
import numpy as np

from model import TrafficClassifier
from dataset import load_and_preprocess, CICIDSDataset

CLASSES = ["Benign", "DDoS", "PortScan", "BruteForce", "WebAttack"]


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    X, y, _ = load_and_preprocess(args.data_dir)
    dataset = CICIDSDataset(X, y)
    n_train = int(0.8 * len(dataset))
    train_ds, val_ds = random_split(dataset, [n_train, len(dataset) - n_train])

    # Pin memory + persistent workers for fast GPU transfer
    loader_kwargs = dict(batch_size=args.batch_size, num_workers=4,
                         pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader   = DataLoader(val_ds,  shuffle=False, **loader_kwargs)

    model = TrafficClassifier(input_dim=X.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()  # Mixed precision

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(y_batch)
            correct += (logits.argmax(1) == y_batch).sum().item()
            total += len(y_batch)
        scheduler.step()

        # ── Validate ──
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad(), autocast():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device, non_blocking=True), y_b.to(device, non_blocking=True)
                preds = model(X_b).argmax(1)
                val_correct += (preds == y_b).sum().item()
                val_total += len(y_b)

        val_acc = val_correct / val_total
        print(f"Epoch {epoch:3d}/{args.epochs} | loss={total_loss/total:.4f} | "
              f"train_acc={correct/total:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/best_model.pt")
            print(f"  ✓ Saved best model (val_acc={val_acc:.4f})")

    print(f"
Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    train(parser.parse_args())
