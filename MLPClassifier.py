import torch
import torch.nn as nn
from torch.optim import AdamW

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print("Device:", device)

# -------------------------
# 1) MLP Classifier
# -------------------------
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def name(self):
        return "MLPClassifier"

    def forward(self, x):
        assert x.dim() == 2, f"x doit être (batch, features), reçu {x.shape}"
        out = self.net(x)
        return out  # logits

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))
    
    def evaluate_MLP(self, loader):
        self.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        all_preds = []
        all_targets = []

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.long().to(device)

                logits = self(X_batch)
                loss = criterion(logits, y_batch)

                preds = logits.argmax(dim=1)

                total_loss += loss.item() * y_batch.size(0)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

                all_preds.append(preds.cpu())
                all_targets.append(y_batch.cpu())

        avg_loss = total_loss / total
        accuracy = correct / total

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        return avg_loss, accuracy, all_preds, all_targets

    
    def train_mlp(self, train_loader, val_loader, num_epochs=20, lr=1e-4, weight_decay=1e-5):
        self.to(device)

        optimizer = AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )

        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.long().to(device)

                optimizer.zero_grad()

                logits = self(X_batch)          # (batch, num_classes)
                loss = criterion(logits, y_batch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()

                batch_size = y_batch.size(0)
                running_loss += loss.item() * batch_size

                preds = logits.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += batch_size

            train_loss = running_loss / total
            train_acc = correct / total

            val_loss, val_acc, _, _ = self.evaluate_MLP(val_loader)

            scheduler.step(val_loss)

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            print(
                f"[{self.name()}] Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f}"
            )

        return train_losses, val_losses, train_accs, val_accs
