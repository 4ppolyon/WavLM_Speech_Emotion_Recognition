import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print("Device:", device)

# -------------------------
# Dataset et DataLoader pour Transformer avec mask
# -------------------------
class PadDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


PAD_VALUE = 0.0  # padding constant

def collate_fn(batch, pad_value=PAD_VALUE):
    X_batch, y_batch = zip(*batch)
    X_batch = [x.float() for x in X_batch]  # ✅ s'assurer que float
    X_batch = pad_sequence(X_batch, batch_first=True, padding_value=pad_value)
    y_batch = torch.tensor(y_batch, dtype=torch.long)
    mask = (X_batch != pad_value).any(dim=2)  # ✅ comparer au pad_value plutôt qu'à 0
    return X_batch, y_batch, mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :] # heu...
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_outputs, d_model=256, n_layers=2, n_heads=8, dropout=0.1, ff_dim=512):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=2000)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # ✅ cls_token correctement défini et initialisé
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        self.layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_outputs)

    def name(self):
        return "TransformerClassifier"
    
    def forward(self, x, mask=None):
        b = x.size(0)
        x = self.input_proj(x)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_encoding(x)
        x = self.layernorm(x)

        if mask is not None:
            cls_mask = torch.ones((b, 1), dtype=torch.bool, device=x.device)
            mask = torch.cat((cls_mask, mask), dim=1)
            src_key_padding_mask = ~mask
            x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        else:
            x = self.encoder(x)

        cls_out = x[:, 0, :]
        cls_out = self.dropout(cls_out)
        out = self.fc(cls_out)
        return out

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, map_location=device):  # ✅ map_location pour GPU/CPU
        self.load_state_dict(torch.load(path, map_location=map_location))
    
    def evaluate_transformer(self, data_loader):
        self.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        criterion = nn.CrossEntropyLoss()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch, mask in data_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.long().to(device)
                mask = mask.to(device)

                logits = self(X_batch, mask)
                loss = criterion(logits, y_batch)

                preds = logits.argmax(dim=1)

                batch_size = y_batch.size(0)
                total_loss += loss.item() * batch_size
                correct += (preds == y_batch).sum().item()
                total += batch_size

                all_preds.append(preds.cpu())
                all_targets.append(y_batch.cpu())

        avg_loss = total_loss / total
        accuracy = correct / total

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        return avg_loss, accuracy, all_preds, all_targets

    
    def train_transformer(
        self,
        train_loader,
        val_loader,
        num_epochs=20,
        lr=1e-4,
        weight_decay=1e-5
    ):
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

            for X_batch, y_batch, mask in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.long().to(device)
                mask = mask.to(device)

                optimizer.zero_grad()

                logits = self(X_batch, mask)
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

            val_loss, val_acc, _, _ = self.evaluate_transformer(val_loader)
            scheduler.step(val_loss)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            print(
                f"[{self.name()}] Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f}"
            )

        return train_losses, val_losses, train_accs, val_accs
