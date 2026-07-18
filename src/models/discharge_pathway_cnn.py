"""
CareFlow AI - Module 3: Discharge Pathway Classification (1D CNN)
----------------------------------------------------------------------
Trains a 1D Convolutional Neural Network (PyTorch) to classify a
patient's discharge pathway (Home / Rehab / Nursing Facility /
Transfer / Home Health Care) based on their sequence of treatment
events during the admission.

Run: python src/models/discharge_pathway_cnn.py
Input:  data/raw/hospital_admissions.csv
Output: models_saved/discharge_pathway_cnn.pt
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import os

RANDOM_STATE = 42
DATA_PATH = "data/raw/hospital_admissions.csv"
MODEL_OUT_PATH = "models_saved/discharge_pathway_cnn.pt"
MAX_SEQ_LEN = 14          # covers longest generated sequence (12 events) with headroom
EMBED_DIM = 16
BATCH_SIZE = 64
EPOCHS = 25
LEARNING_RATE = 1e-3

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ---------------------------------------------------------------
# 1. Tokenization: turn event words into integer IDs
# ---------------------------------------------------------------
class SequenceVocab:
    """Simple vocabulary: maps event tokens <-> integer IDs.
    ID 0 is reserved for padding, ID 1 for unknown tokens."""

    def __init__(self):
        self.token_to_id = {"<PAD>": 0, "<UNK>": 1}

    def build(self, sequences):
        for seq in sequences:
            for token in seq.split():
                if token not in self.token_to_id:
                    self.token_to_id[token] = len(self.token_to_id)
        return self

    def encode(self, seq, max_len):
        ids = [self.token_to_id.get(tok, 1) for tok in seq.split()]
        ids = ids[:max_len]
        ids += [0] * (max_len - len(ids))   # pad with 0s
        return ids

    @property
    def vocab_size(self):
        return len(self.token_to_id)


# ---------------------------------------------------------------
# 2. PyTorch Dataset
# ---------------------------------------------------------------
class TreatmentSequenceDataset(Dataset):
    def __init__(self, sequences, labels, vocab, max_len):
        self.encoded = [vocab.encode(s, max_len) for s in sequences]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.encoded[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# ---------------------------------------------------------------
# 3. The 1D CNN model
# ---------------------------------------------------------------
class DischargePathwayCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, max_len):
        super().__init__()
        # Embedding: each token ID -> a learned dense vector
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Conv1d expects input shaped (batch, channels, length).
        # We treat embed_dim as "channels" and sequence position as "length".
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)   # global max pooling over the sequence
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)                  # (batch, seq_len, embed_dim)
        embedded = embedded.permute(0, 2, 1)           # (batch, embed_dim, seq_len) for Conv1d

        out = self.relu(self.conv1(embedded))          # (batch, 32, seq_len)
        out = self.relu(self.conv2(out))                # (batch, 64, seq_len)
        out = self.pool(out).squeeze(-1)                 # (batch, 64)

        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)                              # (batch, num_classes) - raw logits
        return out


# ---------------------------------------------------------------
# 4. Training loop
# ---------------------------------------------------------------
def train_model(model, train_loader, val_loader, epochs, lr, device, class_weights=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Class weights counteract imbalance so the model can't "cheat" by
    # always predicting the majority class (Home). Without this, a model
    # can look reasonably accurate while having zero recall on minority
    # classes - exactly what happened on the first run of this module.
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_batch.size(0)

        train_loss = total_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss, correct = 0, 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * x_batch.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == y_batch).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:2d}/{epochs} | train_loss={train_loss:.4f} "
                  f"| val_loss={val_loss:.4f} | val_acc={val_acc:.3f}")

    return history


def evaluate_model(model, val_loader, label_encoder, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    print("\n--- Classification Report ---")
    print(classification_report(
        all_labels, all_preds, target_names=label_encoder.classes_
    ))
    acc = accuracy_score(all_labels, all_preds)
    print(f"Overall accuracy: {acc:.3f}")
    return acc


def plot_training_history(history):
    os.makedirs("notebooks/figures", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["val_acc"], label="Val Accuracy", color="green")
    axes[1].set_title("Validation Accuracy over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("notebooks/figures/discharge_cnn_training_history.png")
    plt.close()
    print("Saved -> notebooks/figures/discharge_cnn_training_history.png")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} records")

    # Build vocabulary from all treatment sequences
    vocab = SequenceVocab().build(df["treatment_sequence"])
    print(f"Vocabulary size: {vocab.vocab_size} (event tokens + PAD + UNK)")

    # Encode labels (discharge pathways) as integers
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["discharge_pathway"])
    print(f"Classes: {list(label_encoder.classes_)}")

    X_train_seq, X_val_seq, y_train, y_val = train_test_split(
        df["treatment_sequence"], y, test_size=0.2,
        random_state=RANDOM_STATE, stratify=y
    )

    train_dataset = TreatmentSequenceDataset(
        X_train_seq.tolist(), y_train.tolist(), vocab, MAX_SEQ_LEN
    )
    val_dataset = TreatmentSequenceDataset(
        X_val_seq.tolist(), y_val.tolist(), vocab, MAX_SEQ_LEN
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Inverse-frequency class weights: rarer classes get a higher weight
    # so the loss penalizes ignoring them.
    class_counts = np.bincount(y_train, minlength=len(label_encoder.classes_))
    class_weights = torch.tensor(
        (class_counts.sum() / (len(class_counts) * class_counts)), dtype=torch.float32
    ).to(device)
    print(f"Class weights: {dict(zip(label_encoder.classes_, class_weights.cpu().numpy().round(2)))}")

    model = DischargePathwayCNN(
        vocab_size=vocab.vocab_size,
        embed_dim=EMBED_DIM,
        num_classes=len(label_encoder.classes_),
        max_len=MAX_SEQ_LEN
    ).to(device)

    print(f"\nModel architecture:\n{model}\n")

    history = train_model(model, train_loader, val_loader, EPOCHS, LEARNING_RATE, device, class_weights)
    evaluate_model(model, val_loader, label_encoder, device)
    plot_training_history(history)

    os.makedirs("models_saved", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab": vocab.token_to_id,
        "label_classes": label_encoder.classes_.tolist(),
        "max_seq_len": MAX_SEQ_LEN,
        "embed_dim": EMBED_DIM,
    }, MODEL_OUT_PATH)
    print(f"\nModel saved -> {MODEL_OUT_PATH}")


if __name__ == "__main__":
    main()
