import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import requests
import time

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --------------------
# 1. Data Preparation
# --------------------
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = requests.get(url)
text = response.text

# Create character vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {ch:i for i, ch in enumerate(chars)}
idx_to_char = {i:ch for i, ch in enumerate(chars)}

# Hyperparameters
SEQ_LENGTH = 100  # Context window size
BATCH_SIZE = 64
HIDDEN_SIZE = 256
EMBEDDING_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
LEARNING_RATE = 0.001
EPOCHS = 50

# Create training sequences
encoded_text = [char_to_idx[ch] for ch in text]
sequences = []
targets = []
for i in range(len(encoded_text) - SEQ_LENGTH):
    sequences.append(encoded_text[i:i+SEQ_LENGTH])
    targets.append(encoded_text[i+1:i+SEQ_LENGTH+1])

# Convert to tensors
X = torch.tensor(sequences, dtype=torch.long)
y = torch.tensor(targets, dtype=torch.long)

# Split dataset
train_size = int(0.9 * len(X))
train_dataset = torch.utils.data.TensorDataset(X[:train_size], y[:train_size])
val_dataset = torch.utils.data.TensorDataset(X[train_size:], y[train_size:])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# --------------------
# 2. Model Architecture
# --------------------
class CharLSTM(nn.Module):
    def __init__(self):
        super(CharLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)
        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS,
                           dropout=DROPOUT if NUM_LAYERS > 1 else 0,
                           batch_first=True)
        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(HIDDEN_SIZE, vocab_size)
        
    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        x = self.embedding(x)
        lstm_out, hidden = self.lstm(x, hidden)
        x = self.dropout(lstm_out)
        x = self.fc(x)
        return x, hidden

model = CharLSTM().to(device)
print(model)

# --------------------
# 3. Training Setup
# --------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

def train_epoch(model, loader):
    model.train()
    running_loss = 0.0
    for inputs, targets in tqdm(loader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        
        running_loss += loss.item()
    return running_loss / len(loader)

def validate(model, loader):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Validating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            running_loss += loss.item()
    return running_loss / len(loader)

# --------------------
# Training Execution
# --------------------
train_losses = []
val_losses = []
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    start_time = time.time()
    
    train_loss = train_epoch(model, train_loader)
    val_loss = validate(model, val_loader)
    
    scheduler.step(val_loss)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
    
    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Time: {time.time()-start_time:.2f}s")

# Plot training curves
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()