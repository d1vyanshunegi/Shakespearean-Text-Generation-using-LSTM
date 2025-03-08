import torch
import torch.nn as nn
import requests

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --------------------
# Data and Vocabulary Setup
# --------------------
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = requests.get(url)
text = response.text

chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# --------------------
# Model Hyperparameters (Must match training)
# --------------------
EMBEDDING_DIM = 128
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.3

# --------------------
# Model Definition
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
        x = self.embedding(x)
        lstm_out, hidden = self.lstm(x, hidden)
        x = self.dropout(lstm_out)
        x = self.fc(x)
        return x, hidden

# Instantiate the model and load the trained weights
model = CharLSTM().to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

# --------------------
# Text Generation Function
# --------------------
def generate_text(model, seed_str, length=500, temperature=0.8):
    model.eval()
    generated = []
    input_seq = torch.tensor([char_to_idx[c] for c in seed_str],
                             dtype=torch.long).unsqueeze(0).to(device)
    hidden = None
    for _ in range(length):
        with torch.no_grad():
            outputs, hidden = model(input_seq, hidden)
            logits = outputs[:, -1] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_char = torch.multinomial(probs, num_samples=1)
            generated.append(next_char.item())
            # Slide the window: remove the first character and append the new one
            input_seq = torch.cat([input_seq[:, 1:], next_char], dim=1)
    return seed_str + ''.join([idx_to_char[c] for c in generated])

# --------------------
# Main Execution
# --------------------
if __name__ == '__main__':
    seed_str = input("Enter a seed text: ")
    generated_text = generate_text(model, seed_str, length=500, temperature=0.8)
    print("\nGenerated Text:\n")
    print(generated_text)
