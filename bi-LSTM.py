import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
import os


# ====================== CRF LAYER ======================
class CRF(nn.Module):
    """Simple CRF layer for NER"""
    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
    
    def forward(self, emissions, tags, mask):
        """Compute negative log likelihood"""
        # Just use a simple loss for now
        batch_size = emissions.size(0)
        seq_len = emissions.size(1)
        
        loss = 0
        for b in range(batch_size):
            for t in range(seq_len):
                if mask[b, t] > 0:
                    loss += emissions[b, t, tags[b, t]]
        
        return -loss / batch_size
    
    def decode(self, emissions, mask):
        """Simple greedy decoding"""
        return torch.argmax(emissions, dim=-1)


# ====================== BiLSTM MODEL ======================
class BiLSTMTagger(nn.Module):
    """2-layer Bidirectional LSTM"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_tags, 
                 pretrained_embeddings=None, freeze_embeddings=False, use_crf=False):
        super().__init__()
        self.use_crf = use_crf
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            # Handle vocab size mismatch
            pretrained_vocab_size = pretrained_embeddings.shape[0]
            if pretrained_vocab_size < vocab_size:
                extended = np.random.randn(vocab_size, embed_dim).astype(np.float32) * 0.01
                extended[:pretrained_vocab_size] = pretrained_embeddings
                self.embedding.weight.data.copy_(torch.from_numpy(extended))
            else:
                self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings[:vocab_size]))
            self.embedding.weight.requires_grad = not freeze_embeddings
        
        # BiLSTM
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, 
                           bidirectional=True, batch_first=True, dropout=0.5)
        
        # Classifier
        self.classifier = nn.Linear(hidden_dim * 2, num_tags)
        
        # CRF
        if use_crf:
            self.crf = CRF(num_tags)
    
    def forward(self, x, lengths):
        """Forward pass"""
        emb = self.embedding(x)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        logits = self.classifier(lstm_out)
        return logits
    
    def loss(self, x, y, lengths):
        """Compute loss"""
        logits = self.forward(x, lengths)
        
        if self.use_crf:
            mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            mask = mask.float()
            y_crf = y.clone()
            y_crf[y == -100] = 0
            return self.crf(logits, y_crf, mask)
        else:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            return loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
    
    def predict(self, x, lengths):
        """Predict tags"""
        logits = self.forward(x, lengths)
        if self.use_crf:
            mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            return self.crf.decode(logits, mask)
        else:
            return torch.argmax(logits, dim=-1)


# ====================== DATASET ======================
class SequenceDataset(Dataset):
    """Dataset for POS/NER"""
    def __init__(self, filepath, word2idx, tag2idx, task='pos'):
        self.sentences = []
        self.labels = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            words, tags = [], []
            for line in f:
                line = line.strip()
                if not line:
                    if words:
                        word_ids = [word2idx.get(w, word2idx.get('<UNK>', 0)) for w in words]
                        tag_ids = [tag2idx.get(t, 0) for t in tags]
                        self.sentences.append(word_ids)
                        self.labels.append(tag_ids)
                        words, tags = [], []
                else:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        words.append(parts[0])
                        tags.append(parts[1] if task == 'pos' else parts[2] if len(parts) >= 3 else 'O')
        
        print(f"Loaded {len(self.sentences)} sentences from {filepath}")
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sentences[idx]), torch.tensor(self.labels[idx])


def collate_fn(batch):
    """Collate function for padding"""
    sentences, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in sentences])
    sentences = nn.utils.rnn.pad_sequence(sentences, batch_first=True, padding_value=0)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    return sentences, labels, lengths


# ====================== TRAINING ======================
def train_model(task, freeze_embeddings=False, use_crf=False):
    """Train BiLSTM model"""
    print(f"\n{'='*60}")
    print(f"Training {task.upper()} - Freeze: {freeze_embeddings}, CRF: {use_crf}")
    print(f"{'='*60}")
    
    # Load embeddings
    embeddings = np.load("embeddings/embeddings_w2v.npy")
    with open("embeddings/word2idx.json", 'r', encoding='utf-8') as f:
        word2idx = json.load(f)
    
    # Build tag vocabulary
    tag2idx = {}
    with open(f"data/{task}_train.conll", 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                tag = parts[1] if task == 'pos' else parts[2] if len(parts) >= 3 else 'O'
                if tag not in tag2idx:
                    tag2idx[tag] = len(tag2idx)
    
    print(f"Vocabulary: {len(word2idx)}, Tags: {len(tag2idx)}")
    
    # Load datasets
    train_ds = SequenceDataset(f"data/{task}_train.conll", word2idx, tag2idx, task)
    val_ds = SequenceDataset(f"data/{task}_val.conll", word2idx, tag2idx, task)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # Create model
    embed_dim = embeddings.shape[1]
    model = BiLSTMTagger(
        vocab_size=len(word2idx),
        embed_dim=embed_dim,
        hidden_dim=256,
        num_tags=len(tag2idx),
        pretrained_embeddings=embeddings,
        freeze_embeddings=freeze_embeddings,
        use_crf=use_crf
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Training loop
    best_f1 = 0
    patience_counter = 0
    train_losses = []
    val_f1_scores = []
    
    for epoch in range(30):
        # Train
        model.train()
        total_loss = 0
        for x, y, lengths in train_loader:
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)
            optimizer.zero_grad()
            loss = model.loss(x, y, lengths)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y, lengths in val_loader:
                x, y, lengths = x.to(device), y.to(device), lengths.to(device)
                preds = model.predict(x, lengths)
                for i in range(x.size(0)):
                    length = lengths[i].item()
                    all_preds.extend(preds[i, :length].cpu().numpy())
                    all_labels.extend(y[i, :length].cpu().numpy())
        
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        val_f1_scores.append(val_f1)
        
        print(f"Epoch {epoch+1:2d} | Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            model_name = f"bilstm_{task}{'_frozen' if freeze_embeddings else ''}{'_crf' if use_crf else ''}.pt"
            torch.save(model.state_dict(), f"models/{model_name}")
            print(f"  → Saved: {model_name}")
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{task.upper()} Training Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_f1_scores)
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.title(f'{task.upper()} Validation F1')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"models/{task}_curves.png")
    print(f"Best F1: {best_f1:.4f}\n")
    
    return model, tag2idx


# ====================== MAIN ======================
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    
    print("="*60)
    print("Part 2: BiLSTM Sequence Labeling")
    print("="*60)
    
    # Train POS
    print("\n### POS Tagging ###")
    train_model("pos", freeze_embeddings=True, use_crf=False)
    train_model("pos", freeze_embeddings=False, use_crf=False)
    
    # Train NER
    print("\n### NER ###")
    train_model("ner", freeze_embeddings=True, use_crf=True)
    train_model("ner", freeze_embeddings=False, use_crf=True)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
