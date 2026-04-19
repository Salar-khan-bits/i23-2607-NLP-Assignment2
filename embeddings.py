import numpy as np
import json
from utils import load_data, save_numpy, get_top_vocab, create_word_to_idx


def load_metadata(filepath):
    """Load metadata JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    print(f"Loaded metadata for {len(metadata)} documents")
    return metadata


# ============================================================================
# TF-IDF Implementation
# ============================================================================

def build_term_document_matrix(documents, vocab):
    """Build term-document matrix by counting word occurrences"""
    print("Building term-document matrix...")
    
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    term_doc_matrix = np.zeros((len(documents), len(vocab)))
    
    for doc_idx, doc in enumerate(documents):
        words = doc.split()
        for word in words:
            if word in word_to_idx:
                term_doc_matrix[doc_idx, word_to_idx[word]] += 1
    
    print("Done.")
    return term_doc_matrix


def compute_tfidf(term_doc_matrix):
    """Compute TF-IDF from term-document matrix"""
    print("Computing TF-IDF...")
    
    num_docs, vocab_size = term_doc_matrix.shape
    tfidf_matrix = np.zeros((num_docs, vocab_size))
    
    # Document frequency
    doc_freq = np.sum(term_doc_matrix > 0, axis=0)
    
    # IDF
    idf_values = np.log(num_docs / (1 + doc_freq))
    
    # TF-IDF
    for doc_idx in range(num_docs):
        word_counts = term_doc_matrix[doc_idx, :]
        doc_length = np.sum(word_counts)
        
        if doc_length > 0:
            tf_values = word_counts / doc_length
            tfidf_matrix[doc_idx, :] = tf_values * idf_values
    
    print("Done.")
    return tfidf_matrix


# ============================================================================
# PPMI Implementation
# ============================================================================

def build_cooccurrence_matrix(documents, vocab, window_size=5):
    """Build co-occurrence matrix"""
    print(f"Building co-occurrence matrix (window_size={window_size})...")
    
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    cooccurrence_matrix = np.zeros((len(vocab), len(vocab)))
    
    for doc_idx, doc in enumerate(documents):
        if (doc_idx + 1) % 1000 == 0:
            print(f"  Processing document {doc_idx + 1} / {len(documents)}...")
        
        words = doc.split()
        
        for i, word in enumerate(words):
            if word not in word_to_idx:
                continue
            
            word_idx = word_to_idx[word]
            start = max(0, i - window_size)
            end = min(len(words), i + window_size + 1)
            
            for j in range(start, end):
                if i != j and words[j] in word_to_idx:
                    context_idx = word_to_idx[words[j]]
                    cooccurrence_matrix[word_idx][context_idx] += 1
    
    print("Done.")
    return cooccurrence_matrix


def compute_ppmi(cooccurrence_matrix):
    """Compute PPMI from co-occurrence matrix"""
    print("Computing PPMI...")
    
    vocab_size = cooccurrence_matrix.shape[0]
    total_pairs = np.sum(cooccurrence_matrix)
    
    if total_pairs == 0:
        return np.zeros_like(cooccurrence_matrix)
    
    # Probabilities
    p_w1_w2 = cooccurrence_matrix / total_pairs
    p_w1 = np.sum(cooccurrence_matrix, axis=1) / total_pairs
    p_w2 = np.sum(cooccurrence_matrix, axis=0) / total_pairs
    
    # PPMI
    ppmi_matrix = np.zeros((vocab_size, vocab_size))
    
    for i in range(vocab_size):
        if (i + 1) % 1000 == 0:
            print(f"  Processing word {i + 1} / {vocab_size}...")
        
        for j in range(vocab_size):
            if cooccurrence_matrix[i][j] > 0 and p_w1[i] > 0 and p_w2[j] > 0:
                pmi = np.log2(p_w1_w2[i][j] / (p_w1[i] * p_w2[j]))
                ppmi_matrix[i][j] = max(0, pmi)
    
    print("Done.")
    return ppmi_matrix


# ============================================================================
# Word2Vec Skip-gram
# ============================================================================

class Word2VecSkipGram:
    """Simple Word2Vec Skip-gram with negative sampling"""
    
    def __init__(self, vocab_size, embedding_dim=100, window_size=5, negative_samples=5):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.negative_samples = negative_samples
        
        # Initialize embeddings
        self.V = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.U = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.learning_rate = 0.001  # Standard Word2Vec learning rate
        
        print(f"Initialized Word2Vec: vocab={vocab_size}, dim={embedding_dim}")
    
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def train(self, documents, word_to_idx, epochs=3):
        """Train Word2Vec model"""
        print("Building training pairs...")
        
        # Build pairs
        training_pairs = []
        for doc_idx, doc in enumerate(documents):
            if (doc_idx + 1) % 1000 == 0:
                print(f"  Processing document {doc_idx + 1} / {len(documents)}...")
            
            words = doc.split()
            for i, word in enumerate(words):
                if word not in word_to_idx:
                    continue
                
                center_idx = word_to_idx[word]
                start = max(0, i - self.window_size)
                end = min(len(words), i + self.window_size + 1)
                
                for j in range(start, end):
                    if j != i and j < len(words) and words[j] in word_to_idx:
                        context_idx = word_to_idx[words[j]]
                        training_pairs.append((center_idx, context_idx))
        
        print(f"Total training pairs: {len(training_pairs)}")
        
        # Check if we have training pairs
        if len(training_pairs) == 0:
            print("ERROR: No training pairs found!")
            print("This might happen if:")
            print("  1. Documents are empty")
            print("  2. Words in documents don't match vocabulary")
            print("  3. Window size is too small")
            
            # Debug: check first few documents
            print("\nDebug info:")
            print(f"  Number of documents: {len(documents)}")
            print(f"  First document words: {documents[0].split()[:10] if documents else 'No documents'}")
            print(f"  Vocabulary size: {len(word_to_idx)}")
            print(f"  First 5 vocab words: {list(word_to_idx.keys())[:5]}")
            return
        
        # Noise distribution
        freq = np.zeros(self.vocab_size)
        for center, _ in training_pairs:
            freq[center] += 1
        
        noise_dist = freq ** 0.75
        total = noise_dist.sum()
        
        if total > 0:
            noise_dist /= total
        else:
            # Fallback to uniform distribution
            noise_dist = np.ones(self.vocab_size) / self.vocab_size
        
        # Training
        print(f"Training for {epochs} epochs...")
        
        initial_lr = self.learning_rate
        
        for epoch in range(epochs):
            import random
            random.shuffle(training_pairs)
            
            epoch_loss = 0
            
            for pair_idx, (center_idx, context_idx) in enumerate(training_pairs):
                # Linear learning rate decay within epoch
                progress = (epoch * len(training_pairs) + pair_idx) / (epochs * len(training_pairs))
                current_lr = initial_lr * (1.0 - progress)  # Decay from initial_lr to 0
                current_lr = max(current_lr, initial_lr * 0.0001)  # Don't go below 0.0001 * initial
                
                # Positive pair
                v_center = self.V[center_idx]
                u_context = self.U[context_idx]
                
                dot_prod = np.dot(v_center, u_context)
                output = self.sigmoid(dot_prod)
                
                loss = -np.log(output + 1e-10)
                
                # Gradient update with current learning rate
                grad = (output - 1) * u_context
                self.V[center_idx] -= current_lr * grad
                self.U[context_idx] -= current_lr * (output - 1) * v_center
                
                # Negative samples
                neg_samples = np.random.choice(self.vocab_size, size=self.negative_samples, p=noise_dist)
                
                for neg_idx in neg_samples:
                    u_neg = self.U[neg_idx]
                    dot_neg = np.dot(v_center, u_neg)
                    out_neg = self.sigmoid(dot_neg)
                    
                    loss += -np.log(1 - out_neg + 1e-10)
                    
                    self.V[center_idx] -= current_lr * out_neg * u_neg
                    self.U[neg_idx] -= current_lr * out_neg * v_center
                
                epoch_loss += loss
                
                if (pair_idx + 1) % 50000 == 0:  # Print less frequently
                    print(f"  Epoch {epoch+1}, Pair {pair_idx+1}/{len(training_pairs)}, Loss: {epoch_loss/(pair_idx+1):.4f}, LR: {current_lr:.6f}")
            
            avg_loss = epoch_loss / len(training_pairs)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        print("Training complete!")
    
    def get_embeddings(self):
        return (self.V + self.U) / 2


# ============================================================================
# Evaluation
# ============================================================================

def get_nearest_neighbors(embeddings, word_idx, vocab, k=5):
    """Find k nearest neighbors"""
    word_vec = embeddings[word_idx]
    similarities = []
    
    for i in range(len(embeddings)):
        if i != word_idx:
            other_vec = embeddings[i]
            
            norm1 = np.linalg.norm(word_vec)
            norm2 = np.linalg.norm(other_vec)
            
            if norm1 > 0 and norm2 > 0:
                sim = np.dot(word_vec, other_vec) / (norm1 * norm2)
                similarities.append((i, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [(vocab[idx], sim) for idx, sim in similarities[:k]]


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Load data
    print("Loading data...")
    documents = load_data("data/cleaned.txt")
    metadata = load_metadata("data/metadata.json")
    vocab = get_top_vocab(documents, vocab_size=5000)  # Smaller vocab for speed
    word_to_idx, idx_to_word = create_word_to_idx(vocab)  # Unpack tuple
    
    print("\n" + "="*70)
    print("PART 1: TF-IDF")
    print("="*70)
    
    term_doc_matrix = build_term_document_matrix(documents, vocab)
    tfidf_matrix = compute_tfidf(term_doc_matrix)
    save_numpy(tfidf_matrix, "embeddings/tfidf_matrix.npy")
    print("TF-IDF matrix saved.")
    
    print("\n" + "="*70)
    print("PART 2: PPMI")
    print("="*70)
    
    cooccurrence = build_cooccurrence_matrix(documents, vocab, window_size=3)
    ppmi_matrix = compute_ppmi(cooccurrence)
    save_numpy(ppmi_matrix, "embeddings/ppmi_matrix.npy")
    print("PPMI matrix saved.")
    
    # Test PPMI neighbors
    print("\nSample PPMI neighbors:")
    test_word = vocab[0]
    neighbors = get_nearest_neighbors(ppmi_matrix, 0, vocab, k=5)
    print(f"{test_word}: {[w for w, s in neighbors]}")
    
    print("\n" + "="*70)
    print("PART 3: Word2Vec")
    print("="*70)
    
    w2v_model = Word2VecSkipGram(
        vocab_size=len(vocab),
        embedding_dim=50,  # Smaller for speed
        window_size=3,
        negative_samples=5
    )
    
    w2v_model.train(documents, word_to_idx, epochs=2)  # Fewer epochs for speed
    
    w2v_embeddings = w2v_model.get_embeddings()
    save_numpy(w2v_embeddings, "embeddings/embeddings_w2v.npy")
    print("Word2Vec embeddings saved.")
    
    # Test Word2Vec neighbors
    print("\nSample Word2Vec neighbors:")
    neighbors = get_nearest_neighbors(w2v_embeddings, 0, vocab, k=5)
    print(f"{test_word}: {[w for w, s in neighbors]}")
    
    print("\n" + "="*70)
    print("All embeddings computed successfully!")
    print("="*70)
