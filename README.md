# CS-4063 Assignment 2: BBC Urdu NLP Pipeline

Implementation of TF-IDF, PPMI, Word2Vec, BiLSTM, and Transformer models from scratch for BBC Urdu text processing.

## Overview

This project implements three core NLP components:

- Word embedding techniques (TF-IDF, PPMI, Word2Vec)
- Sequence labeling with BiLSTM
- Text classification using Transformer architecture



## Part 1: Embeddings

Implementation of three embedding approaches for Urdu text representation.

### How to run

```bash
python embeddings.py
```

## Part 2: Sequence Labeling

BiLSTM model for sequence labeling tasks on BBC Urdu dataset.

### How to run

```bash
python bi-LSTM.py
```

## Part 3: Transformer Classification

Transformer-based text classification for BBC Urdu articles.

### How to run

```bash
python transformer.py
```

## Project Structure

```
.
├── data/           # Dataset files
├── embeddings/     # Saved embedding models
├── models/         # Trained model checkpoints
├── notebooks/      # Jupyter notebooks for analysis
├── Part1_embeddings.py
├── Part2_bilstm.py
├── Part3_transformer.py
└── utils.py        # Helper functions
```

## Requirements

- Python 3.8+
- NumPy
- Pandas
- Matplotlib

## Author

CS-4063 NLP Course Assignment
