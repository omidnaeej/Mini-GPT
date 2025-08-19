# Mini-GPT Model

This repository contains an implementation of a simplified GPT (Generative Pre-trained Transformer) model as part of a deep learning assignment from the University of Tehran. The project focuses on training a language model using the "Friends" dataset, which consists of dialogues from the popular TV show.

---

## Repository Contents
- `config/`: Configuration files for training and model parameters  
- `data/`: Data preprocessing and text loading scripts
- `models/`: Transformer-based model definition and saved models.
- `notebooks/`: Jupyter notebook for model testing and experiments  
- `scripts/`: Contains the main scripts for training and generating text.  
- `utils/`: Metrics computation and visualization tools  
- `report.pdf`: Original report (in Persian)

---

## Project Overview

#### 1. **Data Preparation**
- Used a custom dialogue dataset from the **Friends TV series**.
- Preprocessing includes:
  - **Character-level tokenization** was used due to computational costs and limited resources, which simplifies the vocabulary and allows for easier training.
  - Creating a vocabulary.
  - Encoding text into numerical sequences.

#### 2. **Model Architecture**
- A **simplified GPT model** based on the original **Transformer architecture** from the "Attention Is All You Need" paper.
- The model primarily consists of:
  - **Multi-head self-attention mechanisms** to capture relationships between words in a sequence.
  - **Position-wise feed-forward networks**.
- An **embedding layer** and **positional encodings** are used to represent input tokens and their order.

#### 3. **Training**
- **Loss**: Typically a form of **Cross-Entropy** loss.
- **Optimizer**: **Adam** or a similar adaptive optimizer. To be specific, I used the **AdamW** optimizer which is recommended for training transformer models.
- The model is trained to predict the next token in a sequence, given the preceding tokens.

#### 4. **Functionality**
- The model can be used to generate new dialogue in the style of the Friends TV series.
- It provides functionality for loading a pre-trained model and generating text based on a starting prompt.

---

## Setup
Clone the repository:
```bash
git clone https://github.com/omidnaeej/Mini-GPT.git
cd Mini-GPT
```

## Download the dataset

```bash
wget https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-09-08/friends.csv
```

## Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the main script to train and test model:

```bash
python -m scripts.main
```

---

## Configuration

The model uses the following hyperparameters (configurable in `config/config.yaml`):

- `batch_size`: 16
- `block_size`: 32
- `max_iteration`: 5000
- `learning_rate`: 1e-3
- `n_embd`: 64
- `n_head`: 4
- `n_layers`: 10
- `beam_search`: True
- `beam_width`: 3
- `temperature`: 1.0
- `max_length`: 200

---

## Results

- The model successfully learns to mimic the style and patterns of the Friends dialogue dataset.

- The transformer architecture with multi-head attention proves effective for text generation, producing contextually relevant text. While not perfect, the model's ability to generate coherent dialogue can be enhanced significantly by increasing the number of transformer blocks and employing practical methods like subword-level tokenization.

---

## Contributing

Contributions are welcome. Please fork the repository and submit pull requests for any improvements.
