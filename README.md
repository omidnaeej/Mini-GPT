# Mini-GPT Model

This repository contains an implementation of a simplified GPT (Generative Pre-trained Transformer) model as part of a deep learning assignment from the University of Tehran. The project focuses on training a language model using the "Friends" dataset, which consists of dialogues from the popular TV show.

---

## Project Overview

The goal of this project is to build and train a small-scale transformer-based language model to generate text similar to the dialogues in the "Friends" series. The implementation follows the transformer architecture, including components such as multi-head attention, feed-forward networks, and positional embeddings.

---

## Project Structure
- `config/`: YAML files for training configurations and logging  
- `data/`: Contains data-related files and the raw dataset.
- `models/`: Includes the model architecture definition and saved models.
- `notebooks/`: Jupyter notebook for exploration and experiments  
- `scripts/`: Contains the main scripts for training and generating text.  
- `utils/`: Metrics computation and visualization tools  
- `report.pdf`: Original report (in Persian)  

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

Run the main script to train and test both models:

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

The trained model can generate coherent dialogues based on the "Friends" dataset. Sample outputs and detailed analysis are included in the project report.

---

## Contributing

Contributions are welcome. Please fork the repository and submit pull requests for any improvements.
