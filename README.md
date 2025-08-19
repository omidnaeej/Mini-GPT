# Mini-GPT Model

This repository contains an implementation of a simplified GPT (Generative Pre-trained Transformer) model as part of a deep learning assignment from the University of Tehran. The project focuses on training a language model using the "Friends" dataset, which consists of dialogues from the popular TV show.

---

## Project Overview

The goal of this project is to build and train a small-scale transformer-based language model to generate text similar to the dialogues in the "Friends" series. The implementation follows the transformer architecture, including components such as multi-head attention, feed-forward networks, and positional embeddings.

---
## Project Structure

- `data/`: Contains data-related files and the raw dataset.
- `models/`: Includes the model architecture definition and saved models.
- `scripts/`: Contains the main scripts for training and generating text.
- `config/`: Stores configuration files for hyperparameters.
- `utils/`: Utility functions for visualization and metrics.

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/omidnaeej/mini-gpt-model.git
   cd mini-gpt-model
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset:

   ```bash
   wget https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-09-08/friends.csv
   ```

## Usage

### Training the Model

To train the model, use the following command:

```bash
python scripts/train.py --config config/config.yaml
```

### Generating Text

To generate text using the trained model:

```bash
python scripts/generate_text.py --model_path models/saved_models/model.pth
```

## Project Structure

- `data/`: Contains data-related files and the raw dataset.
- `models/`: Includes the model architecture definition and saved models.
- `scripts/`: Contains the main scripts for training and generating text.
- `config/`: Stores configuration files for hyperparameters.
- `utils/`: Utility functions for visualization and metrics.

## Configuration

The model uses the following hyperparameters (configurable in `config/config.yaml`):

- `batch_size`: 16
- `block_size`: 32
- `max_iteration`: 5000
- `learning_rate`: 1e-3
- `n_embd`: 64
- `n_head`: 4

## Results

The trained model can generate coherent dialogues based on the "Friends" dataset. Sample outputs and detailed analysis are included in the project report.

## Contributing

Contributions are welcome. Please fork the repository and submit pull requests for any improvements.

## License

- [ ] This project is licensed under the MIT License. See the `LICENSE` file for details.
