import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_loss(train_loss, val_loss, save_path: str = "loss_plot.png"):
    plt.figure(figsize=(12, 5))
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss,   label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


