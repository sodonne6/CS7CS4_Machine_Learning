import csv
import matplotlib.pyplot as plt
from pathlib import Path

def load_csv_loss_data(filename):
    steps, train_losses, val_losses, train_ppls, val_ppls = [], [], [], [], []
    with open(filename, newline="") as f:
        r = csv.reader(f)
        next(r)  # header
        for row in r:
            steps.append(int(row[0]))
            train_losses.append(float(row[1]))
            val_losses.append(float(row[2]))
            train_ppls.append(float(row[3]))
            val_ppls.append(float(row[4]))
    return {
        "steps": steps,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_ppls": train_ppls,
        "val_ppls": val_ppls,
    }

def plot_loss_compare(runs):
    """
    runs = [
      {"label": "Test A", **load_csv_loss_data(pathA)},
      {"label": "Test B", **load_csv_loss_data(pathB)},
    ]
    """
    plt.figure()
    for run in runs:
        s = run["steps"]
        plt.plot(s, run["train_losses"], label=f'{run["label"]} – Train')
        plt.plot(s, run["val_losses"],   label=f'{run["label"]} – Val')
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss per Step")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_perplexity_compare(runs):
    plt.figure()
    for run in runs:
        s = run["steps"]
        plt.plot(s, run["train_ppls"], label=f'{run["label"]} – Train PPL')
        plt.plot(s, run["val_ppls"],   label=f'{run["label"]} – Val PPL')
    plt.xlabel("Steps")
    plt.ylabel("Perplexity")
    plt.title("Training & Validation Perplexity per Step")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

#C:\Users\irish\Computer_Electronic_Engineering_Year5\Machine_Learning\Week_9\model_data\model_1\gpt_train_losses_model1_loss_ppl_preln=True_resAttn=False_resFfn=True.csv

if __name__ == "__main__":
    path_a = r"C:\Users\irish\Computer_Electronic_Engineering_Year5\Machine_Learning\Week_9\model_data\model_1\gpt_train_losses_model1_loss_ppl_preln=True_resAttn=True_resFfn=False.csv"
    path_b = r"C:\Users\irish\Computer_Electronic_Engineering_Year5\Machine_Learning\Week_9\model_data\model_1\gpt_train_losses_model1_loss_ppl_preln=True_resAttn=False_resFfn=True.csv"

    run_a = {"label": "Test A - FFN Disabled", **load_csv_loss_data(Path(path_a))}
    run_b = {"label": "Test B - Attention Disabled", **load_csv_loss_data(Path(path_b))}

    plot_loss_compare([run_a, run_b])
    plot_perplexity_compare([run_a, run_b])
