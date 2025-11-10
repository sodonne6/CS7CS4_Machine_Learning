import csv
import matplotlib.pyplot as plt


def load_csv_loss_data(filename):
    
    steps = []
    train_losses = []
    val_losses = []
    with open(filename, mode='r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            steps.append(int(row[0]))
            train_losses.append(float(row[1]))
            val_losses.append(float(row[2]))
    return steps, train_losses, val_losses

# plot data 
def plot_loss_per_step(steps, train_losses, val_losses, model_label):
    plt.plot(steps, train_losses, label='Train Loss')
    plt.plot(steps, val_losses, label='Validation Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss per Step - {model_label}')
    plt.legend()
    plt.grid()
    plt.show(
    )
    
#main to run functions
if __name__ == "__main__":
    
    # Load data from CSV files
    steps1, train_losses1, val_losses1 = load_csv_loss_data('gpt_train_losses_model3.csv')
    
    # Plot the loss data for both models
    plot_loss_per_step(steps1, train_losses1, val_losses1, model_label='Model 3')

