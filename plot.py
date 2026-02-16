import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

def plot_regression_curve(model_name, train_losses, val_losses, train_rmses, val_rmses):
    epochs = range(1, len(train_losses) + 1)

    # --- Plot Loss Train ---
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, marker='o', label='Train Loss')
    plt.title(f"{model_name} Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.grid(True)
    plt.legend()

    # --- Plot RMSE ---
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_rmses, marker='o', color='orange', label='Train RMSE')
    plt.title(f"{model_name} Train RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.grid(True)
    plt.legend()
    
    # --- Plot Loss Val ---
    plt.subplot(2, 2, 3)
    plt.plot(epochs, val_losses, marker='o', label='Val Loss')
    plt.title(f"{model_name} Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Val Loss")
    plt.grid(True)
    plt.legend()

    # --- Plot RMSE Val ---
    plt.subplot(2, 2, 4)
    plt.plot(epochs, val_rmses, marker='o', color='orange', label='Val RMSE')
    plt.title(f"{model_name} Validation RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.grid(True)
    plt.legend()

    plt.tight_layout() # Adjust subplots to fit in figure area.
    plt.show()

    return train_losses, val_losses, train_rmses, val_rmses

def plot_classifier_curve(model_name, train_losses, val_losses, train_accs, val_accs):
    epochs = range(1, len(train_losses) + 1)

    # --- Plot Loss Train ---
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, marker='o', label='Train Loss')
    plt.title(f"{model_name} Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.grid(True)
    plt.legend()

    # --- Plot Accuracy Train ---
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_accs, marker='o', color='orange', label='Train Accuracy')
    plt.title(f"{model_name} Train Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    
    # --- Plot Loss Val ---
    plt.subplot(2, 2, 3)
    plt.plot(epochs, val_losses, marker='o', label='Val Loss')
    plt.title(f"{model_name} Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Val Loss")
    plt.grid(True)
    plt.legend()

    # --- Plot Accuracy Val ---
    plt.subplot(2, 2, 4)
    plt.plot(epochs, val_accs, marker='o', color='orange', label='Val Accuracy')
    plt.title(f"{model_name} Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()

    plt.tight_layout() # Adjust subplots to fit in figure area.
    plt.show()

    return train_losses, val_losses, train_accs, val_accs


def plot_classifier_matrix(model_name, conf_matrix, class_names):
    fig, ax = plt.subplots(figsize=(10, 8))  # agrandit la figure

    disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_matrix,
        display_labels=class_names
    )

    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)

    plt.title(f"{model_name} Confusion Matrix")

    # Rotation des labels X
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()  # évite que ça coupe les labels
    plt.show()
