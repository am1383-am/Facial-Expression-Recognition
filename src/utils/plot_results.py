import matplotlib.pyplot as plt
import numpy as np

def plot_history(history, hyperparameters=None):
    if hasattr(history, 'history'):
        metrics = history.history
    else:
        metrics = history

    acc = metrics.get('accuracy', [])
    val_acc = metrics.get('val_accuracy', [])
    loss = metrics.get('loss', [])
    val_loss = metrics.get('val_loss', [])
    
    if not acc:
        print("No training data found.")
        return

    epochs = range(1, len(acc) + 1)
    
    plt.figure(figsize=(14, 7))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training Acc', linewidth=2)
    if val_acc:
        plt.plot(epochs, val_acc, 'ro-', label='Validation Acc', linewidth=2)
    plt.title('Training & Validation Accuracy', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training Loss', linewidth=2)
    if val_loss:
        plt.plot(epochs, val_loss, 'ro-', label='Validation Loss', linewidth=2)
    plt.title('Training & Validation Loss', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    if hyperparameters:
        info_text = (
            f"Hyperparameters:\n"
            f"LR: {hyperparameters.get('lr', 'Auto')} | "
            f"Batch: {hyperparameters.get('batch_size', 64)} | "
            f"Epochs: {len(epochs)}"
        )
        plt.figtext(0.5, 0.01, info_text, ha="center", fontsize=11, 
                    bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        plt.subplots_adjust(bottom=0.15)
    
    plt.show()

    print("\n" + "="*40)
    print("          BEST PERFORMANCE          ")
    print("="*40)
    
    best_acc = np.max(acc)
    best_acc_epoch = np.argmax(acc) + 1
    best_loss = np.min(loss)
    best_loss_epoch = np.argmin(loss) + 1
    
    print(f"TRAINING:")
    print(f"  • Best Accuracy: {best_acc:.4f} (Epoch {best_acc_epoch})")
    print(f"  • Lowest Loss:   {best_loss:.4f} (Epoch {best_loss_epoch})")
    
    if val_acc and val_loss:
        best_val_acc = np.max(val_acc)
        best_val_acc_epoch = np.argmax(val_acc) + 1
        best_val_loss = np.min(val_loss)
        best_val_loss_epoch = np.argmin(val_loss) + 1
        
        print("-" * 40)
        print(f"VALIDATION:")
        print(f"  • Best Accuracy: {best_val_acc:.4f} (Epoch {best_val_acc_epoch})")
        print(f"  • Lowest Loss:   {best_val_loss:.4f} (Epoch {best_val_loss_epoch})")
    
    print("="*40 + "\n")