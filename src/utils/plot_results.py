import matplotlib.pyplot as plt

def plot_history(history, hyperparameters=None):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1, len(acc) + 1)
    
    plt.figure(figsize=(14, 7))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training Accuracy', linewidth=2)
    plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy', linewidth=2)
    plt.title('Training & Validation Accuracy', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss', linewidth=2)
    plt.title('Training & Validation Loss', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    if hyperparameters:
        info_text = (
            f"Hyperparameters:\n"
            f"Learning Rate: {hyperparameters.get('lr', 'Auto')}\n"
            f"Batch Size: {hyperparameters.get('batch_size', 64)}\n"
            f"Optimizer: Adam\n"
            f"Epochs: {len(epochs)}"
        )
        plt.figtext(0.5, 0.02, info_text, ha="center", fontsize=12, 
                    bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        plt.subplots_adjust(bottom=0.2)
    
    plt.show()