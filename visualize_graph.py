import matplotlib.pyplot as plt


def visualize_graph(epochwise_train_acc,epochwise_val_acc,epochwise_train_loss,epochwise_val_loss,folder_name):
    fig,(axs1,axs2)=plt.subplots(1,2, figsize=(20,10))

    axs1.plot(epochwise_train_loss, label='Train Loss')
    axs1.plot(epochwise_val_loss, label='Validation Loss')
    axs1.set_title("Train vs val Loss")
    axs1.legend()

    axs2.plot(epochwise_train_acc, label='Train Accuracy')
    axs2.plot(epochwise_val_acc, label='Validation Accuracy')
    axs2.set_title("Train vs Val Accuracy")
    axs2.legend()

    plt.savefig(f"artifacts/{folder_name}/train_vs_val_graph.png")
    plt.show()
