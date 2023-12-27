import torch



def save_model_checkpoint(
    model,
    optimizer,
    folder_name,
    epoch,
    avg_train_loss,
    avg_val_loss,
    avg_train_acc,
    avg_val_acc,
    best_val_acc,
):
    best_model_path = None  # Initialize the variable

    # Save the model checkpoint
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "train_acc": avg_train_acc,
        "val_acc": avg_val_acc,
    }

    checkpoint_path = f"artifacts/{folder_name}/model_checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)

    # Save the best model if current validation accuracy is better than the previous best
    if avg_val_acc > best_val_acc:
        best_val_acc = avg_val_acc
        best_model_path = f"artifacts/{folder_name}/best_model.pth"
        torch.save(model.state_dict(), best_model_path)

    return best_val_acc, checkpoint_path, best_model_path

