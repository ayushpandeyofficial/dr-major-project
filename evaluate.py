import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from src.dataloader import val_data_loader
from src.models.models_utils import parse_arguments, load_model, get_device

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    for test_images, test_labels in tqdm(test_loader):
        test_images, test_labels = test_images.to(device), test_labels.to(device)
        test_model_out = model(test_images)
        test_labels = test_labels.to(test_model_out.device)
        test_pred = torch.argmax(test_model_out, dim=1)

        all_preds.extend(test_pred.cpu().numpy())
        all_labels.extend(test_labels.cpu().numpy())

    return all_labels, all_preds


def calculate_metrics(all_labels, all_preds):
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds)
    return conf_matrix, class_report


def plot_confusion_matrix(conf_matrix, folder_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=["No_DR", "Mild", "Moderate", "Proliferate_DR", "Severe"],
        yticklabels=["No_DR", "Mild", "Moderate", "Proliferate_DR", "Severe"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(f"artifacts/{folder_name}/confusion_matrix.png")
    plt.close()


def save_classification_report(class_report, folder_name):
    report_path = f"artifacts/{folder_name}/classification_report.txt"
    with open(report_path, "w") as report_file:
        report_file.write("Classification Report:\n\n")
        report_file.write(class_report)


if __name__ == "__main__":
    args = parse_arguments(training=False)
    device = get_device()

    # Load model
    model = load_model(args.model, num_labels=5, device=device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict=checkpoint)

    # Evaluate the model
    all_labels, all_preds = evaluate_model(model, val_data_loader, device)

    # Calculate metrics
    conf_matrix, class_report = calculate_metrics(all_labels, all_preds)

    # Plot confusion matrix
    plot_confusion_matrix(conf_matrix, args.folder_name)

    # Save classification report
    save_classification_report(class_report, args.folder_name)
