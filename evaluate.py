from argparse import ArgumentParser

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from src.dataloader import test_data_loader
from src.models.resnet50 import ResNet50Model

if __name__=="__main__":
    
    parser=ArgumentParser()
    parser.add_argument("-c","--checkpoint",type=str, required=True)
    parser.add_argument("-f","--folder-name",type=str, required=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # parse args
    args=parser.parse_args()
    checkpoint_path = args.checkpoint
    folder_name = args.folder_name
    
    # load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ResNet50Model(num_labels=5).to(device)
    model.load_state_dict(state_dict=checkpoint)
    
    
    # evaluate the model
    all_preds = []
    all_labels = []

    for test_images, test_labels in tqdm(test_data_loader):
        test_images, test_labels = test_images.to(device), test_labels.to(device)
        test_model_out = model(test_images)
        test_labels = test_labels.to(test_model_out.device)
        test_pred = torch.argmax(test_model_out, dim=1)
        
        all_preds.extend(test_pred.cpu().numpy())
        all_labels.extend(test_labels.cpu().numpy())
        
    
    # Calculate confusion matrix and classification report
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds) 
    


    # Plot the confusion matrix with annotations
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

    # Save the classification report to a text file
    report_path = f"artifacts/{folder_name}/classification_report.txt"
    with open(report_path, "w") as report_file:
        report_file.write("Classification Report:\n\n")
        report_file.write(class_report)
