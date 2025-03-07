import torch
import

def calculate_accuracy(predictions, labels):
    _, predicted = torch.max(predictions, 1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy

def evaluate_model(model, data_loader, device):
    model.eval()
    total_accuracy = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            accuracy = calculate_accuracy(outputs, labels)
            total_accuracy += accuracy
    return total_accuracy / len(data_loader)

def main():
    df = pd.read_pickle("data/df_true_false_split.pkl")
    data_loader = DataLoader(df)