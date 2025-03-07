import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datastructure import NetSimple, CoAttentionNetSimple, CoAttentionNetBNDropout, CustomDataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import re
import argparse

def train_model(model, train_loader, eval_loader,criterion, optimizer, device, num_epochs=10):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

        # validation
        val_loss = 0.0 
        model.eval()
        for inputs, labels in eval_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
        print(f'Validation Loss: {val_loss/len(eval_loader):.4f}')
        model.train()

def calculate_accuracy(predictions, labels):
    _, predicted = torch.max(predictions, 1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy

def evaluate_model(model, test_loader, device):
    model.to(device)
    model.eval()
    total_accuracy = 0
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            accuracy = calculate_accuracy(outputs, labels)
            total_accuracy += accuracy
    return total_accuracy / len(test_loader)

def main():
    parser = argparse.ArgumentParser(description='Train a VQA model.')
    parser.add_argument('--model', type=str, choices=['NetSimple', 'CoAttentionNetSimple', 'CoAttentionNetBNDropout'], required=True, help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the model')
    parser.add_argument('--save_model', action='store_true', help='Save the trained model')
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    df = pd.read_pickle("data/final_clean_df.pkl")
    data_loader = CustomDataLoader(df)
    print("Train data columns shape:")
    train_loader = data_loader.get_train_loader()
    print("-"*50)
    print("Eval data columns shape:")
    eval_loader = data_loader.get_eval_loader()
    print("-"*50)

    if args.model == 'NetSimple':
        model = NetSimple().to(device)
    elif args.model == 'CoAttentionNetSimple':
        model = CoAttentionNetSimple().to(device)
    elif args.model == 'CoAttentionNetBNDropout':
        model = CoAttentionNetBNDropout().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training model...")
    train_model(model, train_loader, eval_loader, criterion, optimizer, device, num_epochs=args.epochs)

    if args.save_model:
        torch.save(model.state_dict(), 'vqa_model.pth')

    print("-"*50)
    print("Test data columns shape:")
    test_loader = data_loader.get_test_loader()
    print("-"*50)
    accuracy = evaluate_model(model, test_loader, device)
    print(f'Accuracy: {accuracy:.4f}')

if __name__ == '__main__':
    main()