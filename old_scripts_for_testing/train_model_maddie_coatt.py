import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datastructure import CustomDataLoader, CoAttentionMCQNet
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import re

def train_model(model, train_loader, eval_loader,criterion, optimizer, device, num_epochs=10):
    model.to(device)
    model.train()
    # for epoch in range(num_epochs):
    #     running_loss = 0.0
    #     for img_feats, ques_feats, answer_choices, labels in train_loader:
    #         img_feats, ques_feats, answer_choices, labels = img_feats.to(device), ques_feats.to(device), answer_choices.to(device), labels.to(device)
    #         optimizer.zero_grad()
    #         outputs = model(img_feats, ques_feats, answer_choices) # (B, 4)
    #         # print(inputs.shape)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #         running_loss += loss.item()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # print(inputs.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

        # validation
        val_loss = 0.0 
        model.eval()
        for img_feats, ques_feats, answer_choices, labels in eval_loader:
            img_feats, ques_feats, answer_choices, labels = img_feats.to(device), ques_feats.to(device), answer_choices.to(device), labels.to(device)
            outputs = model(img_feats, ques_feats, answer_choices)
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
        for img_feats, ques_feats, answer_choices, labels in test_loader:
            img_feats, ques_feats, answer_choices, labels = img_feats.to(device), ques_feats.to(device), answer_choices.to(device), labels.to(device)
            outputs = model(img_feats, ques_feats, answer_choices)
            accuracy = calculate_accuracy(outputs, labels)
            total_accuracy += accuracy
    return total_accuracy / len(test_loader)



def load_cleaned_dataframes(directory, prefix):
    # List all cleaned files in the directory with the specified prefix
    cleaned_files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith('.pkl')]
    
    # Sort the files numerically
    cleaned_files.sort(key=lambda f: int(re.search(r'\d+', f).group()))
    
    # Load and concatenate all cleaned DataFrames
    cleaned_dfs = []
    for cleaned_file in cleaned_files:
        print(f"Loading {cleaned_file}...")
        df = pd.read_pickle(os.path.join(directory, cleaned_file))
        cleaned_dfs.append(df)
        print(f"Loaded {cleaned_file} with shape {df.shape}")
    
    concatenated_df = pd.concat(cleaned_dfs, ignore_index=True)
    print(f"Concatenated DataFrame shape: {concatenated_df.shape}")
    return concatenated_df

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    directory = "data"
    prefix = "df_wide_clean_chunk_"
    
    # Load all cleaned DataFrames
    df = load_cleaned_dataframes(directory, prefix)
    data_loader = CustomDataLoader(df)
    train_loader = data_loader.get_train_loader()
    eval_loader = data_loader.get_eval_loader()

    model = CoAttentionMCQNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, eval_loader, criterion, optimizer, device)

    torch.save(model.state_dict(), 'vqa_model.pth')

    test_loader = data_loader.get_test_loader()
    accuracy = evaluate_model(model, test_loader, device)
    print(f'Accuracy: {accuracy:.4f}')

if __name__ == '__main__':
    main()