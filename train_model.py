import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datastructure import Net, CustomDataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import re

def train_model(model, train_loader, eval_loader,criterion, optimizer, device, num_epochs=10):
    model.to(device)
    model.train()
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



def transform_to_wide_format(df):

    print("Original DataFrame shape:", df.shape)
    print("Original DataFrame columns:", df.columns)
    # Convert numpy arrays to tuples to make them hashable
    df['feature_vector'] = df['feature_vector'].apply(tuple)
    df['question_embedding'] = df['question_embedding'].apply(tuple)
    
    def convert_choice_embeddings(x):
        if isinstance(x[0], list):
            return [tuple(emb) for emb in x]
        else:
            return [tuple(x)]
    
    df['choice_embeddings'] = df['choice_embeddings'].apply(convert_choice_embeddings)

    # Group by the specified columns
    grouped = df.groupby(['image_id', 'split', 'filename', 'qa_id', 'question', 'type', 'feature_vector', 'question_embedding'])

    new_data = []

    for name, group in grouped:
        # Shuffle the group rows
        group = group.sample(frac=1).reset_index(drop=True)
        
        # Flatten the choice embeddings
        choice_embs = [emb for sublist in group['choice_embeddings'].tolist() for emb in sublist]
        
        # Convert answers to binary labels
        labels = [1 if answer == 'TRUE' else 0 for answer in group['answer']]
        
        # Append the new row to new_data
        new_data.append([name[1], name[6], name[7]] + choice_embs + [labels])
    print("New data length:", len(new_data))
    # if new_data:
    #     print("First row of new data:", new_data[0])
    #     print("Length of first row of new data:", len(new_data[0]))
    # Create a new DataFrame from new_data
    new_df = pd.DataFrame(new_data, columns=["split", "feature_vector", "question_embedding", "choice1_embedding", "choice2_embedding", "choice3_embedding", "choice4_embedding", "answer"])
    
    print("New DataFrame shape:", new_df.shape)
    print("New DataFrame columns:", new_df.columns)
    return new_df
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
    # df = pd.read_pickle("data/df_true_false_split.pkl")
    # # # print("1"*50)
    # # # print(df.shape)
    # # print(df.head())
    # # print(df.columns)
    
    # # df = transform_to_wide_format(df)
    # # print("2"*50)
    # # print(df.shape)
    # # print(df.head())
    # # print(df.columns)

    # # Read the DataFrame in chunks
    # # Process the DataFrame in chunks
    # chunk_size = 10000
    # num_chunks = len(df) // chunk_size + 1
    # transformed_chunks = []
    
    # for i in tqdm(range(num_chunks)):
    #     print(f"Processing chunk {i+1}/{num_chunks}")
    #     chunk = df[i*chunk_size:(i+1)*chunk_size]
    #     transformed_chunk = transform_to_wide_format(chunk)
    #     transformed_chunks.append(transformed_chunk)
    #     print(f"Finished processing chunk {i+1}/{num_chunks}")
    
    # # Concatenate all transformed chunks
    # print("Concatenating transformed chunks...")
    # df = pd.concat(transformed_chunks, ignore_index=True)
    # print("Finished concatenating transformed chunks.")
    # print(df.shape)
    
    #     # Save the transformed DataFrame in smaller chunks
    # print("Saving transformed DataFrame to pickle file in chunks...")
    # chunk_size = 10000
    # num_chunks = len(df) // chunk_size + 1
    
    # for i in range(num_chunks):
    #     chunk = df[i*chunk_size:(i+1)*chunk_size]
    #     chunk.to_pickle(f"data/df_wide_chunk_{i+1}.pkl")
    #     print(f"Finished saving chunk {i+1}/{num_chunks}")
    
    # print("Finished saving all chunks of the transformed DataFrame.")

    directory = "data"
    prefix = "df_wide_clean_chunk_"
    
    # Load all cleaned DataFrames
    df = load_cleaned_dataframes(directory, prefix)
    data_loader = CustomDataLoader(df)
    train_loader = data_loader.get_train_loader()
    eval_loader = data_loader.get_eval_loader()

    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, eval_loader, criterion, optimizer, device)

    torch.save(model.state_dict(), 'vqa_model.pth')

    test_loader = data_loader.get_test_loader()
    accuracy = evaluate_model(model, test_loader, device)
    print(f'Accuracy: {accuracy:.4f}')

if __name__ == '__main__':
    main()