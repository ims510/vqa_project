import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datastructure import Net, CustomDataLoader
import pandas as pd
import numpy as np

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
    """
    Transforma el DataFrame de formato largo a formato ancho para softmax.
    
    - df: DataFrame con las columnas ['index', 'question_embedding', 'image_embedding', 'choices', 'choice_embedding', 'answer']
    """
    #add index column from df.index
    df['index'] = df.index
    grouped = df.groupby('index')
    new_data = []

    for idx, group in grouped:
        group = group.sample(frac=1).reset_index(drop=True)
        question_emb = group.iloc[0]['question_embedding']
        image_emb = group.iloc[0]['feature_vector']

        choice_embs = []
        labels = []

        # Ensure all embeddings have the same number of dimensions
        question_emb = question_emb.flatten()
        image_emb = image_emb.flatten()
        choice_embs = [emb.flatten() for emb in choice_embs]
        # make a list for the 4 rows in the group to obtain a list that has 0 if the answer is FALSE and 1 if the answer is TRUE
        labels = [1 if answer == 'TRUE' else 0 for answer in group['answer']]
        split = group.iloc[0]['split']

        # Concatenar embeddings (pregunta + imagen + opciones)
        full_embedding = np.concatenate([question_emb, image_emb] + choice_embs)
        new_data.append([idx, full_embedding, labels, split])

    new_df = pd.DataFrame(new_data, columns=["index", "input_embedding", "label", "split"])
    
    return new_df

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    df = pd.read_pickle("data/df_true_false_split.pkl")
    # print("1"*50)
    # print(df.shape)
    df = transform_to_wide_format(df)
    # print("2"*50)
    # print(df.shape)
    print(df.head())
    # print(df.columns)
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