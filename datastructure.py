import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import ast

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.fc1 = nn.Linear(5888, 1024)
        # self.fc2 = nn.Linear(1024, 4)  # Output layer for 4 multiple-choice answers

        self.fc1 = nn.Linear(2816,1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 4)
        
    def forward(self, x):
        # if x.shape[1] != 5888:
        #     raise ValueError(f"Expected input shape (batch_size, 5888), but got {x.shape}")
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)  # No activation here, as we will apply softmax in the loss function
        # return x
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class CustomDataLoader:
    def __init__(self, dataframe, batch_size=32):
        if isinstance(dataframe, pd.DataFrame):
            self.dataframe = dataframe
        else:
            raise ValueError("Expected a pandas DataFrame")
        self.batch_size = batch_size
        self.train_data = None
        self.test_data = None
        self.eval_data = None
        self.prepare_data()

    def prepare_data(self):
        # print("Preparing data...")
        # print(self.dataframe.head())  # Print the first few rows of the DataFrame for debugging
        
        # Train is the df if the value in the column split is "train", same for test and val
        self.train_data = self.dataframe[self.dataframe['split'] == 'train'].drop('split', axis=1).reset_index(drop=True)
        self.test_data = self.dataframe[self.dataframe['split'] == 'test'].drop('split', axis=1).reset_index(drop=True)
        self.eval_data = self.dataframe[self.dataframe['split'] == 'val'].drop('split', axis=1).reset_index(drop=True)


        # self.train_data['input_embedding'] = self.train_data['input_embedding'].apply(ast.literal_eval)
        # self.test_data['input_embedding'] = self.test_data['input_embedding'].apply(ast.literal_eval)
        # self.eval_data['input_embedding'] = self.eval_data['input_embedding'].apply(ast.literal_eval)

        if not self.train_data.empty:
            self.input_dim = len(self.train_data['input_embedding'].iloc[0])

        print(f"input_dim: {self.input_dim}") 
        # print("Train data:")
        # print(self.train_data.head())
        # print("Test data:")
        # print(self.test_data.head())
        # print("Eval data:")
        # print(self.eval_data.head())

    def get_train_loader(self):
        return self._create_loader(self.train_data)

    def get_test_loader(self):
        return self._create_loader(self.test_data)

    def get_eval_loader(self):
        return self._create_loader(self.eval_data)

    # def _create_loader(self, data):
    #     data = data.apply(pd.to_numeric, errors='coerce').fillna(0)
    #     # features are all the columns but the one called "answer"
    #     features = torch.tensor(data.drop('answer', axis=1).values).float()
    #     # labels are 0 if it's false and 1 if it's true
    #     labels = torch.tensor((data['answer'] == 'TRUE').astype(int).values).long()

    #     # Check the shapes of features and labels
    #     assert features.ndim == 2, f"Expected features to be 2D, got {features.ndim}D"
    #     assert labels.ndim == 1, f"Expected labels to be 1D, got {labels.ndim}D"

    #     dataset = TensorDataset(features, labels)
    #     return DataLoader(dataset, batch_size=self.batch_size)

    def _create_loader(self, data):
        # print("Creating loader...")
        # print(data.head())  # Print the first few rows of the DataFrame for debugging

        data = data.apply(pd.to_numeric, errors='coerce').fillna(0)
        # print("3"*50)
        # print(data.shape)
        # Ensure 'split' column is not included in features
        features = torch.tensor(data.drop(['label'], axis=1).values).float()
        labels = torch.tensor(data['label'].values).long()

        # print(f"Features shape: {features.shape}")
        # print(f"Labels shape: {labels.shape}")
        
        # assert features.ndim == 2, f"Expected features to be 2D, got {features.ndim}D"
        # assert labels.ndim == 1, f"Expected labels to be 1D, got {labels.ndim}D"
        
        dataset = TensorDataset(features, labels)
        return DataLoader(dataset, batch_size=self.batch_size)
    

