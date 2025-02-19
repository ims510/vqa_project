import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(5888, 1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 4)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return self.softmax(x)

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
        self.train_data = self.dataframe[self.dataframe['split'] == 'train'].drop('split', axis=1).reset_index(drop=True)
        self.test_data = self.dataframe[self.dataframe['split'] == 'test'].drop('split', axis=1).reset_index(drop=True)
        self.eval_data = self.dataframe[self.dataframe['split'] == 'val'].drop('split', axis=1).reset_index(drop=True)

    def get_train_loader(self):
        return self._create_loader(self.train_data)

    def get_test_loader(self):
        return self._create_loader(self.test_data)

    def get_eval_loader(self):
        return self._create_loader(self.eval_data)

    def _create_loader(self, data):
        feature_columns = ['feature_vector', 'question_embedding', 'choice1_embedding', 
                        'choice2_embedding', 'choice3_embedding', 'choice4_embedding']
        
        feature_matrices = []
        
        for col in feature_columns:
            try:
                # Convert each cell into a NumPy array
                column_data = np.array([np.array(eval(x)) if isinstance(x, str) else np.array(x) for x in data[col].values])
                
                print(f"Column: {col}, Extracted Shape: {column_data.shape}")  # Should be (N, embedding_dim)
                
                feature_matrices.append(column_data)
            except Exception as e:
                print(f"Error processing column {col}: {e}")

        # Stack along axis 1
        features = np.concatenate(feature_matrices, axis=1)
        print(f"Final feature matrix shape: {features.shape}")  # Should be (N, 5888)
        
        features = torch.tensor(features).float()

        # print(data['answer'].head())
        # print(data['answer'].dtype)
        labels = torch.tensor(np.array(data['answer'].tolist()).argmax(axis=1)).long()
        
        dataset = TensorDataset(features, labels)
        return DataLoader(dataset, batch_size=self.batch_size)

