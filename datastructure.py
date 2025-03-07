"""
This script contains 3 neural network architectures for a visual question answering task. The first architecture is a simple
feedforward neural network with 3 hidden layers. The second architecture is a co-attention network that uses attention to
combine the image features and question embeddings. The third architecture is a co-attention network with batch normalization
and dropout layers. The script also contains a custom data loader class that prepares the data for training and evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.linalg import multi_dot


class NetSimple(nn.Module):
    """
    A simple feedforward neural network with 3 hidden layers.
    """
    def __init__(self):
        super(NetSimple, self).__init__()
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


class CoAttentionNetSimple(nn.Module):
    """
    A co-attention network that uses attention to combine the image features and question embeddings.
    """
    def __init__(self, hidden_dim=512):
        super(CoAttentionNetSimple, self).__init__()

        self.hidden_dim = hidden_dim

        # Placeholder for the weight matrix, will be initialized in forward pass
        self.weight_matrix = None

        # Attention transformation layers
        self.Wv = None
        self.Wq = None

        # Attention score layers
        self.w_v = nn.Linear(hidden_dim, 1)
        self.w_q = nn.Linear(hidden_dim, 1)

        self.fc1 = nn.Linear(8704, 1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 4)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        """
        Inputs are features from the image, question, and answer choices of the following
        shapes:
        image_features: (B, 2048)
        question_embedding: (B, 768)
        choice_1: (B, 768)
        choice_2: (B, 768)
        choice_3: (B, 768)
        choice_4: (B, 768)
        """
        image_features = inputs[:, :2048]
        question_embedding = inputs[:, 2048:2816]
        choice_1 = inputs[:, 2816:3584]
        choice_2 = inputs[:, 3584:4352]
        choice_3 = inputs[:, 4352:5120]
        choice_4 = inputs[:, 5120:5888]

        batch_size = inputs.size(0)
        device = inputs.device
        # Initialize the weight matrix based on the batch size
        if self.weight_matrix is None or self.weight_matrix.size(0) != batch_size:
            self.weight_matrix = nn.Parameter(
                torch.randn(batch_size, batch_size).to(device)
            )

        # Initialize the attention transformation layers based on the input dimensions
        # because it can be not the batch size we give as argument if the division of rows by batches is not exact
        if self.Wv is None or self.Wv.in_features != batch_size:
            self.Wv = nn.Linear(batch_size, self.hidden_dim).to(device)
        if self.Wq is None or self.Wq.in_features != batch_size:
            self.Wq = nn.Linear(batch_size, self.hidden_dim).to(device)

        affinity_matrix = torch.matmul(
            question_embedding.transpose(1, 0), self.weight_matrix
        )
        affinity_matrix = torch.matmul(affinity_matrix, image_features) # affinity matrix: torch.Size([768, 2048])
        affinity_matrix = torch.tanh(affinity_matrix)  # Apply tanh activation

        # Compute Attention Scores
        H_v = torch.tanh(
            self.Wv(image_features.transpose(1, 0)).transpose(1, 0)
            + torch.matmul(
                self.Wq(question_embedding.transpose(1, 0)).transpose(1, 0),
                affinity_matrix,
            )
        )  # Hv shape: torch.Size([512, 2048])
        H_q = torch.tanh(
            self.Wq(question_embedding.transpose(1, 0)).transpose(1, 0)
            + torch.matmul(
                self.Wv(image_features.transpose(1, 0)).transpose(1, 0),
                affinity_matrix.transpose(1, 0),
            )
        )  # Hq shape: torch.Size([512, 768])


        # Compute attention weights
        a_v = F.softmax(
            self.w_v(H_v.transpose(1, 0)), dim=1
        )  # a_v shape: torch.Size([2048, 1]),
        a_q = F.softmax(
            self.w_q(H_q.transpose(1, 0)), dim=1
        )  # a_q shape: torch.Size([768, 1])

        a_v = a_v.squeeze()  # now it's shape 2048
        a_q = a_q.squeeze()  # now it's shape 768

        # Apply attention weights to get weighted representations
        v_hat = a_v * image_features  # v_hat shape (torch.Size([32, 2048])
        q_hat = a_q * question_embedding  # q_hat shape: torch.Size([32, 768])

        combined_features = torch.cat(
            (
                image_features,
                question_embedding,
                choice_1,
                choice_2,
                choice_3,
                choice_4,
                v_hat,
                q_hat,
            ),
            dim=1,
        )

        x = self.fc1(combined_features)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return self.softmax(x)


class CoAttentionNetBNDropout(nn.Module):
    """
    A co-attention network with batch normalization and dropout layers.
    """
    def __init__(self, hidden_dim=512):
        super(CoAttentionNetBNDropout, self).__init__()

        self.hidden_dim = hidden_dim

        # Placeholder for the weight matrix, will be initialized in forward pass
        self.weight_matrix = None

        # Attention transformation layers
        self.Wv = None
        self.Wq = None

        # Attention score layers
        self.w_v = nn.Linear(hidden_dim, 1)
        self.w_q = nn.Linear(hidden_dim, 1)

        self.fc1 = nn.Linear(8704, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(512, 4)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        """
        Inputs are features from the image, question, and answer choices of the following
        shapes:
        image_features: (B, 2048)
        question_embedding: (B, 768)
        choice_1: (B, 768)
        choice_2: (B, 768)
        choice_3: (B, 768)
        choice_4: (B, 768)
        """
        image_features = inputs[:, :2048]
        question_embedding = inputs[:, 2048:2816]
        choice_1 = inputs[:, 2816:3584]
        choice_2 = inputs[:, 3584:4352]
        choice_3 = inputs[:, 4352:5120]
        choice_4 = inputs[:, 5120:5888]

        batch_size = inputs.size(0)
        device = inputs.device
        # Initialize the weight matrix based on the batch size
        if self.weight_matrix is None or self.weight_matrix.size(0) != batch_size:
            self.weight_matrix = nn.Parameter(
                torch.randn(batch_size, batch_size).to(device)
            )

        # Initialize the attention transformation layers based on the input dimensions
        # because it can be not the batch size we give as argument if the division of rows by batches is not exact
        if self.Wv is None or self.Wv.in_features != batch_size:
            self.Wv = nn.Linear(batch_size, self.hidden_dim).to(device)
        if self.Wq is None or self.Wq.in_features != batch_size:
            self.Wq = nn.Linear(batch_size, self.hidden_dim).to(device)

        affinity_matrix = torch.matmul(
            question_embedding.transpose(1, 0), self.weight_matrix
        )
        affinity_matrix = torch.matmul(affinity_matrix, image_features) # affinity matrix: torch.Size([768, 2048])
        affinity_matrix = torch.tanh(affinity_matrix)  # Apply tanh activation

        # Compute Attention Scores
        H_v = torch.tanh(
            self.Wv(image_features.transpose(1, 0)).transpose(1, 0)
            + torch.matmul(
                self.Wq(question_embedding.transpose(1, 0)).transpose(1, 0),
                affinity_matrix,
            )
        )  # Hv shape: torch.Size([512, 2048])
        H_q = torch.tanh(
            self.Wq(question_embedding.transpose(1, 0)).transpose(1, 0)
            + torch.matmul(
                self.Wv(image_features.transpose(1, 0)).transpose(1, 0),
                affinity_matrix.transpose(1, 0),
            )
        )  # Hq shape: torch.Size([512, 768])

        # Compute attention weights
        a_v = F.softmax(
            self.w_v(H_v.transpose(1, 0)), dim=1
        )  # a_v shape: torch.Size([2048, 1]),
        a_q = F.softmax(
            self.w_q(H_q.transpose(1, 0)), dim=1
        )  # a_q shape: torch.Size([768, 1])


        a_v = a_v.squeeze()  # now it's shape 2048
        a_q = a_q.squeeze()  # now it's shape 768

        # Apply attention weights to get weighted representations
        v_hat = a_v * image_features  # v_hat shape: torch.Size([32, 2048])
        q_hat = a_q * question_embedding  # q_hat shape: torch.Size([32, 768])

        combined_features = torch.cat(
            (
                image_features,
                question_embedding,
                choice_1,
                choice_2,
                choice_3,
                choice_4,
                v_hat,
                q_hat,
            ),
            dim=1,
        )

        x = self.fc1(combined_features)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
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
        self.train_data = (
            self.dataframe[self.dataframe["split"] == "train"]
            .drop("split", axis=1)
            .reset_index(drop=True)
        )
        self.test_data = (
            self.dataframe[self.dataframe["split"] == "test"]
            .drop("split", axis=1)
            .reset_index(drop=True)
        )
        self.eval_data = (
            self.dataframe[self.dataframe["split"] == "val"]
            .drop("split", axis=1)
            .reset_index(drop=True)
        )

    def get_train_loader(self):
        return self._create_loader(self.train_data)

    def get_test_loader(self):
        return self._create_loader(self.test_data)

    def get_eval_loader(self):
        return self._create_loader(self.eval_data)

    def _create_loader(self, data):
        feature_columns = [
            "feature_vector",
            "question_embedding",
            "choice1_embedding",
            "choice2_embedding",
            "choice3_embedding",
            "choice4_embedding",
        ]

        feature_matrices = []

        for col in feature_columns:
            try:
                # Convert each cell into a NumPy array
                column_data = np.array(
                    [
                        np.array(eval(x)) if isinstance(x, str) else np.array(x)
                        for x in data[col].values
                    ]
                )

                print(
                    f"Column: {col}, Extracted Shape: {column_data.shape}"
                )  # Should be (N, embedding_dim)

                feature_matrices.append(column_data)
            except Exception as e:
                print(f"Error processing column {col}: {e}")

        # Stack along axis 1
        features = np.concatenate(feature_matrices, axis=1)
        # print(f"Final feature matrix shape: {features.shape}")  # Should be (N, 5888)

        features = torch.tensor(features).float()


        labels = torch.tensor(
            np.array(data["answer_array"].tolist()).argmax(axis=1)
        ).long()

        dataset = TensorDataset(features, labels)
        print(f"Dataset length: {len(dataset)}")
        print(f"Dataset features shape: {dataset[0][0].shape}")
        print(f"Dataset labels shape: {dataset[0][1].shape}")

        return DataLoader(dataset, batch_size=self.batch_size)
