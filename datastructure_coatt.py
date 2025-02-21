import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np


class CoAttention(nn.Module):
    def __init__(self, d1, d2, k):
        super(CoAttention, self).__init__()
        # une matrice de learnables weights, je vais la remplir avec ma matrice d'affinité
        self.Wc = nn.Linear(d1, k, bias=False)  
        self.Wv = nn.Linear(d1, k, bias=False)  # image
        self.Wq = nn.Linear(d2, k, bias=False)  # question
        
        #pour multiplier chaque vecteur Hv (dimension k) por whv (dimension kx1)
        self.whv = nn.Linear(k, 1, bias=False)  # produit scalaire pour l'image
        self.whq = nn.Linear(k, 1, bias=False)  # produit scalaire pour la question

    def forward(self, V, Q):
        '''
        - V est une représentation de l'image sans une matrice de (Nv x d1)
              Nv c'est la quantité de features de l'image
              d1 est la dimension du vecteur
        - Q est une représentation de la question (Nq x d2)
            Nq est la quantité de mots
            d2 est la dimendion de l'embedding
        '''
        #-------------------------------------------
        '''Relation entre los features de l'image et  ceux de la question.
        On utilise la matrice d'affinité pour que l'image et la question 'cohabitent' 
        dans un espace.
        
        Ici, je multiplie les features de l'image pour Wc et je fais la transposition
        pour pouvoir multiplier avec Q.
        
        Je multiplie Q avec WcV pour obtenir la matrice d'affinité et je l'active avec tanh.
        
        La tangente est une activation entre [-1 et 1], on l'utilise pour l'attention
        parce que comme elle prend en compte les valeurs négatifs, elle peut apprendre les affinités de 
        entre la question et l'image, mais aussi les 'repulsions' (ce que je ne demande pas). Si une valeur
        est negative, cette region de l'image n'est pas pertinente pour la question.
        
        
        '''
        C = torch.tanh(torch.matmul(Q, self.Wc(V).transpose(1, 2)))
        #--------------------------------------------------------------
        '''En utilisant la matrice d'affinité C, l'image est modifiée en fonction de la question
        et la question est modifiée en fonction de l'image'''

        Hv = torch.tanh(self.Wv(V) + torch.matmul(C, self.Wq(Q)))  # features de ;'image
        Hq = torch.tanh(self.Wq(Q) + torch.matmul(C.transpose(1, 2), self.Wv(V)))  # features de la question
        
        #-------------------------------------------------------------
        '''
        Ici, j'obtiens un vecteur qui a les probabilités des poids d'attention de chaque region de l'image
        (a quel point je veux porter mon attention) et de la question (qu'est-ce que je demande). 
        '''

        av = F.softmax(self.whv(Hv), dim=1) #dim= chaque image a un seul vector réprésentatif
        aq = F.softmax(self.whq(Hq), dim=1) 
        
        #-----------------------------------------------------------
        '''
        Multiplie les poids d'attention de l'image pour les features de l'image
        Multiplie les poids d'attention de la question pour les features de la question.
        '''
        v_hat = torch.sum(av * V, dim=1)  
        q_hat = torch.sum(aq * Q, dim=1) 

        return v_hat, q_hat

class Net(nn.Module):
    def __init__(self, d1=2048, d2=300, k=512):  #d1: imagee, d2: question
        super(Net, self).__init__()
        self.co_attention = CoAttention(d1, d2, k)
        self.fc1 = nn.Linear(d1 + d2, 1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 4)

    def forward(self, V, Q):
        v_hat, q_hat = self.co_attention(V, Q)
        x = torch.cat([v_hat, q_hat], dim=1)  #je concatene mes vecteurs d'attention
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
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

