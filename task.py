# task.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

def get_partitions_and_label():
    # Load the dataset
    data = pd.read_csv("merged_output.csv")
    data = data.drop(columns=['timestamp'])
    
    # Sort the data by 'sleep_log_entry_id' to align across clients
    data.sort_values('sleep_log_entry_id', inplace=True)
    data.reset_index(drop=True, inplace=True)
    
    # Fill missing values
    data.fillna(data.mean(), inplace=True)
    
    # Prepare features and labels
    # Target variable
    labels = data['overall_score'].values.reshape(-1, 1)
    
    # Standardize labels
    label_scaler = StandardScaler()
    labels = label_scaler.fit_transform(labels).flatten()
    
    # Partition the features among clients
    # Client 0 features (Label holder)
    client0_features = data[['composition_score', 'revitalization_score', 'duration_score',
                             'deep_sleep_in_minutes', 'resting_heart_rate']]
    
    # Client 1 features
    client1_features = data[['restlessness', 'minutes_asleep', 'minutes_awake',
                             'minutes_after_wakeup', 'time_in_bed', 'efficiency',
                             'duration', 'minutes_to_fall_asleep']]
    
    # Ensure that the features are numpy arrays
    client0_features = client0_features.values
    client1_features = client1_features.values
    
    # Prepare partitions
    partitions = [client0_features, client1_features]
    
    return partitions, labels

class ClientModel(nn.Module):
    def __init__(self, input_size):
        super(ClientModel, self).__init__()
        self.fc = nn.Linear(input_size, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x
