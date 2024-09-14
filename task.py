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

    # Prepare labels
    labels = data['overall_score'].values.reshape(-1, 1)
    
    # Standardize labels
    label_scaler = StandardScaler()
    labels = label_scaler.fit_transform(labels).flatten()

    # Partition the features among clients
    # Client 0 (Label holder)
    client0_features = data[['composition_score', 'revitalization_score']]

    # Client 1
    client1_features = data[['duration_score', 'deep_sleep_in_minutes']]

    # Client 2
    client2_features = data[['resting_heart_rate', 'restlessness']]

    # Client 3
    client3_features = data[['minutes_asleep', 'minutes_awake', 'minutes_after_wakeup']]

    # Client 4
    client4_features = data[['time_in_bed', 'efficiency', 'duration', 'minutes_to_fall_asleep']]

    # Convert features to numpy arrays
    partitions = [
        client0_features.values,
        client1_features.values,
        client2_features.values,
        client3_features.values,
        client4_features.values,
    ]

    return partitions, labels

class ClientModel(nn.Module):
    def __init__(self, input_size, embedding_size=4):
        super(ClientModel, self).__init__()
        self.fc = nn.Linear(input_size, embedding_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x
