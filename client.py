# client.py
import flwr as fl
import torch
from sklearn.preprocessing import StandardScaler

from task import ClientModel

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, data):
        self.cid = cid
        # Standardize the data
        self.scaler = StandardScaler()
        self.train = torch.tensor(self.scaler.fit_transform(data)).float()
        self.model = ClientModel(input_size=self.train.shape[1])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.embedding = None

    def get_parameters(self, config):
        pass  # Not used in vertical federated learning

    def fit(self, parameters, config):
        # Compute the embeddings
        self.embedding = self.model(self.train)
        return [self.embedding.detach().numpy()], len(self.train), {}

    def evaluate(self, parameters, config):
        # Update the model using the received gradients
        self.model.zero_grad()
        # parameters is a list of numpy arrays, one per client
        # We need to get the gradient corresponding to this client
        gradients = torch.from_numpy(parameters[int(self.cid)])
        self.embedding.backward(gradients)
        self.optimizer.step()
        return 0.0, len(self.train), {}
