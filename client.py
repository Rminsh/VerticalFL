# client.py
import flwr as fl
import torch
import random
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch.optim as optim


from task import MLPClientModel
from task import LinearRegressionClientModel

# Function to set random seeds for reproducibility
def set_random_seeds(seed_value=42):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, data):
        self.cid = cid
        # Standardize the data
        self.scaler = StandardScaler()
        self.train = torch.tensor(self.scaler.fit_transform(data)).float()
        input_size = self.train.shape[1]
        self.model = MLPClientModel(input_size=input_size, embedding_size=16)
        # self.model = LinearRegressionClientModel(input_size=input_size, embedding_size=16)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.embedding = None

    def get_parameters(self, config):
        pass  # Not used in vertical federated learning

    def fit(self, parameters, config):
        # Compute the embeddings
        #---- Used for LinearRegressionModel
        # self.model.train()
        # self.optimizer.zero_grad()
        #---- 

        self.embedding = self.model(self.train)
        # No need to compute loss or backward here; server handles it
        return [self.embedding.detach().numpy()], len(self.train), {}

    def evaluate(self, parameters, config):
        # Update the model using the received gradients
        #---- Used for MLP
        self.model.zero_grad()
        #----
        #---- Used for LinearRegressionModel
        # self.model.train()
        #----

        # parameters is a list of numpy arrays, one per client
        # We need to get the gradient corresponding to this client
        gradients = torch.from_numpy(parameters[int(self.cid)])
        #---- Used for LinearRegressionModel
        # Ensure that gradients are properly shaped
        # if self.embedding.grad is not None:
            # self.embedding.grad.zero_()
        #----
        self.embedding.backward(gradients)
        self.optimizer.step()
        return 0.0, len(self.train), {}
