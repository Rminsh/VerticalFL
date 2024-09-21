# strategy.py
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

num_clients = 5

class ServerModel(nn.Module):
    def __init__(self, input_size):
        super(ServerModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # Increased neurons
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 1)  # Output layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

class LinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)  # Single linear layer

    def forward(self, x):
        return self.linear(x)

class Strategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        labels,
        *,
        fraction_fit=1.0,
        fraction_evaluate=1.0,  # Disable client-side evaluation
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_fn=None,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        accept_failures=True,
        initial_parameters=None,
        fit_metrics_aggregation_fn=None,
        evaluate_metrics_aggregation_fn=None,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        # The input size is sum of the embeddings from all clients
        total_embedding_size = 16 * num_clients  # Each client outputs embedding of size 16
        self.model = ServerModel(total_embedding_size)
        # self.model = LinearRegressionModel(total_embedding_size)
        self.initial_parameters = ndarrays_to_parameters(
            [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.label = torch.tensor(labels).float().unsqueeze(1)

        # To store metrics
        self.loss_history = []
        self.mae_history = []
        self.r2_history = []
        self.rmse_history = []
        self.embeddings = None  # To store embeddings for evaluation

    def aggregate_fit(
        self,
        rnd,
        results,
        failures,
    ):
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Collect embeddings from clients
        embedding_results = [
            torch.from_numpy(parameters_to_ndarrays(fit_res.parameters)[0])
            for _, fit_res in results
        ]
        # Concatenate embeddings along the feature dimension
        embeddings_aggregated = torch.cat(embedding_results, dim=1)
        embedding_server = embeddings_aggregated.detach().requires_grad_()
        # Forward pass
        output = self.model(embedding_server)
        loss = self.criterion(output, self.label)
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        # Store embeddings for evaluation
        self.embeddings = embedding_server.detach()

        # Collect gradients to send back to clients
        # Each client gets the gradient corresponding to its embedding
        embedding_sizes = [16] * num_clients  # Embedding sizes per client
        embedding_grads = embedding_server.grad.split(embedding_sizes, dim=1)
        np_grads = [grad.numpy() for grad in embedding_grads]
        parameters_aggregated = ndarrays_to_parameters(np_grads)

        # Save loss for plotting
        loss_value = loss.item()
        self.loss_history.append(loss_value)
        print(f"Round {rnd} - Loss: {loss_value:.4f}")

        # Perform evaluation
        with torch.no_grad():
            output = self.model(self.embeddings)
            mae = torch.mean(torch.abs(output - self.label)).item()
            total_variance = torch.var(self.label)
            residual_variance = torch.var(self.label - output)
            r2 = 1 - (residual_variance / total_variance)
            r2 = r2.item()
            mse = torch.mean((output - self.label) ** 2).item()
            rmse = mse ** 0.5
        self.mae_history.append(mae)
        self.r2_history.append(r2)
        # RMSE history
        self.rmse_history.append(rmse)
        print(f"Round {rnd} - Evaluation - MAE: {mae:.4f}, R²: {r2:.4f}, RMSE: {rmse:.4f}")

        # Return the parameters (gradients) to clients
        metrics_aggregated = {"loss": loss_value, "mae": mae, "r2": r2}

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(self, rnd, results, failures):
        # Server-side evaluation; no client metrics to aggregate
        return None, {}
