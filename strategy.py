# strategy.py
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

class ServerModel(nn.Module):
    def __init__(self, input_size):
        super(ServerModel, self).__init__()
        self.fc = nn.Linear(input_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x

class Strategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        labels,
        *,
        fraction_fit=1.0,
        fraction_evaluate=1.0,  # Disable client-side evaluation
        min_fit_clients=5,
        min_evaluate_clients=3,
        min_available_clients=5,
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
        total_embedding_size = 4 * 5  # Each client outputs embedding of size 4
        self.model = ServerModel(total_embedding_size)
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
        embedding_sizes = [4] * 5  # Embedding sizes per client
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
        self.mae_history.append(mae)
        self.r2_history.append(r2)
        print(f"Round {rnd} - Evaluation - MAE: {mae:.4f}, R²: {r2:.4f}")

        # Return the parameters (gradients) to clients
        metrics_aggregated = {"loss": loss_value, "mae": mae, "r2": r2}

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(self, rnd, results, failures):
        # Server-side evaluation; no client metrics to aggregate
        return None, {}
