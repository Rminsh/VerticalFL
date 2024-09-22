# strategy.py
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from sklearn.model_selection import train_test_split
import numpy as np

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
        num_clients,
        min_fit_clients,
        min_evaluate_clients,
        min_available_clients,
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
        self.num_clients=num_clients
        # The input size is sum of the embeddings from all clients
        total_embedding_size = 16 * self.num_clients  # Each client outputs embedding of size 16
        self.model = ServerModel(total_embedding_size)
        # self.model = LinearRegressionModel(total_embedding_size)
        self.initial_parameters = ndarrays_to_parameters(
            [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        # Define device and move model to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.label = torch.tensor(labels).float().unsqueeze(1)

        # To store metrics
        self.embeddings = None  # To store embeddings for evaluation
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_mae_history = []
        self.val_mae_history = []
        self.train_r2_history = []
        self.val_r2_history = []
        self.train_accuracy_history = []
        self.val_accuracy_history = []

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
        embeddings_aggregated.requires_grad = True  # Enable gradient computation

        # Move embeddings and labels to the appropriate device
        embeddings_aggregated = embeddings_aggregated.to(self.device)
        self.label = self.label.to(self.device)

        # Split embeddings and labels into training and validation sets
        (
            self.train_embeddings,
            self.val_embeddings,
            self.train_labels,
            self.val_labels,
        ) = train_test_split(
            embeddings_aggregated, self.label, test_size=0.2, random_state=42
        )

        # Training phase
        self.model.train()
        output = self.model(self.train_embeddings)
        loss = self.criterion(output, self.train_labels)
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        # Collect gradients to send back to clients
        embedding_grads = embeddings_aggregated.grad
        embedding_sizes = [16] * self.num_clients  # Embedding sizes per client
        embedding_grads_split = torch.split(embedding_grads, embedding_sizes, dim=1)
        np_grads = [grad.cpu().numpy() for grad in embedding_grads_split]
        parameters_aggregated = ndarrays_to_parameters(np_grads)

        # Save training loss
        train_loss_value = loss.item()
        self.train_loss_history.append(train_loss_value)

        # Perform evaluation on training and validation data
        with torch.no_grad():
            self.model.eval()

            # Training metrics
            train_output = self.model(self.train_embeddings)
            train_mae = torch.mean(torch.abs(train_output - self.train_labels)).item()
            train_r2 = self.calculate_r2(train_output, self.train_labels)
            train_accuracy = self.calculate_accuracy(train_output, self.train_labels)
            self.train_mae_history.append(train_mae)
            self.train_r2_history.append(train_r2)
            self.train_accuracy_history.append(train_accuracy)

            # Validation metrics
            val_output = self.model(self.val_embeddings)
            val_loss = self.criterion(val_output, self.val_labels)
            val_loss_value = val_loss.item()
            self.val_loss_history.append(val_loss_value)

            val_mae = torch.mean(torch.abs(val_output - self.val_labels)).item()
            val_r2 = self.calculate_r2(val_output, self.val_labels)
            val_accuracy = self.calculate_accuracy(val_output, self.val_labels)
            self.val_mae_history.append(val_mae)
            self.val_r2_history.append(val_r2)
            self.val_accuracy_history.append(val_accuracy)

        print(
            f"Round {rnd} - Training Loss: {train_loss_value:.4f}, "
            f"MAE: {train_mae:.4f}, R²: {train_r2:.4f}, Accuracy: {train_accuracy:.4f}"
        )
        print(
            f"Round {rnd} - Validation Loss: {val_loss_value:.4f}, "
            f"MAE: {val_mae:.4f}, R²: {val_r2:.4f}, Accuracy: {val_accuracy:.4f}"
        )

        # Return the parameters (gradients) to clients
        metrics_aggregated = {
            "train_loss": train_loss_value,
            "train_mae": train_mae,
            "train_r2": train_r2,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss_value,
            "val_mae": val_mae,
            "val_r2": val_r2,
            "val_accuracy": val_accuracy,
        }

        return parameters_aggregated, metrics_aggregated


    def aggregate_evaluate(self, rnd, results, failures):
        # Server-side evaluation; no client metrics to aggregate
        return None, {}

    def calculate_accuracy(self, outputs, labels):
        """
        Calculates the proportion of predictions within ±10% of the true labels.
        """
        absolute_errors = torch.abs(outputs - labels)
        tolerance_threshold = 1.0  # Adjust based on your data scale
        correct_predictions = torch.sum(absolute_errors <= tolerance_threshold)
        total_predictions = labels.size(0)
        accuracy = correct_predictions.item() / total_predictions
        return accuracy

    def calculate_r2(self, outputs, labels):
        """
        Calculates the R² score.
        """
        y_true = labels.cpu().numpy().flatten()
        y_pred = outputs.cpu().numpy().flatten()
        total_variance = np.var(y_true)
        residual_variance = np.var(y_true - y_pred)
        r2 = 1 - (residual_variance / total_variance)
        return r2
