# simulation.py
import flwr as fl
import matplotlib.pyplot as plt

from client import FlowerClient
from strategy import Strategy
from task import get_partitions_and_label
from client import set_random_seeds

num_clients = 5
num_rounds = 70
details = str(num_clients) + 'clients_' + str(num_rounds) + 'rounds_'
partitions, labels = get_partitions_and_label(num_clients)

def client_fn(cid: str):
    print(f"Creating client with cid: {cid}")
    cid_int = int(cid)
    set_random_seeds(42)  # Set random seeds for reproducibility
    data = partitions[cid_int]
    return FlowerClient(cid_int, data).to_client()

if __name__ == "__main__":
    # Create the strategy with appropriate parameters
    strategy = Strategy(
        labels=labels,
        fraction_fit=1.0,            # Sample 100% of available clients
        fraction_evaluate=0.0,       # Disable evaluation rounds
        min_fit_clients=num_clients,           # Minimum number of clients to be sampled for training
        min_evaluate_clients=num_clients,     # Should be 0 since fraction_evaluate=0.0
        min_available_clients=num_clients,     # Minimum number of clients that need to be connected
        accept_failures=True         # Accept failures to handle clients that are not created
    )

    # Start the simulation
    hist = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    # After training, plot the loss over rounds
    rounds = range(1, len(strategy.loss_history) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(rounds, strategy.loss_history, marker='o')
    plt.title('Global Model Loss over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Loss (MSE)')
    plt.grid(True)
    plt.savefig('results/' + details + 'loss_over_rounds.png')
    plt.show()

    # Plot MAE over rounds
    plt.figure(figsize=(10, 5))
    plt.plot(rounds, strategy.mae_history, marker='o', color='orange')
    plt.title('Global Model MAE over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Mean Absolute Error')
    plt.grid(True)
    plt.savefig('results/' + details + 'mae_over_rounds.png')
    plt.show()

    # Plot R² over rounds
    plt.figure(figsize=(10, 5))
    plt.plot(rounds, strategy.r2_history, marker='o', color='green')
    plt.title('Global Model R² Score over Rounds')
    plt.xlabel('Round')
    plt.ylabel('R² Score')
    plt.grid(True)
    plt.savefig('results/' + details + 'r2_over_rounds.png')
    plt.show()

    # if hasattr(strategy, 'rmse_history') and strategy.rmse_history:
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(rounds, strategy.rmse_history, marker='o', color='red', label='RMSE')
    #     plt.title('Global Model RMSE over Rounds')
    #     plt.xlabel('Round')
    #     plt.ylabel('Root Mean Squared Error')
    #     plt.grid(True)
    #     plt.legend()
    #     plt.savefig('rmse_over_rounds.png')
    #     plt.show()