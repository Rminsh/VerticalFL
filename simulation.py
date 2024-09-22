# simulation.py
import flwr as fl
import matplotlib.pyplot as plt

from client import FlowerClient
from strategy import Strategy
from task import get_partitions_and_label
from client import set_random_seeds

num_clients_1 = 3
num_clients_2 = 5
num_clients_3 = 6
num_rounds = 70
details = str(num_rounds) + 'rounds_'
partitions_1, labels_1 = get_partitions_and_label(num_clients_1)
partitions_2, labels_2 = get_partitions_and_label(num_clients_2)
partitions_3, labels_3 = get_partitions_and_label(num_clients_3)

def client_fn_1(cid: str):
    print(f"Creating client with cid: {cid}")
    cid_int = int(cid)
    set_random_seeds(42)  # Set random seeds for reproducibility
    data = partitions_1[cid_int]
    return FlowerClient(cid_int, data).to_client()

def client_fn_2(cid: str):
    print(f"Creating client with cid: {cid}")
    cid_int = int(cid)
    set_random_seeds(42)  # Set random seeds for reproducibility
    data = partitions_2[cid_int]
    return FlowerClient(cid_int, data).to_client()

def client_fn_3(cid: str):
    print(f"Creating client with cid: {cid}")
    cid_int = int(cid)
    set_random_seeds(42)  # Set random seeds for reproducibility
    data = partitions_3[cid_int]
    return FlowerClient(cid_int, data).to_client()

if __name__ == "__main__":
    # Create the strategy with appropriate parameters
    strategy_1 = Strategy(
        labels=labels_1,
        fraction_fit=1.0,            # Sample 100% of available clients
        fraction_evaluate=0.0,       # Disable evaluation rounds
        num_clients=num_clients_1,
        min_fit_clients=num_clients_1,           # Minimum number of clients to be sampled for training
        min_evaluate_clients=num_clients_1,     # Should be 0 since fraction_evaluate=0.0
        min_available_clients=num_clients_1,     # Minimum number of clients that need to be connected
        accept_failures=True         # Accept failures to handle clients that are not created
    )

    strategy_2 = Strategy(
        labels=labels_2,
        fraction_fit=1.0,            # Sample 100% of available clients
        fraction_evaluate=0.0,       # Disable evaluation rounds
        num_clients=num_clients_2,
        min_fit_clients=num_clients_2,           # Minimum number of clients to be sampled for training
        min_evaluate_clients=num_clients_2,     # Should be 0 since fraction_evaluate=0.0
        min_available_clients=num_clients_2,     # Minimum number of clients that need to be connected
        accept_failures=True         # Accept failures to handle clients that are not created
    )

    strategy_3 = Strategy(
        labels=labels_3,
        fraction_fit=1.0,            # Sample 100% of available clients
        fraction_evaluate=0.0,       # Disable evaluation rounds
        num_clients=num_clients_3,
        min_fit_clients=num_clients_3,           # Minimum number of clients to be sampled for training
        min_evaluate_clients=num_clients_3,     # Should be 0 since fraction_evaluate=0.0
        min_available_clients=num_clients_3,     # Minimum number of clients that need to be connected
        accept_failures=True         # Accept failures to handle clients that are not created
    )

    # Start the simulation
    hist_1 = fl.simulation.start_simulation(
        client_fn=client_fn_1,
        num_clients=num_clients_1,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy_1,
    )

    hist_2 = fl.simulation.start_simulation(
        client_fn=client_fn_2,
        num_clients=num_clients_2,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy_2,
    )

    hist_3 = fl.simulation.start_simulation(
        client_fn=client_fn_3,
        num_clients=num_clients_3,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy_3,
    )

    # After training, plot the loss over rounds
    rounds_1 = range(1, len(strategy_1.loss_history) + 1)
    rounds_2 = range(1, len(strategy_2.loss_history) + 1)
    rounds_3 = range(1, len(strategy_3.loss_history) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(rounds_1, strategy_1.loss_history, marker='o', color='blue', label='3 Clients')
    plt.plot(rounds_2, strategy_2.loss_history, marker='o', color='gold', label='5 Clients')
    plt.plot(rounds_3, strategy_3.loss_history, marker='o', color='crimson', label='6 Clients')
    plt.title('Global Model Loss over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Loss (MSE)')
    plt.grid(True)
    plt.legend()
    plt.savefig('results/' + details + 'loss_over_rounds.png')
    plt.show()

    # Plot MAE over rounds
    plt.figure(figsize=(10, 5))
    plt.plot(rounds_1, strategy_1.mae_history, marker='o', color='orange', label='3 Clients')
    plt.plot(rounds_2, strategy_2.mae_history, marker='o', color='lime', label='5 Clients')
    plt.plot(rounds_3, strategy_3.mae_history, marker='o', color='royalblue', label='6 Clients')
    plt.title('Global Model MAE over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Mean Absolute Error')
    plt.grid(True)
    plt.legend()
    plt.savefig('results/' + details + 'mae_over_rounds.png')
    plt.show()

    # Plot R² over rounds
    plt.figure(figsize=(10, 5))
    plt.plot(rounds_1, strategy_1.r2_history, marker='o', color='green', label='3 Clients')
    plt.plot(rounds_2, strategy_2.r2_history, marker='o', color='peru', label='5 Clients')
    plt.plot(rounds_3, strategy_3.r2_history, marker='o', color='gray', label='6 Clients')
    plt.title('Global Model R² Score over Rounds')
    plt.xlabel('Round')
    plt.ylabel('R² Score')
    plt.grid(True)
    plt.legend()
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