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

    # Run simulations
    print("Starting Simulation 1 with 3 Clients")
    hist_1 = fl.simulation.start_simulation(
        client_fn=client_fn_1,
        num_clients=num_clients_1,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy_1,
    )
    print("Simulation 1 Completed\n")

    print("Starting Simulation 2 with 5 Clients")
    hist_2 = fl.simulation.start_simulation(
        client_fn=client_fn_2,
        num_clients=num_clients_2,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy_2,
    )
    print("Simulation 2 Completed\n")

    print("Starting Simulation 3 with 6 Clients")
    hist_3 = fl.simulation.start_simulation(
        client_fn=client_fn_3,
        num_clients=num_clients_3,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy_3,
    )
    print("Simulation 3 Completed\n")

    # Define the number of rounds
    rounds_1 = range(1, len(strategy_1.train_loss_history) + 1)
    rounds_2 = range(1, len(strategy_2.train_loss_history) + 1)
    rounds_3 = range(1, len(strategy_3.train_loss_history) + 1)

    # Plot Training Loss
    plt.figure(figsize=(10, 5))
    plt.plot(rounds_1, strategy_1.train_loss_history, marker='o', color='blue', label='3 Clients')
    plt.plot(rounds_2, strategy_2.train_loss_history, marker='o', color='gold', label='5 Clients')
    plt.plot(rounds_3, strategy_3.train_loss_history, marker='o', color='crimson', label='6 Clients')
    plt.title('Training Loss over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Training Loss (MSE)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'results/{details}training_loss_over_rounds.png')
    plt.show()

    # Plot Validation Loss
    plt.figure(figsize=(10, 5))
    plt.plot(rounds_1, strategy_1.val_loss_history, marker='o', color='blue', label='3 Clients')
    plt.plot(rounds_2, strategy_2.val_loss_history, marker='o', color='gold', label='5 Clients')
    plt.plot(rounds_3, strategy_3.val_loss_history, marker='o', color='crimson', label='6 Clients')
    plt.title('Validation Loss over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Validation Loss (MSE)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'results/{details}validation_loss_over_rounds.png')
    plt.show()

    # Plot Training MAE
    plt.figure(figsize=(10, 5))
    plt.plot(rounds_1, strategy_1.train_mae_history, marker='o', color='orange', label='3 Clients')
    plt.plot(rounds_2, strategy_2.train_mae_history, marker='o', color='lime', label='5 Clients')
    plt.plot(rounds_3, strategy_3.train_mae_history, marker='o', color='royalblue', label='6 Clients')
    plt.title('Training MAE over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Training Mean Absolute Error')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'results/{details}training_mae_over_rounds.png')
    plt.show()

    # Plot Validation MAE
    plt.figure(figsize=(10, 5))
    plt.plot(rounds_1, strategy_1.val_mae_history, marker='o', color='orange', label='3 Clients')
    plt.plot(rounds_2, strategy_2.val_mae_history, marker='o', color='lime', label='5 Clients')
    plt.plot(rounds_3, strategy_3.val_mae_history, marker='o', color='royalblue', label='6 Clients')
    plt.title('Validation MAE over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Validation Mean Absolute Error')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'results/{details}validation_mae_over_rounds.png')
    plt.show()

    # Plot Training R² Score
    plt.figure(figsize=(10, 5))
    plt.plot(rounds_1, strategy_1.train_r2_history, marker='o', color='green', label='3 Clients')
    plt.plot(rounds_2, strategy_2.train_r2_history, marker='o', color='peru', label='5 Clients')
    plt.plot(rounds_3, strategy_3.train_r2_history, marker='o', color='gray', label='6 Clients')
    plt.title('Training R² Score over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Training R² Score')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'results/{details}training_r2_over_rounds.png')
    plt.show()

    # Plot Validation R² Score
    plt.figure(figsize=(10, 5))
    plt.plot(rounds_1, strategy_1.val_r2_history, marker='o', color='green', label='3 Clients')
    plt.plot(rounds_2, strategy_2.val_r2_history, marker='o', color='peru', label='5 Clients')
    plt.plot(rounds_3, strategy_3.val_r2_history, marker='o', color='gray', label='6 Clients')
    plt.title('Validation R² Score over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Validation R² Score')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'results/{details}validation_r2_over_rounds.png')
    plt.show()

    # Plot Training Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(rounds_1, strategy_1.train_accuracy_history, marker='o', color='magenta', label='3 Clients')
    plt.plot(rounds_2, strategy_2.train_accuracy_history, marker='o', color='pink', label='5 Clients')
    plt.plot(rounds_3, strategy_3.train_accuracy_history, marker='o', color='purple', label='6 Clients')
    plt.title('Training Accuracy over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Training Accuracy (Proportion within ±10%)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'results/{details}training_accuracy_over_rounds.png')
    plt.show()

    # Plot Validation Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(rounds_1, strategy_1.val_accuracy_history, marker='o', color='magenta', label='3 Clients')
    plt.plot(rounds_2, strategy_2.val_accuracy_history, marker='o', color='pink', label='5 Clients')
    plt.plot(rounds_3, strategy_3.val_accuracy_history, marker='o', color='purple', label='6 Clients')
    plt.title('Validation Accuracy over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Validation Accuracy (Proportion within ±10%)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'results/{details}validation_accuracy_over_rounds.png')
    plt.show()