# simulation.py
import flwr as fl
import matplotlib.pyplot as plt

from client import FlowerClient
from strategy import Strategy
from task import get_partitions_and_label

partitions, labels = get_partitions_and_label()

def client_fn(cid):
    print(f"Creating client with cid: {cid}")
    # Map the cid to an index in the partitions list
    if cid in ["0", "1"]:
        cid_int = int(cid)
        data = partitions[cid_int]
        return FlowerClient(cid_int, data)
    else:
        # If cid is not "0" or "1", we do not create a client
        print(f"Client with cid {cid} does not exist.")
        return None

if __name__ == "__main__":
    # Create the strategy with appropriate parameters
    strategy = Strategy(
        labels=labels,
        fraction_fit=1.0,            # Sample 100% of available clients
        fraction_evaluate=0.0,       # Disable evaluation
        min_fit_clients=2,           # Minimum number of clients to be sampled for training
        min_available_clients=2,     # Minimum number of clients that need to be connected
        accept_failures=True         # Accept failures to handle clients that are not created
    )

    # Start the simulation
    hist = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=2,                 # Total number of clients
        config=fl.server.ServerConfig(num_rounds=20),
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
    plt.savefig('loss_over_rounds.png')
    plt.show()
