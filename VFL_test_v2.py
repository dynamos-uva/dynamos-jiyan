import pandas as pd
import numpy as np
import sys
import os
import io
import json
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler


# ---------------------- SET SEED FOR REPRODUCIBILITY ----------------------
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
# --------------------------------------------------------------------------


server_data = pd.read_csv(r"C:\Users\alkou\Documents\GitHub\Scattered-Directive\python\vfl-train-model\datasets\outcomeData.csv", delimiter=',', index_col=0)
print(server_data.head())

client1_data = pd.read_csv(r"C:\Users\alkou\Documents\GitHub\Scattered-Directive\python\vfl-train\datasets\clientoneData.csv", delimiter=',', index_col=0)
client2_data = pd.read_csv(r"C:\Users\alkou\Documents\GitHub\Scattered-Directive\python\vfl-train\datasets\clienttwoData.csv", delimiter=',', index_col=0)
client3_data = pd.read_csv(r"C:\Users\alkou\Documents\GitHub\Scattered-Directive\python\vfl-train\datasets\clientthreeData.csv", delimiter=',', index_col=0)
print(client1_data.head())
print(client2_data.head())
print(client3_data.head())

# I put the client 2 at the end so that it is excluded when NOF_CLIENTS=2. For demonstration purposes, because client 3 din't have a big effect.
client_datasets = [client1_data, client3_data, client2_data]  

NOF_CLIENTS = 3
REMOVE_CLIENT_ROUND = -1  # remove one client after these rounds
SHRINK_SERVER = True  # if True, reinstantiate the server when a client is removed. Otherwise, keep the neurons the same, just fewer. Truncate the last neurons.

ADD_CLIENT_ROUND = -1  # add one client after these rounds
ADD_CLIENT_CLEAN = False  # if True, reinstantiate the added client. Otherwise, just keep the client the way it is. It might be pretrained already.

TOTAL_ROUNDS = 220

SERVER_CHECKPOINT_PATH = "server_state.pth"

# # Dummy data loading (replace with your CSV)
# data = pd.read_csv("your_data.csv")  # should contain features for all clients + 'Survived' label

# # Simulate 3 clients, each with 4 features
# client_features = [
#     data.iloc[:, 0:4].values,   # Client 1: columns 0-3
#     data.iloc[:, 4:8].values,   # Client 2: columns 4-7
#     data.iloc[:, 8:12].values   # Client 3: columns 8-11
# ]
# labels = data["Survived"].values

np.set_printoptions(threshold=sys.maxsize)  # To print full numpy arrays

# note: to revert in production


class ClientModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 4)

    def forward(self, x):
        return self.fc(x)


def serialise_array(array):
    return json.dumps([
        str(array.dtype),
        array.tobytes().decode("latin1"),
        array.shape])


def deserialise_array(string, hook=None):
    encoded_data = json.loads(string, object_pairs_hook=hook)
    # logger.info(string, encoded_data)
    dataType = np.dtype(encoded_data[0])
    dataArray = np.frombuffer(encoded_data[1].encode("latin1"), dataType)

    if len(encoded_data) > 2:
        return dataArray.reshape(encoded_data[2])

    return dataArray

class VFLClient():
    def __init__(self, data, learning_rate=0.01, model_state=None, optimiser_state=None):
        self.data = torch.tensor(StandardScaler().fit_transform(data)).float()
        self.model = ClientModel(data.shape[1])
        if model_state is not None:
            self.model.load_state_dict(model_state)

        self.optimiser = None

    def create_optimiser(self, learning_rate):
        if self.optimiser is None:
            self.optimiser = torch.optim.SGD(
                self.model.parameters(), lr=learning_rate)

    def train_model(self):
        self.embedding = self.model(self.data)
        return serialise_array(self.embedding.detach().numpy())

    def gradient_descent(self, gradients):
        if self.optimiser is None:
            print("Optimiser is not defined.")
            pass 

        try:
            self.model.zero_grad()
            # embedding = self.model(self.data)
            self.embedding.backward(torch.from_numpy(gradients))
            self.optimiser.step()
        except Exception as e:
            print(f"Error occurred: {e}")



def shrink_server_model(old_model, new_input_size):
    """
    Creates a new ServerModel with fewer input neurons and copies over the trained weights
    from the old model for the first new_input_size neurons.
    """
    # Create the new model
    new_model = ServerModel(new_input_size)
    # Copy weights: old_model.fc.weight shape: [1, old_input_size]
    with torch.no_grad():
        # Take only first new_input_size columns (neurons)
        new_model.fc.weight[:, :] = old_model.fc.weight[:, :new_input_size]
        new_model.fc.bias[:] = old_model.fc.bias[:]
    return new_model


class ServerModel(nn.Module):
    def __init__(self, input_size):
        super(ServerModel, self).__init__()
        self.fc = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        return self.sigmoid(x)


class VFLServer():
    def __init__(self, data):
        self.model = ServerModel(4 * NOF_CLIENTS)  # Assuming each client outputs 4 features
        # self.initial_parameters = ndarrays_to_parameters(
        #     [val.cpu().numpy()
        #      for _, val in server_configuration.model.state_dict().items()]
        # )
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.BCELoss()
        self.labels = torch.tensor(
            data["Survived"].values).float().unsqueeze(1)

    def aggregate_fit(self, results):
        global server_configuration

        try:
            embedding_results = [
                torch.from_numpy(embedding.copy())
                for embedding in results
            ]
        except Exception as e:
            print(f"Converting the results to torch failed: {e}")

        try:
            embeddings_aggregated = torch.cat(embedding_results, dim=1)
            embedding_server = embeddings_aggregated.detach().requires_grad_()
            output = self.model(embedding_server)
            loss = self.criterion(output, self.labels)
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()
        except Exception as e:
            print(f"Running gradient descent failed: {e}")

        try:
            grads = embedding_server.grad.split([4]*NOF_CLIENTS, dim=1)
            np_gradients = [serialise_array(grad.numpy()) for grad in grads]
        except Exception as e:
            print(f"Converting the gradients failed: {e}")

        with torch.no_grad():
            correct = 0
            output = self.model(embedding_server)
            predicted = (output > 0.5).float()

            correct += (predicted == self.labels).sum().item()

            accuracy = correct / len(self.labels) * 100

        data = []
        data.append({"accuracy": accuracy, "gradients": np_gradients})

        print(f"Accuracy achieved: {accuracy}")

        return data
    
    def save_state(self, filepath):
        """Save the state dicts for both model and optimizer to disk."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filepath)
        print(f"Server state saved to {filepath}")

    def load_state(self, filepath):
        """Load the state dicts for both model and optimizer from disk."""
        state = torch.load(filepath)
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        print(f"Server state loaded from {filepath}")




# define clients
all_clients = []
for client_data in client_datasets[:NOF_CLIENTS]:
    client = VFLClient(client_data)
    all_clients.append(client)

clients = all_clients[:NOF_CLIENTS] # the active clients

vfl_server = VFLServer(server_data)


train_results = []

# Training loop
for round in range(TOTAL_ROUNDS):

    print("--------------------------------------------------")
    print(f"Round {round+1}")
    if round == REMOVE_CLIENT_ROUND:
        if NOF_CLIENTS > 1:
            NOF_CLIENTS -= 1
            clients.pop()  # remove the last client
            print(f"Client removed. Number of clients is now {NOF_CLIENTS}.")
        else:
            print("Cannot remove more clients.")

        print("Saving server state before modification...")
        vfl_server.save_state(SERVER_CHECKPOINT_PATH)

        if SHRINK_SERVER:
            print(f"Shrinking server model for {NOF_CLIENTS}...")
            # Shrink the server model to match the new number of clients
            old_model = vfl_server.model
            # Each client outputs 4 features
            vfl_server.model = shrink_server_model(old_model, 4*NOF_CLIENTS)
        else:
            print(f"Reinstantiating server for {NOF_CLIENTS}...")
            vfl_server = VFLServer(server_data)
        
        

    if round == ADD_CLIENT_ROUND:
        if NOF_CLIENTS < 3:
            NOF_CLIENTS += 1
            
            if ADD_CLIENT_CLEAN:
                print(f"Reinstantiating client {NOF_CLIENTS-1}")
                all_clients[NOF_CLIENTS-1] = VFLClient(client_datasets[NOF_CLIENTS-1])  # reinstantiate the client

            clients.append(all_clients[NOF_CLIENTS-1])  # add the next client
            print(f"Client added. Number of clients is now {NOF_CLIENTS}.")
        else:
            print("Cannot add more clients.")

        if os.path.isfile(SERVER_CHECKPOINT_PATH):
            
            print("Loading server state before modification...")
            vfl_server = VFLServer(server_data)  # clean server with the correct input size
            vfl_server.load_state(SERVER_CHECKPOINT_PATH)
        else:
            print(f"Reinstantiating server for {NOF_CLIENTS}...")
            vfl_server = VFLServer(server_data)

    #  1. Clients compute embeddings

    results = []
    for client in clients:
        results.append(client.train_model())

    # deserialise_array(embeddings[2])
    deserialized_results = [deserialise_array(embedding) for embedding in results]

    # 2. Server aggregates embeddings and returns accuracy and gradients
    data = vfl_server.aggregate_fit(deserialized_results)

    # print(data[-1].keys())
    print(data[-1]['accuracy'])

    gradients = data[-1]['gradients']

    # 3. backpropagate gradients to clients

    for i, vfl_client in enumerate(clients):
        vfl_client.create_optimiser(0.05)

        vfl_client.gradient_descent(deserialise_array(gradients[i]))
    

    train_results.append({
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "train_round": round+1,
        "accuracy": data[-1]['accuracy'],
        "clients": NOF_CLIENTS
    })

payload = {
    "metadata": {
        "total_rounds": TOTAL_ROUNDS,
        "REMOVE_CLIENT_ROUND": REMOVE_CLIENT_ROUND,
        "SHRINK_SERVER": SHRINK_SERVER,
        "ADD_CLIENT_ROUND": ADD_CLIENT_ROUND,
        "ADD_CLIENT_CLEAN": ADD_CLIENT_CLEAN,
    },
    "results": train_results
}

filename = f"./run_dumps/vfl_test_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
print(f"Saving to file {filename}")
with open(filename, "w") as f:
    json.dump(payload, f, indent=2)