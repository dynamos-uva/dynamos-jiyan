import pandas as pd
import numpy as np
import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from google.protobuf.struct_pb2 import Struct
from dynamos.ms_init import NewConfiguration
from dynamos.signal_flow import signal_continuation, signal_wait
from dynamos.logger import InitLogger
import rabbitMQ_pb2 as rabbitTypes

from google.protobuf.empty_pb2 import Empty
import microserviceCommunication_pb2 as msCommTypes
import threading
from opentelemetry.context.context import Context

np.set_printoptions(threshold=sys.maxsize)

# --- DYNAMOS Interface code At the TOP ---------------------------
if os.getenv('ENV') == 'PROD':
    import config_prod as config
else:
    import config_local as config

logger = InitLogger()
# tracer = InitTracer(config.service_name, config.tracing_host)

# Events to start the shutdown of this Microservice, can be used to call 'signal_shutdown'
stop_event = threading.Event()
stop_microservice_condition = threading.Condition()

# Events to make sure all services have started before starting to process a message
# Might be overkill, but good practice
wait_for_setup_event = threading.Event()
wait_for_setup_condition = threading.Condition()

ms_config = None
vfl_server = None

# --- END DYNAMOS Interface code At the TOP ----------------------

# ---- LOCAL TEST SETUP OPTIONAL!

# Go into local test code with flag '-t'
# parser = argparse.ArgumentParser()
# parser.add_argument("-t", "--test", action='store_true')
# args = parser.parse_args()
# test = args.test

# --------------------------------


DEFAULT_LEARNING_RATE = 0.1 
DEFAULT_NOF_CLIENTS = 3  # TODO: make it dynamic 
SERVER_CHECKPOINT_PATH = "server_checkpoint.pth"

def load_data(file_path):
    DATA_STEWARD_NAME = os.getenv("DATA_STEWARD_NAME").lower()

    file_name = f"{file_path}/outcomeData.csv"

    if DATA_STEWARD_NAME == "":
        logger.error("DATA_STEWARD_NAME not set.")
        file_name = f"{file_path}Data.csv"

    try:
        data = pd.read_csv(file_name, delimiter=',')
        logger.debug("after read csv")
    except FileNotFoundError:
        logger.error(f"CSV file for table {file_name} not found.")
        return None

    return data


def serialise_array(array):
    return json.dumps([
        str(array.dtype),
        array.tobytes().decode("latin1"),
        array.shape])


def deserialise_array(string, hook=None):
    encoded_data = json.loads(string, object_pairs_hook=hook)
    dataType = np.dtype(encoded_data[0])
    dataArray = np.frombuffer(encoded_data[1].encode("latin1"), dataType)

    if len(encoded_data) > 2:
        return dataArray.reshape(encoded_data[2])

    return dataArray


class ServerModel(nn.Module):
    def __init__(self, input_size):
        super(ServerModel, self).__init__()
        hidden_size = 16 # A small hidden layer 

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        # Optional dropout 
        # self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # Optional dropout
        # x = self.dropout(x)
        x = self.fc2(x)
        return x


class VFLServer():
    def __init__(self, data):
        self.intermediate_neurons = 4  # Assuming each client outputs 4 features
        self.nof_clients = DEFAULT_NOF_CLIENTS
        self.model = ServerModel(self.intermediate_neurons * self.nof_clients) 
        # self.initial_parameters = ndarrays_to_parameters(
        #     [val.cpu().numpy()
        #      for _, val in server_configuration.model.state_dict().items()]
        # )
        self.optimizer = optim.Adam(self.model.parameters(), lr=DEFAULT_LEARNING_RATE, weight_decay=1e-4)  # optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
        self.labels = torch.tensor(
            data["REL_TOTALBTU"].values).float().unsqueeze(1)

    def shrink_server_model(self, new_nof_clients, backtrack):
        """
        Creates a new ServerModel with fewer input neurons and copies over the trained weights
        from the old model for the first new_input_size neurons.
        """
        if backtrack and new_nof_clients==2:  # for now hardcoded to work only when reducing size from 3 to 2 clients
            # save model state to file
            logger.info("Saving server state before shrinking...")
            self.save_state(SERVER_CHECKPOINT_PATH)
        self.nof_clients = new_nof_clients
        # Create the new model
        # note: this is a completely new model with random weights
        self.model = ServerModel(self.intermediate_neurons * new_nof_clients)
        self.optimizer = optim.Adam(self.model.parameters(), lr=DEFAULT_LEARNING_RATE, weight_decay=1e-4)  # optim.SGD(self.model.parameters(), lr=0.01)

    
    def expand_server_model(self, new_nof_clients, backtrack):
        """
        Creates a new ServerModel with fewer input neurons and copies over the trained weights
        from the old model for the first new_input_size neurons.
        """
        self.nof_clients = new_nof_clients
        if backtrack and self.nof_clients==3:  # for now hardcoded to work only for 3 clients
            # save model state to file
            self.model = ServerModel(self.nof_clients * self.intermediate_neurons)
            self.optimizer = optim.Adam(self.model.parameters(), lr=DEFAULT_LEARNING_RATE, weight_decay=1e-4)  # optim.SGD(self.model.parameters(), lr=0.01)
            logger.info("Loading previous server state...")
            self.load_state(SERVER_CHECKPOINT_PATH)
        else:
            # Create the new model
            # note: this is a completely new model with random weights
            self.model = ServerModel(self.nof_clients * self.intermediate_neurons)
            self.optimizer = optim.Adam(self.model.parameters(), lr=DEFAULT_LEARNING_RATE, weight_decay=1e-4)  # optim.SGD(self.model.parameters(), lr=0.01)
    
    def update_server_model_architecture(self, old_nof_clients, new_nof_clients, backtrack):
        if new_nof_clients == old_nof_clients:
            # No change needed
            logger.debug("Number of clients unchanged, no model architecture update needed.")
        
        if new_nof_clients < old_nof_clients:
            logger.info(f"Number of clients decreased from {old_nof_clients} to {new_nof_clients}, shrinking model.")
            self.shrink_server_model(new_nof_clients, backtrack)
        
        if new_nof_clients > old_nof_clients:
            logger.info(f"Number of clients increased from {old_nof_clients} to {new_nof_clients}, expanding model.")
            self.expand_server_model(new_nof_clients, backtrack)


    def aggregate_fit(self, results,backtrack=False):
        global server_configuration

        # infer the number of clients based on the data received
        new_nof_clients = len(results)
        if new_nof_clients != self.nof_clients:
            print(f"Number of clients in results: {new_nof_clients}")
            print(f"Current number of clients: {self.nof_clients}")
            logger.info(f"Number of clients {new_nof_clients} does not match expected {self.nof_clients}, updating server architecture...")
            # TODO: update the architecture of the model
            self.update_server_model_architecture(self.nof_clients, new_nof_clients, backtrack)


        try:
            embedding_results = [
                torch.from_numpy(embedding.copy())
                for embedding in results
            ]
        except Exception as e:
            logger.info(f"Converting the results to torch failed: {e}")

        try:
            embeddings_aggregated = torch.cat(embedding_results, dim=1)
            embedding_server = embeddings_aggregated.detach().requires_grad_()
            output = self.model(embedding_server)
            loss = self.criterion(output, self.labels)
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()
        except Exception as e:
            logger.info(f"Running gradient descent failed: {e}")

        try:
            grads = embedding_server.grad.split([4]*self.nof_clients, dim=1)
            np_gradients = [serialise_array(grad.numpy()) for grad in grads]
        except Exception as e:
            logger.info(f"Converting the gradients failed: {e}")

        with torch.no_grad():
            output = self.model(embedding_server)

            mse = nn.MSELoss()(output, self.labels).item()
            rmse = torch.sqrt(torch.tensor(mse)).item()
            mae = nn.L1Loss()(output, self.labels).item()
            total_sum_of_squares = torch.sum((self.labels - self.labels.mean()) ** 2)
            residual_sum_of_squares = torch.sum((self.labels - output) ** 2)
            r2 = 1 - (residual_sum_of_squares / total_sum_of_squares)
            r2_score = r2.item()

            metrics = {
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "r2": r2_score
            }
            # Example of printing the metrics
            # print(f"Regression Metrics - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2_score:.4f}")
            pass 


        data = Struct()
        data.update({"accuracy": r2_score, "gradients": np_gradients})  # TODO: maybe try to rename the field
        # data = []
        # data.append({"r2": r2_score, "gradients": np_gradients})

        logger.info(f"R2 achieved: {r2_score}")

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


def handleAggregateRequest(msComm):
    global ms_config
    global vfl_server

    request = rabbitTypes.Request()
    msComm.original_request.Unpack(request)

    backtrack = False
    try:
        training_backtrack_flag = request.data["trainingBacktrack"]
        logger.debug(f"Training backtrack flag: {training_backtrack_flag}")
        logger.debug(f"Training backtrack flag: {type(training_backtrack_flag)}")
        if training_backtrack_flag.number_value == 1:
            backtrack = True
            logger.debug(f"Training backtrack flag is 'True'")
    except Exception as e:
        logger.warning(f"Error when retrieving training backtrack flag: {e}")

    try:
        data = request.data["embeddings"]
        # logger.debug(f"Received data: {data}")
        # logger.debug(f"Embedding len: {len(data)}")
        clients_embeddings = [deserialise_array(
            embeddings.string_value) for embeddings in data.list_value.values]
    except Exception as e:
        logger.error(f"Errored when deserialising client data: {e}")

    data = vfl_server.aggregate_fit(clients_embeddings, backtrack)

    ms_config.next_client.ms_comm.send_data(msComm, data, {})


# ---  DYNAMOS Interface code At the Bottom --------

def request_handler(msComm: msCommTypes.MicroserviceCommunication,
                    ctx: Context = None):
    global ms_config

    logger.info(f"Received original request type: {msComm.request_type}")

    # Ensure all connections have finished setting up before processing data
    signal_wait(wait_for_setup_event, wait_for_setup_condition)

    try:
        request = rabbitTypes.Request()
        msComm.original_request.Unpack(request)
    except Exception as e:
        logger.error(f"Unexpected original request received: {e}")
        ms_config.next_client.ms_comm.send_data(msComm, msComm.data, {})
        return Empty()

    DATA_STEWARD_NAME = os.getenv("DATA_STEWARD_NAME").lower()

    if DATA_STEWARD_NAME != "server":
        if request.type == "vflShutdownRequest":
            logger.info(
                "Received vflShutdownRequest, shutting down service.")
            ms_config.next_client.ms_comm.send_data(msComm, msComm.data, {})
            signal_continuation(stop_event, stop_microservice_condition)
        else:
            logger.info("This is the server (not client), relaying request.")
            ms_config.next_client.ms_comm.send_data(msComm, msComm.data, {})

    else:
        if request.type == "vflAggregateRequest":
            logger.info("Received a vflAggregateRequest.")
            handleAggregateRequest(msComm)

        elif request.type == "vflPingRequest":
            logger.info("Received a vflPingRequest.")
            ms_config.next_client.ms_comm.send_data(msComm, msComm.data, {})

        elif request.type == "vflShutdownRequest":
            logger.info("Received a vflShutdownRequest.")
            signal_continuation(stop_event, stop_microservice_condition)

        return Empty()


def main():
    global config
    global ms_config
    global vfl_server

    data = load_data(config.dataset_filepath)
    vfl_server = VFLServer(data)

    ms_config = NewConfiguration(
        config.service_name, config.grpc_addr, request_handler)

    # Signal the message handler that all connections have been created
    signal_continuation(wait_for_setup_event, wait_for_setup_condition)

    # Wait for the end of processing to shutdown this Microservice
    try:
        signal_wait(stop_event, stop_microservice_condition)

    except KeyboardInterrupt:
        logger.debug("KeyboardInterrupt received, stopping server...")
        signal_continuation(stop_event, stop_microservice_condition)

    ms_config.stop(2)
    logger.debug(f"Exiting {config.service_name}")
    sys.exit(0)

# ---  END DYNAMOS Interface code At the Bottom -----------------


if __name__ == "__main__":
    main()
