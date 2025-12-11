import pandas as pd
import numpy as np
import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
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
        self.fc = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.sigmoid(x)


class VFLServer():
    def __init__(self, data):
        self.model = ServerModel(12)
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
            grads = embedding_server.grad.split([4, 4, 4], dim=1)
            np_gradients = [serialise_array(grad.numpy()) for grad in grads]
        except Exception as e:
            logger.info(f"Converting the gradients failed: {e}")

        with torch.no_grad():
            correct = 0
            output = self.model(embedding_server)
            predicted = (output > 0.5).float()

            correct += (predicted == self.labels).sum().item()

            accuracy = correct / len(self.labels) * 100

        data = Struct()
        data.update({"accuracy": accuracy, "gradients": np_gradients})

        logger.info(f"Accuracy achieved: {accuracy}")

        return data


def handleAggregateRequest(msComm):
    global ms_config
    global vfl_server

    request = rabbitTypes.Request()
    msComm.original_request.Unpack(request)

    try:
        data = request.data["embeddings"]
        clients_embeddings = [deserialise_array(
            embeddings.string_value) for embeddings in data.list_value.values]
    except Exception as e:
        logger.error(f"Errored when deserialising client data: {e}")

    # TODO: Fetch model from PVC if not loaded yet
    # try:
    #     data = request.data["model_state"].string_value
    #     clients_model_state += [data]
    # except Exception as e:
    #     logger.error(
    #         f"Errored when deserialising client model state: {e}")

    # Hardcoded the number of clients for now
    data = vfl_server.aggregate_fit(clients_embeddings)

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
