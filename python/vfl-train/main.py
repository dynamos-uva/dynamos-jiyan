import pandas as pd
import numpy as np
import sys
import os
import io
import json
import torch
import torch.nn as nn
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler
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

    file_name = f"{file_path}/{DATA_STEWARD_NAME}Data.csv"

    if DATA_STEWARD_NAME == "":
        logger.error("DATA_STEWARD_NAME not set.")
        file_name = f"{file_path}Data.csv"

    try:
        data = pd.read_csv(file_name, delimiter=',')
    except FileNotFoundError:
        logger.error(f"CSV file for table {file_name} not found.")
        return None

    return data


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
    logger.info(string, encoded_data)
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
            logger.error("Optimiser is not defined.")

        try:
            self.model.zero_grad()
            # embedding = self.model(self.data)
            self.embedding.backward(torch.from_numpy(gradients))
            self.optimiser.step()
        except Exception as e:
            logger.error(f"Error occurred: {e}")


# # Note: Gradients sent by server are for this client only to preserve privacy
# def vfl_train(learning_rate, model_state, gradients):
#
#     optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)
#
#     if gradients is not None:
#         vfl_evaluate(data, model, optimiser, gradients)
#
#     embeddings = train_model(data, model)
#     model_state = model.state_dict()
#
#     buffer = io.BytesIO()
#     torch.save(model_state, buffer)
#
#     data = Struct()
#     data.update({"embeddings": serialise_array(embeddings),
#                  "model_state": buffer.getvalue().decode("latin1")})
#
#     return data


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

    if DATA_STEWARD_NAME == "server":
        if request.type == "vflShutdownRequest":
            logger.info(
                "Received vflShutdownRequest, shutting down service.")
            ms_config.next_client.ms_comm.send_data(msComm, msComm.data, {})
            signal_continuation(stop_event, stop_microservice_condition)
        else:
            logger.info("This is the server (not client), relaying request.")
            ms_config.next_client.ms_comm.send_data(msComm, msComm.data, {})
    else:
        if request is not None:
            if request.type == "vflTrainRequest":
                logger.info("Received a vflTrainRequest.")

                try:
                    embeddings = vfl_client.train_model()
                    data = Struct()
                    data.update({"embeddings":  embeddings})
                except Exception as e:
                    logger.error(f"Unexpected error: {e}")
                    data = Struct()

                ms_config.next_client.ms_comm.send_data(msComm, data, {})
            elif request.type == "vflGradientDescentRequest":
                try:
                    learning_rate = request.data["learning_rate"].number_value
                    vfl_client.create_optimiser(learning_rate)
                except Exception:
                    vfl_client.create_optimiser(0.05)

                try:
                    gradients = request.data["gradients"].string_value
                    gradients = deserialise_array(gradients)
                except Exception as e:
                    logger.error(f"Gradients did not get parsed properly: {e}")
                    logger.info(msComm.data)
                    gradients = None

                try:
                    vfl_client.gradient_descent(gradients)
                except Exception as e:
                    logger.error(f"Unexpected error: {e}")

                try:
                    data = Struct()
                except Exception as e:
                    logger.error(f"Unexpected error: {e}")

                ms_config.next_client.ms_comm.send_data(msComm, data, {})

            elif request.type == "vflShutdownRequest":
                logger.info(
                    "Received vflShutdownRequest, shutting down service.")
                signal_continuation(stop_event, stop_microservice_condition)

            elif request.type == "vflPingRequest":
                logger.info("Received a vflPingRequest.")
                ms_config.next_client.ms_comm.send_data(
                    msComm, msComm.data, {})

            else:
                logger.error(f"An unknown request_type: {msComm.data.type}")

            return Empty()


def main():
    global config
    global ms_config
    global vfl_client

    try:
        data = load_data(config.dataset_filepath)
        vfl_client = VFLClient(data)
    except Exception as e:
        logger.error(f"Error occurred: {e}")

    ms_config = NewConfiguration(
        config.service_name, config.grpc_addr, request_handler)

    signal_continuation(wait_for_setup_event, wait_for_setup_condition)

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
