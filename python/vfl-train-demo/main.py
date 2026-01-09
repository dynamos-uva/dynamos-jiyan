import pandas as pd
import numpy as np
import sys
import os
import io
import json
import torch
import time
import socket
import torch.nn as nn
import torch.nn.functional as F
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

DEFAULT_LEARNING_RATE = 0.1 
DEFAULT_WEIGHT_DECAY = 1e-4

# --- END DYNAMOS Interface code At the TOP ----------------------

# ---- LOCAL TEST SETUP OPTIONAL!

# Go into local test code with flag '-t'
# parser = argparse.ArgumentParser()
# parser.add_argument("-t", "--test", action='store_true')
# args = parser.parse_args()
# test = args.test

# --------------------------------

# Took this function from other microservice.
def protobuf_to_dataframe(data_struct: Struct, metadata: dict = None) -> pd.DataFrame:
    if metadata is None:
        metadata = {}
    dtypes = metadata if isinstance(metadata, dict) else {}

    data = {}
    for key, value in data_struct.fields.items():
        value_list = value.list_value.values
        items = [v.string_value for v in value_list]

        dtype = str(dtypes.get(key, "object")).lower()

        if dtype.startswith("int"):
            data[key] = [int(float(x)) if x != "" else 0 for x in items]
        elif dtype.startswith("float"):
            data[key] = [float(x) if x != "" else 0.0 for x in items]
        elif dtype in ("bool", "boolean"):
            data[key] = [str(x).strip().lower() in ("true", "1", "t", "yes") for x in items]
        else:
            data[key] = items

    return pd.DataFrame(data)

# Was necessary in all microservices I used since services would try to connect to sidecar instantly, resulting in crash
def wait_for_port(host, port, wait_time):
    deadline = time.time() + wait_time
    last_err = None
    # While deadline has not passed keeps trying to connect to sidecar.
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError as e:
            last_err = e
            time.sleep(1)
    # If time is up raises error
    raise RuntimeError(f"Timed out waiting for {host}:{port} to open. Last error: {last_err}")


class ClientModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # Layer 1: Input features -> Hidden layer (e.g., 64 neurons)
        self.fc1 = nn.Linear(input_size, 64)
        # Layer 2: Hidden layer -> Output embedding (8 neurons)
        self.fc2 = nn.Linear(64, 4)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


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
    def __init__(self, data, learning_rate=DEFAULT_LEARNING_RATE, model_state=None, optimiser_state=None):
        self.data = torch.tensor(data.to_numpy(dtype=np.float32), dtype=torch.float32)
        
        self.model = ClientModel(data.shape[1])
        if model_state is not None:
            self.model.load_state_dict(model_state)

        self.optimiser = None

    def create_optimiser(self, learning_rate):
        if self.optimiser is None:
            self.optimiser = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=DEFAULT_WEIGHT_DECAY)   #  torch.optim.SGD(self.model.parameters(), lr=learning_rate)

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
                global vfl_client
                # Decode dataframe from msComm using metadata that was sent through
                try:
                    md = dict(msComm.metadata)
                    df_meta_json = md.get("dataframe_metadata", "")
                    if not df_meta_json:
                        raise ValueError("Missing metadata. Anonymize issue likely")
                    df_meta = json.loads(df_meta_json)

                    df = protobuf_to_dataframe(msComm.data, df_meta)
                    # Necessary for issues with test data
                    if "Unnamed: 0" in df.columns:
                        df = df.drop(columns=["Unnamed: 0"])
                    for col in df.columns:
                        if df[col].dtype == "object":
                            lowered = df[col].astype(str).str.strip().str.lower()
                            if lowered.isin(["true", "false"]).all():
                                df[col] = lowered == "true"
                            else:
                                df[col] = pd.to_numeric(df[col], errors="coerce")
                    df = df.fillna(0).astype(np.float32)
                except Exception as e:
                    logger.error(f"Failed to decode dataframe from msComm.data: {e}")
                    data = Struct()
                    ms_config.next_client.ms_comm.send_data(msComm, data, dict(msComm.metadata))
                    return Empty()
                try:
                    # If round 1 create client and continue, otherwise compare to check for mismatch
                    # Had mismatch issues earlier, just as a safeguard
                    if vfl_client is None:
                        logger.info("Initializing VFLClient from incoming dataframe.")
                        vfl_client = VFLClient(df)
                    else:
                        if df.shape[1] != vfl_client.model.fc1.in_features:
                            logger.warning(f"Feature count changed {vfl_client.model.fc1.in_features} -> {df.shape[1]}. Reinitializing client model.")
                            vfl_client = VFLClient(df)
                        else:
                            vfl_client.data = torch.tensor(df.to_numpy(dtype=np.float32), dtype=torch.float32)
                except Exception as e:
                    logger.error(f"Failed initializing/updating VFLClient: {e}")
                    data = Struct()
                    ms_config.next_client.ms_comm.send_data(msComm, data, dict(msComm.metadata))
                    return Empty()
                try:
                    embeddings = vfl_client.train_model()
                    logger.debug(f"size of serialized array in bytes: {sys.getsizeof(embeddings)}")
                    data = Struct()
                    data.update({"embeddings": embeddings})
                except Exception as e:
                    logger.error(f"Unexpected error during train_model: {e}")
                    data = Struct()

                ms_config.next_client.ms_comm.send_data(msComm, data, dict(msComm.metadata))
                return Empty()
            elif request.type == "vflGradientDescentRequest":
                if vfl_client is None:
                    logger.error("Received vflGradientDescentRequest before vflTrainRequest initialized vfl_client")
                    data = Struct()
                    ms_config.next_client.ms_comm.send_data(msComm, data, {})
                    return Empty()
                try:
                    learning_rate = request.data["learning_rate"].number_value
                    vfl_client.create_optimiser(learning_rate)
                except Exception:
                    vfl_client.create_optimiser(DEFAULT_LEARNING_RATE)

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

    vfl_client = None
    sidecar_port = int(os.getenv("SIDECAR_PORT", "50051"))
    logger.info(f"Waiting for sidecar gRPC on localhost:{sidecar_port} ...")
    wait_for_port("127.0.0.1", sidecar_port, wait_time=90)

    port = int(os.getenv("DESIGNATED_GRPC_PORT", "0"))
    last = int(os.getenv("LAST", "0"))
    if last == 0 and port > 0:
        next_port = port + 1
        logger.info(f"Waiting for next gRPC service on localhost:{next_port} ...")
        wait_for_port("127.0.0.1", next_port, wait_time=120)

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
