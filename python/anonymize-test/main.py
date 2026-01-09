import pandas as pd
import os
import json
import sys
import threading
import socket
import time

from dynamos.ms_init import NewConfiguration
from dynamos.signal_flow import signal_continuation, signal_wait
from dynamos.logger import InitLogger

import rabbitMQ_pb2 as rabbitTypes
import microserviceCommunication_pb2 as msCommTypes
from google.protobuf.struct_pb2 import Struct, Value, ListValue
from google.protobuf.empty_pb2 import Empty

if os.getenv("ENV") == "PROD":
    import config_prod as config
else:
    import config_local as config

logger = InitLogger()

stop_event = threading.Event()
stop_microservice_condition = threading.Condition()
wait_for_setup_event = threading.Event()
wait_for_setup_condition = threading.Condition()

ms_config = None

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


# i took this from one of the existing microservices
def dataframe_to_protobuf(df):
    # Convert the DataFrame to a dictionary of lists (one for each column)
    data_dict = df.to_dict(orient='list')

    # Convert the dictionary to a Struct
    data_struct = Struct()

    # Iterate over the dictionary and add each value to the Struct
    for key, values in data_dict.items():
        # Pack each item of the list into a Value object
        value_list = [Value(string_value=str(item)) for item in values]
        # Pack these Value objects into a ListValue
        list_value = ListValue(values=value_list)
        # Add the ListValue to the Struct
        data_struct.fields[key].CopyFrom(Value(list_value=list_value))

    # Create the metadata
    # Infer the data types of each column
    data_types = df.dtypes.apply(lambda x: x.name).to_dict()
    # Convert the data types to string values
    metadata = {k: str(v) for k, v in data_types.items()}

    return data_struct, metadata

# this function as well
def register_service_on_metadata(metadata:dict, service_name:str) -> dict:
    """
    Adds a JSON encoded list of the services that took place on the field "services".
    """
    if "services" in metadata:
        services = json.loads(metadata["services"])
        services.append(service_name)
        metadata["services"] = json.dumps(services)
        return metadata

    metadata["services"] = json.dumps([service_name])

    return metadata

# Loads whichever dataset is necessary based on name
# I.e. clientone loads clientoneData.csv
def load_local_dataset():
    ds = os.getenv("DATA_STEWARD_NAME", "").lower()
    file_name = os.path.join("/app", "data", f"{ds}Data.csv")
    logger.info(f"Client dataset: {file_name} loading...")
    # Reads and returns csv as a pandas dataframe
    return pd.read_csv(file_name, delimiter=",")

def request_handler(msComm: msCommTypes.MicroserviceCommunication, ctx=None):
    global ms_config

    signal_wait(wait_for_setup_event, wait_for_setup_condition)

    request = rabbitTypes.Request()
    try:
        msComm.original_request.Unpack(request)
    except Exception as e:
        logger.error(f"Failed to unpack request: {e}")
        ms_config.next_client.ms_comm.send_data(msComm, msComm.data, dict(msComm.metadata))
        return Empty()

    # If not one of theses requests, just forward
    if request.type not in ("vflTrainRequest", "vflTrainModelRequest"):
        ms_config.next_client.ms_comm.send_data(msComm, msComm.data, dict(msComm.metadata))
        return Empty()

    ds_name = os.getenv("DATA_STEWARD_NAME", "").lower()
    
    # If server, do nothing.
    # Otherwise load data and continue
    if ds_name == "server":
        ms_config.next_client.ms_comm.send_data(msComm, msComm.data, dict(msComm.metadata))
        return Empty()

    try:
        df = load_local_dataset()
    except Exception as e:
        logger.error(f"Failed loading dataset: {e}")
        ms_config.next_client.ms_comm.send_data(msComm, msComm.data, dict(msComm.metadata))
        return Empty()

    # Pretty much all this ms does for now, just for testing. Logs head of df.
    logger.info(f"[{ds_name}] dataset head(10):\n{df.head(10).to_string(index=False)}")

    data_pb, df_meta = dataframe_to_protobuf(df)

    md = dict(msComm.metadata)
    md = register_service_on_metadata(md, service_name=config.service_name)
    md["dataframe_metadata"] = json.dumps(df_meta)

    ms_config.next_client.ms_comm.send_data(msComm, data_pb, md)
    return Empty()


def main():
    global ms_config

    sidecar_port = int(os.getenv("SIDECAR_PORT", "50051"))
    logger.info(f"Waiting for sidecar gRPC on localhost:{sidecar_port} ...")
    wait_for_port("127.0.0.1", sidecar_port, wait_time=90)

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


if __name__ == "__main__":
    main()
