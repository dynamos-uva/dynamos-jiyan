import pandas as pd
import os
import json
import sys
import threading
import socket
import time
import hashlib
import numpy as np

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
ANON_DIR = "/shared"
ANON_LOCK = threading.Lock()

# Anonymizes first 1 column, converts hash into float, necessary for later type checks.
def apply_anonymization(df, ds_name):
    enable_anon = True
    df_anon = df.copy()
    salt = "salt_aaa_ooo_iii"
    cols = list(df_anon.columns[:1])
    logger.info(f"Anonymizing columns: {cols}")

    def hash_to_float(val, col):
        val_str = str(val)
        msg = f"{salt}|{ds_name}|{col}|{val_str}".encode("utf-8")
        digest = hashlib.sha256(msg).digest()
        u64 = int.from_bytes(digest[:8], byteorder="big", signed=False)
        return np.float32(u64 / 2**64)
    
    if enable_anon == True:
        for c in cols:
            df_anon[c] = df_anon[c].map(lambda v, _c=c: hash_to_float(v, _c))
        return df_anon
    else:
        return df


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

# Saves anonymized dataset to correct directory
def save_anonymized_dataset(df, ds_name):
    out_csv = os.path.join(ANON_DIR, f"{ds_name}Data_anonymized.csv")

    tmp = out_csv + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, out_csv)

    return out_csv

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

    # Grabs temp pod name
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

    logger.info(f"[{ds_name}] dataset loaded")

    anon_path = os.path.join(ANON_DIR, f"{ds_name}Data_anonymized.csv")


    # Checks if anon file exists and can be reused, otherwise regenerates and logs
    with ANON_LOCK:
        if os.path.exists(anon_path) and os.path.getsize(anon_path) > 0:
            logger.info(f"Reusing anonymized dataset: {anon_path}")
        else:
            df_anon = apply_anonymization(df, ds_name)
            anon_path = save_anonymized_dataset(df_anon, ds_name)
            logger.info(f"Created anonymized dataset: {anon_path}")

    # Builds metadata payload as python dict
    df_meta = {
        "columns": df.columns.tolist(),
        "dtypes": {k: str(v) for k, v in df.dtypes.apply(lambda x: x.name).to_dict().items()},
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "source": "anonymize-test",
        "anon_dir": ANON_DIR,
        "anon_path": anon_path}

    logger.info(f"[{ds_name}] anonymized ready: {anon_path} (rows={df.shape[0]}, cols={df.shape[1]})")

    md = dict(msComm.metadata)
    md = register_service_on_metadata(md, service_name=config.service_name)
    md["dataframe_metadata"] = json.dumps(df_meta)
    md["anon_path"] = anon_path

    ms_config.next_client.ms_comm.send_data(msComm, Struct(), md)
    return Empty()

def main():
    global ms_config

    sidecar_port = int(os.getenv("SIDECAR_PORT", "50051"))
    logger.info(f"Waiting for sidecar {sidecar_port}")
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
