import pandas as pd
from pandasql import sqldf
import re
import time
import sys
import os
from google.protobuf.struct_pb2 import Struct, Value, ListValue
import json
import argparse
from dynamos.ms_init import NewConfiguration
from dynamos.signal_flow import signal_continuation, signal_wait
from dynamos.logger import InitLogger
from dynamos.tracer import InitTracer

from google.protobuf.empty_pb2 import Empty
import microserviceCommunication_pb2 as msCommTypes
import rabbitMQ_pb2 as rabbitTypes
import threading
import time
import sys
from opentelemetry.context.context import Context

from config_prod import service_name
import hashlib
import random

# from pathlib import Path
# __file__ is the path to the current script (main.py)
# service_name = Path(__file__).parent.name  # proxy for service name assuming

# --- DYNAMOS Interface code At the TOP ----------------------------------------------------
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

# --- END DYNAMOS Interface code At the TOP ----------------------------------------------------

#---- LOCAL TEST SETUP OPTIONAL!

# Go into local test code with flag '-t'
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--test", action='store_true')
args = parser.parse_args()
test = args.test

#--------------------------------



def anonymize_building_ids(df: pd.DataFrame, column: str = 'building_id', seed: int = None) -> pd.DataFrame:
    """
    Replace building_id values with a hashed anonymized version using a random seed.
    Returns: (anonymized_df, mapping, seed)
    If anonymization fails, returns (empty DataFrame, empty dict, None) and logs an error.
    """
    try:
        if seed is None:
            seed = random.randint(0, int(1e9))
        rng = random.Random(seed)
        salt = str(rng.random())

        def hash_id(x):
            # Combine salt with building_id, encode, and hash
            return hashlib.sha256(f"{salt}_{x}".encode()).hexdigest()

        # Map original ids to anonymized ids
        unique_ids = df[column].unique()
        mapping = {orig_id: hash_id(orig_id) for orig_id in unique_ids}

        # Replace the column in the DataFrame
        anonymized_df = df.copy()
        anonymized_df[column] = anonymized_df[column].map(mapping)
        return anonymized_df

    except Exception as e:
        logger.error(f"Anonymization failed: {e}")
        return pd.DataFrame()


def load_and_query_csv(file_path_prefix, query):
    # Extract table names from the query
    table_names = re.findall(r'FROM (\w+)', query) + re.findall(r'JOIN (\w+)', query)
    # Create a dictionary to hold DataFrames, keyed by table name
    dfs = {}
    DATA_STEWARD_NAME = os.getenv("DATA_STEWARD_NAME")
    if DATA_STEWARD_NAME == "":
        logger.error(f"DATA_STEWARD_NAME not set.")


    for table_name in table_names:
        try:
            file_name = f"{file_path_prefix}{table_name}_{DATA_STEWARD_NAME}.csv"
            logger.debug(f"Loading file {file_name}")
            dfs[table_name] = pd.read_csv(file_name, delimiter=';')
            logger.debug(f"after read csv")
        except FileNotFoundError:
            logger.error(f"CSV file for table {table_name}_{DATA_STEWARD_NAME} not found.")
            return None

    try:
        # Use pandasql's sqldf function to execute the SQL query
        result_df = sqldf(query, dfs)
    except Exception as e:
        logger.error(f"An error occurred while executing the query: {str(e)}")

    logger.debug(f"after result_df")

    return result_df


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


def protobuf_to_dataframe(data_struct: Struct, metadata: dict = None) -> pd.DataFrame:
    """
    Convert a google.protobuf.Struct and metadata dict back to a pandas DataFrame.
    Assumes data_struct was created by dataframe_to_protobuf above.
    """
    if metadata is None:
        metadata = {}
    data = {}

    for key, value in data_struct.fields.items():
        value_list = value.list_value.values
        items = [v.string_value for v in value_list]
        dtype = metadata.get(key, "object")
        if dtype.startswith("int"):
            data[key] = [int(x) for x in items]
        elif dtype.startswith("float"):
            data[key] = [float(x) for x in items]
        elif dtype == "bool":
            data[key] = [x.lower() in ("true", "1") for x in items]
        else:
            data[key] = items

    return pd.DataFrame(data)

def process_sql_data_request(sqlDataRequest, ctx):
    global config
    logger.debug("Start process_sql_data_request")

    try:
        result = load_and_query_csv(config.dataset_filepath, sqlDataRequest.query)
        logger.debug("after load and query csv")
        data, metadata = dataframe_to_protobuf(result)

        return data, metadata
    except FileNotFoundError:
        logger.error(f"File not found at path {config.dataset_filepath}")
        return None, {}
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return None, {}


# ---  DYNAMOS Interface code At the Bottom -----------------------------------------------------

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


def request_handler(msComm: msCommTypes.MicroserviceCommunication, ctx: Context = None):
    global ms_config

    logger.info(f"[{config.service_name}] Request_type: {msComm.request_type}")

    # Ensure all connections have finished setting up before processing data
    signal_wait(wait_for_setup_event, wait_for_setup_condition)

    try:
        mscomm_metadata = dict(msComm.metadata)
        logger.debug(f"[{config.service_name}] msComm metadata original: {str(mscomm_metadata)}")
        mscomm_metadata = register_service_on_metadata(mscomm_metadata, service_name=config.service_name,)
        logger.debug(f"[{config.service_name}] msComm metadata updated: {str(mscomm_metadata)}")

        # No dataframe_metadata > forward unchanged
        if "dataframe_metadata" not in msComm.metadata:
            logger.warning(f"[{config.service_name}] No 'dataframe_metadata' in msComm.metadata")
            ms_config.next_client.ms_comm.send_data(msComm, data, mscomm_metadata)
            signal_continuation(stop_event, stop_microservice_condition)
            return Empty()

        dataframe_metadata_dict = json.loads(msComm.metadata["dataframe_metadata"])
        data_df = protobuf_to_dataframe(msComm.data, dataframe_metadata_dict)

        logger.debug(f"[{config.service_name}] df head:{data_df.head()}")

        # Add new column to df to see if microservice works
        newColumn = "<<< test-microservice WORKING !!!!!>>>"
        data_df["test-microserviceMarker"] = newColumn

        logger.debug(f"[{config.service_name}] df head with new column:{data_df.head()}")

        data, dataframe_metadata = dataframe_to_protobuf(data_df)
        mscomm_metadata["dataframe_metadata"] = json.dumps(dataframe_metadata)
        
        logger.debug(f"[{config.service_name}] Forwarding result, metadata: {mscomm_metadata}")
        ms_config.next_client.ms_comm.send_data(msComm, data, mscomm_metadata)
        signal_continuation(stop_event, stop_microservice_condition)

        return Empty()

    except Exception as e:
        logger.error(f"[{config.service_name}] An unexpected error occurred: {e}")
        return Empty()


def main():
    global config
    global ms_config

    if test:
        logger.info("Running in test mode")
        return

    ms_config = NewConfiguration(config.service_name, config.grpc_addr, request_handler)

    # Signal the message handler that all connections have been created
    signal_continuation(wait_for_setup_event, wait_for_setup_condition)

    # Wait for the end of processing to shutdown this Microservice
    try:
        signal_wait(stop_event, stop_microservice_condition)

    except KeyboardInterrupt:
        print("KeyboardInterrupt received, stopping server...")
        signal_continuation(stop_event, stop_microservice_condition)


    ms_config.stop(2)
    logger.debug(f"Exiting {config.service_name}")
    sys.exit(0)

# ---  END DYNAMOS Interface code At the Bottom -------------------------------------------------


if __name__ == "__main__":
    main()
