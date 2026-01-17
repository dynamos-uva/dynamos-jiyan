import os
import sys
import json
import time
import base64
import socket
import hashlib
import threading
import numpy as np

from google.protobuf.empty_pb2 import Empty
from google.protobuf.struct_pb2 import Struct, Value, ListValue

from dynamos.ms_init import NewConfiguration
from dynamos.signal_flow import signal_continuation, signal_wait
from dynamos.logger import InitLogger

import rabbitMQ_pb2 as rabbitTypes
import microserviceCommunication_pb2 as msCommTypes
from opentelemetry.context.context import Context

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


# Just for testing, mark so I can see if it works
def fp(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]

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

def serialise_array(array):
    array = np.ascontiguousarray(array)
    payload = {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "data": base64.b64encode(array.tobytes()).decode("ascii"),
    }
    return json.dumps(payload, separators=(",", ":"))

def deserialise_array(string):
    obj = json.loads(string)
    dtype = np.dtype(obj["dtype"])
    raw = base64.b64decode(obj["data"].encode("ascii"))
    arr = np.frombuffer(raw, dtype=dtype)
    return arr.reshape(obj["shape"])

# Computes norm and clips if norm exceeds clip_norm
def l2_clip(arr, clip_norm):
    if arr.ndim == 1:
        norm = np.linalg.norm(arr, ord=2)
        if norm <= 0:
            return arr
        scale = min(1.0, clip_norm / norm)
        return arr * scale
    norms = np.linalg.norm(arr, ord=2, axis=1, keepdims=True) + 1e-12
    scales = np.minimum(1.0, clip_norm / norms)
    return arr * scales

# Computes gaussian noise sigma
def gaussian_sigma(epsilon, delta, sensitivity):
    return sensitivity * np.sqrt(2.0 * np.log(1.25 / delta)) / epsilon

# Applies the clipping + noise
def dp_sanitize_embedding(serialized, clip_norm, epsilon, delta, seed, noise_enabled = True):
    # Deserializes embeddings
    arr = deserialise_array(serialized).astype(np.float32, copy=False)

    arr = l2_clip(arr, clip_norm=clip_norm)

    if not noise_enabled:
        return serialise_array(arr)
    sigma = gaussian_sigma(epsilon=epsilon, delta=delta, sensitivity=clip_norm)
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=arr.shape).astype(np.float32)
    arr_noisy = arr + noise
    # Returns serialized array but with noise/clipped
    return serialise_array(arr_noisy)

def request_handler(msComm: msCommTypes.MicroserviceCommunication, ctx: Context = None):
    global ms_config

    signal_wait(wait_for_setup_event, wait_for_setup_condition)
    request = rabbitTypes.Request()
    try:
        msComm.original_request.Unpack(request)
    except Exception as e:
        logger.error(f"Failed to unpack original request: {e}")
        ms_config.next_client.ms_comm.send_data(msComm, msComm.data, dict(msComm.metadata))
        return Empty()

    epsilon = 1000.0
    delta = 1e-5
    clip_norm = 3000.0
    noise_enabled = True 
    base_seed = 12345
    dp_enabled = True

    cid = getattr(msComm.request_metadata, "correlation_id", "")
    if cid:
        derived = int(hashlib.sha256(cid.encode("utf-8")).hexdigest()[:8], 16)
        seed = (base_seed ^ derived) & 0xFFFFFFFF
    else:
        seed = base_seed & 0xFFFFFFFF

    if request.type == "vflAggregateRequest":
        try:
            if dp_enabled == False:
                md = dict(msComm.metadata)
                md["dp_applied"] = "0"
                md["dp_enabled"] = "0"
                ms_config.next_client.ms_comm.send_data(msComm, msComm.data, md)
                return Empty()
            embeddings_val = request.data.get("embeddings", None)
            if embeddings_val is None:
                logger.warning("vflAggregateRequest had no embeddings in request.data")
                ms_config.next_client.ms_comm.send_data(msComm, msComm.data, dict(msComm.metadata))
                return Empty()

            embeddings_in = [v.string_value for v in embeddings_val.list_value.values]
            embeddings_out = [
                dp_sanitize_embedding(e, clip_norm=clip_norm, epsilon=epsilon, delta=delta, seed=None if seed is None else (seed + i) & 0xFFFFFFFF, noise_enabled=noise_enabled)
                for i, e in enumerate(embeddings_in)
            ]

            lv = ListValue()
            lv.values.extend([Value(string_value=s) for s in embeddings_out])
            request.data["embeddings"].CopyFrom(Value(list_value=lv))

            msComm.original_request.Pack(request)

            md = dict(msComm.metadata)
            md["dp_applied"] = "1"
            md["dp_clip_norm"] = str(clip_norm)
            md["dp_noise"] = "0" if not noise_enabled else "1"

            if noise_enabled:
                md["dp_epsilon"] = str(epsilon)
                md["dp_delta"] = str(delta)

            ms_config.next_client.ms_comm.send_data(msComm, msComm.data, md)
            return Empty()

        except Exception as e:
            logger.error(f"DP failed; forwarding unchanged. Error: {e}")
            ms_config.next_client.ms_comm.send_data(msComm, msComm.data, dict(msComm.metadata))
            return Empty()

    ms_config.next_client.ms_comm.send_data(msComm, msComm.data, dict(msComm.metadata))
    return Empty()

def main():
    global ms_config

    sidecar_port = int(os.getenv("SIDECAR_PORT", "50051"))
    logger.info(f"Waiting for sidecar gRPC on localhost:{sidecar_port} ...")
    wait_for_port("127.0.0.1", sidecar_port, wait_time=90)

    ms_config = NewConfiguration(config.service_name, config.grpc_addr, request_handler)

    signal_continuation(wait_for_setup_event, wait_for_setup_condition)

    try:
        signal_wait(stop_event, stop_microservice_condition)
    except KeyboardInterrupt:
        signal_continuation(stop_event, stop_microservice_condition)

    ms_config.stop(2)
    logger.debug(f"Exiting {config.service_name}")
    sys.exit(0)


if __name__ == "__main__":
    main()
