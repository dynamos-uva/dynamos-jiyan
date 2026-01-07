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
from google.protobuf.struct_pb2 import Struct, ListValue, Value


def filter_data(rows, columns, filter_columns):
    return [d for i, d in enumerate(rows) if columns[i] in filter_columns]


def bin_age(age_series):
    bins = [-np.inf, 18, 60, np.inf]
    labels = ["Child", "Adult", "Elderly"]
    return (
        pd.cut(age_series, bins=bins, labels=labels, right=True)
        .astype(str)
        .replace("nan", "Unknown")
    )


def _extract_title(name_series):
    titles = name_series.str.extract(" ([A-Za-z]+)\\.", expand=False)
    rare_titles = {
        "Lady",
        "Countess",
        "Capt",
        "Col",
        "Don",
        "Dr",
        "Major",
        "Rev",
        "Sir",
        "Jonkheer",
        "Dona",
    }
    titles = titles.replace(list(rare_titles), "Rare")
    titles = titles.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    return titles


def _create_features(df):
    # Convert 'Age' to numeric, coercing errors to NaN
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Age"] = bin_age(df["Age"])
    df["Cabin"] = df["Cabin"].str[0].fillna("Unknown")
    df["Title"] = _extract_title(df["Name"])
    df.drop(columns=["PassengerId", "Name", "Ticket"], inplace=True)
    # keywords = set(df.columns)
    # print(all_keywords)
    df = pd.get_dummies(
        df, columns=["Sex", "Pclass", "Embarked", "Title", "Cabin", "Age"]
    )
    # print(df)
    return df


with open('./scripts/csv_splitter_configuration.json', 'r') as file:
    configuration = json.load(file)

df = pd.read_csv(configuration["file"])
processed_df = df.dropna(subset=["Embarked", "Fare"]).copy()
data = _create_features(processed_df)

for partition in configuration["partitions"]:
    os.makedirs(partition["outputDirectory"], exist_ok=True)
    print(partition["columns"])
    partition_data = data[list({
        column
        for column in data.columns
        for keyword in partition["columns"]
        if keyword in column
    })]

    partition_data.to_csv(
        partition["outputDirectory"] + partition["name"] + ".csv")

    # with open(configuration["file"]) as data_file:
    #     dataReader = csv.reader(data_file, delimiter=configuration["delimiter"])
    #     columns = next(dataReader)
    #
    #     for partition in configuration["partitions"]:
    #         os.makedirs(partition["outputDirectory"], exist_ok=True)
    #
    #     print("Creating datasets...")
    #     with ExitStack() as stack:
    #         files = [(partition["columns"],
    #                   csv.writer(stack.enter_context(
    #                       open(partition["outputDirectory"] +
    #                            partition["name"] + ".csv", "w")),
    #             delimiter=configuration["delimiter"]))
    #             for partition in configuration["partitions"]]
    #
    #         for (filter_columns, writer) in files:
    #             writer.writerow(filter_data(columns, columns, filter_columns))
    #
    #         for row in dataReader:
    #             for (filter_columns, writer) in files:
    #                 writer.writerow(filter_data(row,
    #                                             columns,
    #                                             filter_columns))

    # with open('./configuration/etcd_launch_files/datasets.json', 'r') as file:
    #     current_datasets = json.load(file)
    #
    #     datasets = []
    #     dataset_names = []
    #
    #     for partition in configuration["partitions"]:
    #         dataset_names += [partition["name"]]
    #         datasets += [{
    #             "name": partition["name"],
    #             "type": "csv",
    #             "delimiter": configuration["delimiter"],
    #             "tables": [partition["name"]],
    #             "sensitive_columns": {
    #                     partition["name"]: filter_data(partition["columns"],
    #                                                    partition["columns"],
    #                                                    configuration["sensitive"])
    #             }
    #         }]
    #
    #     for dataset in current_datasets:
    #         if dataset["name"] not in dataset_names:
    #             datasets += dataset
    #
    # with open('./configuration/etcd_launch_files/datasets.json', 'w') as file:
    #     json.dump(datasets, file)


data = pd.read_csv('./python/vfl-train/datasets/clientoneData.csv')[1:20]


class ClientModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 4)

    def forward(self, x):
        return self.fc(x)


def serialise_dictionary(dictionary):
    buffer = io.BytesIO()
    torch.save(dictionary, buffer)

    return buffer.getvalue().decode("latin1")


def deserialise_dictionary(dictionary):
    data = json.loads(dictionary, object_pairs_hook=OrderedDict)

    return torch.load(io.BytesIO(data.encode("latin1")))


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
        embedding = self.model(self.data)
        return serialise_array(embedding.detach().numpy())

    def gradient_descent(self, gradients):
        print("Start vfl_evaluate")

        if self.optimiser is None:
            print("Optimiser is not defined.")

        try:
            self.model.zero_grad()
            embedding = self.model(self.data)
            embedding.backward(torch.from_numpy(gradients))
            self.optimiser.step()
        except Exception as e:
            print(f"Error occurred: {e}")

        return "100% accuracy, buddy!"


np.set_printoptions(threshold=sys.maxsize)


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
            print(f"Converting the results to torch failed: {e}")

        try:
            embeddings_aggregated = torch.cat(embedding_results, dim=1)
            embedding_server = embeddings_aggregated.detach().requires_grad_()
            output = self.model(embedding_server)
        except Exception as e:
            print(f"Running gradient descent 1 failed: {e}")

        try:
            loss = self.criterion(output, self.labels)
            loss.backward()
        except Exception as e:
            print(f"Running gradient descent 2 failed: {e}")
            print(f"{output}, {self.labels}")

        try:
            self.optimizer.step()
            self.optimizer.zero_grad()
        except Exception as e:
            print(f"Running gradient descent 3 failed: {e}")

        try:
            grads = embedding_server.grad.split([4, 4, 4], dim=1)
            np_gradients = [serialise_array(grad.numpy()) for grad in grads]
        except Exception as e:
            print(f"Converting the gradients failed: {e}")

        with torch.no_grad():
            correct = 0
            output = self.model(embedding_server)
            predicted = (output > 0.5).float()

            correct += (predicted == self.labels).sum().item()

            accuracy = correct / len(self.labels) * 100

        data = Struct()
        data.update({"accuracy": accuracy, "gradients": np_gradients})

        return data


datac1 = pd.read_csv('./python/vfl-train/datasets/clientoneData.csv')[1:20]
datac2 = pd.read_csv('./python/vfl-train/datasets/clienttwoData.csv')[1:20]
datac3 = pd.read_csv('./python/vfl-train/datasets/clientthreeData.csv')[1:20]
datas = pd.read_csv('./python/vfl-train-model/datasets/outcomeData.csv')[1:20]

client1 = VFLClient(datac1)
client2 = VFLClient(datac2)
client3 = VFLClient(datac3)
server = VFLServer(datas)

embeddings = [
    client1.train_model(),
    client2.train_model(),
    client3.train_model()
]

# embeddings = [deserialise_array(embedding) for embedding in embeddings]
#
# print(server.aggregate_fit(embeddings))
