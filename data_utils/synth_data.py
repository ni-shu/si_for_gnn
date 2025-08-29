import numpy as np
import networkx as nx
import sys
import torch

from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))


eeg_gen_mode_list = ["eye", "full"]  # , "exp", "linear"]

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

rho = 0.1


def all_pairs_shortest_distances(adj_matrix):

    # Convert adjacency matrix to a graph
    adj_matrix_copy = adj_matrix.copy()
    np.fill_diagonal(adj_matrix_copy, 0)
    binary_adj_matrix = (adj_matrix_copy > 0).astype(int)
    G = nx.from_numpy_array(binary_adj_matrix)

    # Compute shortest paths between all pairs of nodes
    shortest_paths = dict(nx.all_pairs_shortest_path_length(G))

    # Convert node distances to matrix form
    n = len(adj_matrix)
    distances = np.full((n, n), -1, dtype=int)

    for src, paths in shortest_paths.items():
        for dest, distance in paths.items():
            distances[src, dest] = distance

    return distances


def create_exp_matrix(n):
    # Σ_{i,j} = 0.5^{|i - j|}
    exp_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            exp_matrix[i, j] = rho ** (abs(i - j))
    return exp_matrix


def make_cov_based_on_A(A, num_features):
    """
    Generate covariance matrix based on adjacency matrix A.    

    Parameters:
        A (numpy.ndarray): Adjacency matrix.

    Returns:
        numpy.ndarray: Covariance matrix.
    """
    distance_matrix = all_pairs_shortest_distances(A)

    space_cov = rho ** (distance_matrix)
    space_cov[distance_matrix == -1] = 0

    feature_cov = create_exp_matrix(num_features)
    cov = np.kron(space_cov, feature_cov)
    return cov, space_cov, create_exp_matrix(num_features)


def add_bias(X, A, anomaly_node, bias, rng):

    if anomaly_node is None:
        num_nodes, num_features = X.shape
        anomaly_node_ratio = 0.1
        anomaly_node_count = int(
            np.ceil(num_nodes * anomaly_node_ratio)
        ) 

        selected_node = rng.integers(0, num_nodes)
        anomaly_node = [selected_node]
        selected_node_idx = 0 

        while len(anomaly_node) < anomaly_node_count:
            if selected_node_idx < len(anomaly_node):
                selected_node = anomaly_node[selected_node_idx]
                selected_node_idx += 1
            else:
                new_node = rng.integers(0, num_nodes)
                if new_node not in anomaly_node:
                    anomaly_node.append(new_node)
                continue

            neibor_nodes = np.where((A[:, selected_node] + A[selected_node, :]) > 0)[0]
            anomaly_node.extend(neibor_nodes)
            anomaly_node = list(dict.fromkeys(anomaly_node))
            anomaly_node = anomaly_node[:anomaly_node_count]

    if isinstance(bias, list):
        X[anomaly_node] += rng.uniform(bias[0], bias[1])
    elif isinstance(bias, float) or isinstance(bias, int):
        X[anomaly_node] += bias
    else:
        raise ValueError("Invalid input type. Expected list or integer.")
    return X


def generate_random_graph(
    num_nodes,
    num_features=1,
    rng=None,
    directed=True,
    anomaly=False,
    bias=1.0,
    anomaly_node=None,
    normalized=True,
    edge_prob=0.3,
    A=None,
    mu=None,
    cov=None,
    var_based_on_A=False,
):

    if rng is None:
        rng = np.random.default_rng()

    # Generate adjacency matrix A (with edge probability edge_prob)
    if A is None:
        A = rng.choice(
            [0.0, 1.0], size=(num_nodes, num_nodes), p=[1 - edge_prob, edge_prob]
        )
        A = A - np.diag(np.diag(A))
        if not directed:
            A = np.triu(A, 1) + np.triu(A, 1).T
        A += np.eye(num_nodes)
        if normalized:
            D_inv = np.diag(1 / np.sum(A, axis=1))
            A = D_inv @ A

    if var_based_on_A:
        assert cov is None
        assert not directed
        assert mu is not None

        # covariance matrix based on A
        cov, space_cov, time_cov = make_cov_based_on_A(A, num_features)
        X = rng.multivariate_normal(mu, cov, size=1).reshape(num_nodes, num_features)


    else:
        if mu is None or cov is None:
            X = rng.normal(0, 1, size=(num_nodes, num_features))
        if mu is not None and cov is not None:
            if np.all((cov - np.diag(np.diag(cov))) == 0):
                X = (
                    rng.normal(0, 1, size=num_nodes * num_features)
                    * np.sqrt(np.diag(cov))
                    + mu
                )
                X = X.reshape(num_nodes, num_features)
            else:
                X = rng.multivariate_normal(mu, cov, size=1).reshape(
                    num_nodes, num_features
                )

    if anomaly:
        X = add_bias(X, A, anomaly_node, bias, rng=rng)

    return A, X, cov


class GraphDataset(Dataset):

    def __init__(
        self,
        num_data,
        num_features,
        num_nodes,
        var=1.0,
        anomaly_p=0.0,
        edge_prob=0.3,
        bias=[1],
        rng=None,
        eeg_mode=False,
        eeg_mu=None,
        eeg_cov=None,
        eeg_A=None,
        var_based_on_A=False,
    ):
        self.num_data = num_data
        self.num_features = num_features
        self.num_nodes = num_nodes
        self.edge_prob = edge_prob
        self.anomaly_p = anomaly_p
        self.rng = rng if rng is not None else np.random.default_rng()
        self.eeg_mode = eeg_mode
        self.eeg_A = eeg_A
        self.eeg_mu = eeg_mu
        self.eeg_cov = eeg_cov
        self.var = var

        assert eeg_mode in eeg_gen_mode_list + [False]
        self.data = self._graph_generator(
            self.rng,
            num_data,
            self.num_features,
            self.num_nodes,
            self.anomaly_p,
            self.edge_prob,
            bias=bias,
            var_based_on_A=var_based_on_A,
        )

    def _graph_generator(
        self,
        rng,
        num_data,
        num_features,
        num_nodes_list=32,
        anomaly_p=0.5,
        edge_prob=0.3,
        bias=[1],
        var_based_on_A=False,
    ):
        generated_data_list = []
        directed = False
        for _ in range(num_data):
            if isinstance(num_nodes_list, list):
                num_nodes = rng.choice(num_nodes_list, size=1, replace=False)[0]
            elif isinstance(num_nodes_list, int):
                num_nodes = num_nodes_list
            else:
                raise ValueError(
                    "num_nodes, Invalid input type. Expected list or integer."
                )

            if self.eeg_mode in eeg_gen_mode_list:
                assert var_based_on_A is False
                norm_A = (
                    self.eeg_A
                )  # eeg_data.get_eeg_dist_parametersを使って取得したAは既に正規化されている
                mu = self.eeg_mu
                cov = self.eeg_cov
                if np.all((cov - np.diag(np.diag(cov))) == 0):
                    X = (
                        rng.normal(0, 1, size=num_nodes * num_features)
                        * np.sqrt(np.diag(cov))
                        + mu
                    )
                    X = X.reshape(num_nodes, num_features)
                else:
                    X = rng.multivariate_normal(mu, cov, size=1).reshape(
                        num_nodes, num_features
                    )
                onehot_label = None

            elif self.eeg_mode is False:
                A = None
                mu = np.zeros(num_features * num_nodes)
                cov = (
                    np.eye(num_features * num_nodes) * self.var
                    if type(self.var) == float
                    else self.var
                )
                anomaly = 1 if rng.random() <= anomaly_p else 0
                num_classes = 2
                onehot_label = np.zeros(num_classes)
                onehot_label[anomaly] = 1

                norm_A, X, self.cov = generate_random_graph(
                    num_nodes,
                    num_features=num_features,
                    rng=rng,
                    directed=directed,
                    anomaly=anomaly,
                    bias=bias,
                    anomaly_node=None,
                    edge_prob=edge_prob,
                    A=A,
                    mu=mu,
                    cov=cov,
                    var_based_on_A=var_based_on_A,
                )
            else:
                raise ValueError("Invalid eeg_mode")

            generated_data_list.append(((norm_A, X), onehot_label))
        return generated_data_list

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        (adj_matrix, node_features), label = self.data[idx]

        adj_matrix = torch.FloatTensor(adj_matrix)
        node_features = torch.FloatTensor(node_features)
        if label is not None:
            label = torch.FloatTensor(label)

        return (adj_matrix, node_features), label


def create_splitted_dataset(
    num_data,
    num_features=1,
    num_nodes=32,
    batch_size=5,
    test_ratio=0.2,
    val_ratio=0.2,
    edge_prob=0.3,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    anomaly_p = 0.5
    dataset = GraphDataset(
        num_data,
        num_features,
        num_nodes,
        anomaly_p=anomaly_p,
        edge_prob=edge_prob,
        rng=rng,
        bias=[0.1, 0.2],
    )

    num_batches = (
        num_data + batch_size - 1
    ) // batch_size

    test_data_size_batches = int(num_batches * test_ratio)
    val_data_size_batches = int(num_batches * val_ratio)
    # train_data_size_batches = num_batches - (
    #     test_data_size_batches + val_data_size_batches
    # )

    test_size = test_data_size_batches * batch_size
    val_size = val_data_size_batches * batch_size
    train_size = len(dataset) - (test_size + val_size)

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # print("Number of data points:", num_data)
    # print("Batch size:", batch_size)
    # print("Total batches:", num_batches)
    # print("Test data size (batches):", test_data_size_batches)
    # print("Validation data size (batches):", val_data_size_batches)
    # print("Training data size (batches):", train_data_size_batches)
    # print("Test data size (data points):", test_data_size_batches * batch_size)
    # print("Validation data size (data points):", val_data_size_batches * batch_size)
    # print(
    #     "Training data size (data points):",
    #     num_data - (test_data_size_batches + val_data_size_batches) * batch_size,
    # )
    return train_loader, val_loader, test_loader





def split_time_series(X, chunk_length, slide_width):
    """
    Splits the time series data X into chunks of specified length and slide width,
    and reshapes it to the size [batch_size, sensor_count x chunk_count, chunk_length].
    Args:
        X: torch.Tensor, size [batch_size, sensor_count, time_length]
        chunk_length: int, length of the time series to split
        slide_width: int, slide width
    Returns:
        torch.Tensor, new size [batch_size, sensor_count x chunk_count, chunk_length]
    """
    batch_size, sensor_count, time_length = X.size()
    chunks = (time_length - chunk_length) // slide_width + 1  # チャンク数を計算

    X_split = torch.stack(
        [
            X[:, :, i : i + chunk_length]
            for i in range(0, time_length - chunk_length + 1, slide_width)
        ],
        dim=2,
    )

    X_reshaped = X_split.reshape(batch_size, sensor_count * chunks, chunk_length)

    return X_reshaped


def norm_A(A):
    assert torch.all(torch.diagonal(A, dim1=1, dim2=2).sum(dim=1) == A.size(1))
    row_sums = A.sum(dim=2, keepdim=True)
    D_inv = 1 / row_sums
    A_normalized = A * D_inv
    return A_normalized


def denorm_A(A_normalized):
    row_sums = (A_normalized != 0).sum(dim=2, keepdim=True).float()
    A_original = A_normalized * row_sums
    return A_original


def adjust_adjacency_matrix(A, X_split):
    """
    隣接行列をチャンク間の関係を考慮して拡張し、
    サイズ (バッチ数, センサ数 x チャンク数, センサ数 x チャンク数) に調整。

    Args:
        A: torch.Tensor, サイズ [バッチ, センサ数, センサ数]
        X_split: torch.Tensor, サイズ [バッチ, センサ数, チャンク数, チャンク長]

    Returns:
        torch.Tensor, サイズ [バッチ, センサ数 x チャンク数, センサ数 x チャンク数]
    
    """
    batch_size, sensor_count_x_chunk_count, _ = X_split.size()
    sensor_count = A.size(1)
    chunk_count = sensor_count_x_chunk_count // sensor_count

    norm_flag = False
    if not torch.all((A == 0) | (A == 1)):
        A = denorm_A(A)
        norm_flag = True
        assert torch.all((A == 0) | (A == 1))

    A_block = torch.zeros(
        (batch_size, sensor_count * chunk_count, sensor_count * chunk_count),
        device=A.device,
        dtype=A.dtype,
    )

    for b in range(batch_size):
        for i in range(chunk_count):
            start_i = i * sensor_count
            end_i = (i + 1) * sensor_count
            A_block[b, start_i:end_i, start_i:end_i] = A[b]

            if i < chunk_count - 1:
                next_start_i = (i + 1) * sensor_count
                A_block[
                    b, start_i:end_i, next_start_i : next_start_i + sensor_count
                ] = torch.eye(sensor_count, device=A.device)
                A_block[
                    b, next_start_i : next_start_i + sensor_count, start_i:end_i
                ] = torch.eye(sensor_count, device=A.device)

    new_indices = torch.tensor(
        [i * chunk_count + j for j in range(chunk_count) for i in range(sensor_count)]
    )
    new_indices = torch.argsort(new_indices)
    A_rearranged = A_block[:, new_indices, :][:, :, new_indices]

    if norm_flag:
        A_rearranged = norm_A(A_rearranged)

    return A_rearranged


def chunk_time_series(A, X, chunk_length, slide_width):
    X_split = split_time_series(X, chunk_length, slide_width)
    A_adjusted = adjust_adjacency_matrix(A, X_split)
    return A_adjusted, X_split


if __name__ == "__main__":
    import doctest
    doctest.testmod()
