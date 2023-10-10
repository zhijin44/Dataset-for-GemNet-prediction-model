from abc import ABC

import numpy as np
import pandas as pd
import os
import torch
from torch_geometric.data import Data, Dataset


def sort_files(root_dir):
    files_wifi, files_5g = [], []
    for f_name in os.listdir(root_dir):
        if f_name.endswith('wifi.p2m'):
            files_wifi.append(f_name)
        if f_name.endswith('5g.p2m'):
            files_5g.append(f_name)
    files_wifi.sort(), files_5g.sort()

    if len(files_wifi) != len(files_5g):
        print(f"{root_dir} dimension is different!!! files_wifi({len(files_wifi)}) files_5g({len(files_5g)})")
    else:
        print(f"{root_dir} parse and sort successfully")
    return files_wifi, files_5g


def process_files(files, root_dir, threshold):
    all_matrices = []

    for file in files:
        df = pd.read_csv(os.path.join(root_dir, file), delimiter=' ', skiprows=3, header=None, usecols=[5])
        df['binary'] = np.where(df[5] >= threshold, 1, 0)
        matrix = df['binary'].values.reshape(-1, 1)  # reshape into a 400x1 vector
        all_matrices.append(matrix)
    return np.array(all_matrices)


def get_mat(root_dir, threshold_wifi, threshold_5g):
    files_wifi, files_5g = sort_files(root_dir)

    label = []
    for file_wifi, file_5g in zip(files_wifi, files_5g):
        parts_wifi, parts_5g = file_wifi.split("_"), file_5g.split("_")
        if parts_wifi[:2] == parts_5g[:2]:
            label.append(list(map(float, parts_wifi[:2])))

    mat_wifi = process_files(files_wifi, root_dir, threshold_wifi)
    mat_5g = process_files(files_5g, root_dir, threshold_5g)

    return mat_wifi, mat_5g, np.array(label)


def process_all_folders(folders_list, threshold_wifi, threshold_5g):
    all_mat_wifi = []
    all_mat_5g = []
    all_labels = []

    for folder in folders_list:
        mat_wifi, mat_5g, labels = get_mat(folder, threshold_wifi, threshold_5g)

        all_mat_wifi.append(mat_wifi)
        all_mat_5g.append(mat_5g)
        all_labels.append(labels)

    # Concatenate all the matrices into one big numpy array
    all_mat_wifi = np.concatenate(all_mat_wifi, axis=0)
    all_mat_5g = np.concatenate(all_mat_5g, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_mat_wifi, all_mat_5g, all_labels


def get_edge_index():
    # total nodes num_nodes = 20 * 20
    edges = []

    # for each node, add edges for its four neighbors
    for i in range(20):
        for j in range(20):
            node_idx = i * 20 + j
            # up
            if i > 0:
                edges.append((node_idx, node_idx - 20))
            # down
            if i < 19:
                edges.append((node_idx, node_idx + 20))
            # left
            if j > 0:
                edges.append((node_idx, node_idx - 1))
            # right
            if j < 19:
                edges.append((node_idx, node_idx + 1))

    # convert to tensor and add dimension
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


class GraphDataset(Dataset, ABC):
    def __init__(self, root, features, edge_index, labels, transform=None, pre_transform=None):
        self.features = features
        self.edge_index = edge_index
        self.labels = labels
        super(GraphDataset, self).__init__(root, transform, pre_transform)

    def len(self):
        return len(self.features)

    def get(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        data = Data(x=x, edge_index=self.edge_index, y=y)
        return data


def save_dataset_double(folders_list, threshold_wifi, threshold_5g):
    all_mat_wifi, all_mat_5g, all_labels = process_all_folders(folders_list, threshold_wifi, threshold_5g)

    # stack all_mat_wifi & all_mat_5g with shape (N, 400, 1) -> (2N, 400, 2)
    stacked_features_1 = np.stack((all_mat_wifi, all_mat_5g), axis=-1)
    stacked_features_2 = np.stack((all_mat_5g, all_mat_wifi), axis=-1)
    # 交替组合
    assert stacked_features_1.shape[0] == stacked_features_2.shape[0]
    # Combine the two arrays such that the elements from the two are alternated
    stacked_features = np.empty((2 * stacked_features_1.shape[0],) + stacked_features_1.shape[1:],
                                dtype=stacked_features_1.dtype)
    stacked_features[0::2] = stacked_features_1
    stacked_features[1::2] = stacked_features_2
    stacked_features = np.squeeze(stacked_features, axis=2)  # Remove the third dimension
    x = torch.tensor(stacked_features, dtype=torch.float64)  # x will have shape (2N, 400, 2)

    all_labels_interleaved = np.repeat(all_labels, 2, axis=0)  # y will have shape (2N, 2)
    y = torch.tensor(all_labels_interleaved, dtype=torch.float64).view(-1, 1, 2)  # y will have shape (2N, 1, 2)

    edge_index = get_edge_index()

    root_dir = os.path.join(os.getcwd(), 'dataset')
    dataset = GraphDataset(root=root_dir, features=x, edge_index=edge_index, labels=y)

    # Save the dataset
    torch.save(dataset, 'dataset/dataset.pt')


def save_dataset(folders_list, threshold_wifi, threshold_5g):
    all_mat_wifi, all_mat_5g, all_labels = process_all_folders(folders_list, threshold_wifi, threshold_5g)

    # stack all_mat_wifi & all_mat_5g with shape (N, 400, 1) -> (N, 400, 2)
    stacked_features = np.stack((all_mat_wifi, all_mat_5g), axis=-1)
    stacked_features = np.squeeze(stacked_features, axis=2)  # Remove the third dimension
    x = torch.tensor(stacked_features, dtype=torch.float64)  # x will have shape (N, 400, 2)

    y = torch.tensor(all_labels, dtype=torch.float64).view(-1, 1, 2)  # y will have shape (N, 1, 2)

    edge_index = get_edge_index()

    root_dir = os.path.join(os.getcwd(), 'dataset')
    dataset = GraphDataset(root=root_dir, features=x, edge_index=edge_index, labels=y)

    # Save the dataset
    torch.save(dataset, 'dataset/no_double_dataset.pt')


def save_dataset_for_per_con(folders_list, threshold_wifi, threshold_5g):
    all_mat_wifi, all_mat_5g, all_labels = process_all_folders(folders_list, threshold_wifi, threshold_5g)

    # stack all_mat_wifi & all_mat_5g with shape (N, 400, 1) -> (2N, 400, 2)
    stacked_features_1 = np.stack((all_mat_wifi, all_mat_5g), axis=-1)
    stacked_features_2 = np.stack((all_mat_5g, all_mat_wifi), axis=-1)
    # 交替组合
    assert stacked_features_1.shape[0] == stacked_features_2.shape[0]
    # Combine the two arrays such that the elements from the two are alternated
    stacked_features = np.empty((2 * stacked_features_1.shape[0],) + stacked_features_1.shape[1:],
                                dtype=stacked_features_1.dtype)
    stacked_features[0::2] = stacked_features_1
    stacked_features[1::2] = stacked_features_2
    stacked_features = np.squeeze(stacked_features, axis=2)  # Remove the third dimension
    x = torch.tensor(stacked_features, dtype=torch.float64)  # x will have shape (2N, 400, 2)

    all_labels_per = all_labels[:, 1].repeat(2)  # Per values are in the second column
    all_labels_con = all_labels[:, 0].repeat(2)  # Con values are in the first column
    y_per = torch.tensor(all_labels_per, dtype=torch.float64).view(-1, 1, 1)  # y_per will have shape (2N, 1, 1)
    y_con = torch.tensor(all_labels_con, dtype=torch.float64).view(-1, 1, 1)  # y_con will have shape (2N, 1, 1)

    edge_index = get_edge_index()

    root_dir = os.path.join(os.getcwd(), 'dataset')
    dataset_per = GraphDataset(root=root_dir, features=x, edge_index=edge_index, labels=y_per)
    dataset_con = GraphDataset(root=root_dir, features=x, edge_index=edge_index, labels=y_con)

    # Save the datasets
    torch.save(dataset_per, 'dataset/dataset_per_22.5k.pt')
    torch.save(dataset_con, 'dataset/dataset_con_22.5k.pt')


def main():
    # threshold_wifi = -92
    # threshold_5g = -77
    # folders_list = ['e-3_n3000', 'e-4_n580', 'e-5_n540', 'e-6_n540', 'e-7_n540', 'e-8_n540', 'e-9_n540', 'e-10_n540', 'e-11_n540', 'e-12_n540', 'e-13_n500', 'e-14_n540', 'e-15_n540']
    #
    # save_dataset_for_per_con(folders_list, threshold_wifi, threshold_5g)
    #
    # print("------------------------------------------")

    # Load the dataset
    # dataset = torch.load('dataset/dataset.pt')
    dataset_con = torch.load('dataset/dataset_con_22.5k.pt')
    dataset_per = torch.load('dataset/dataset_per_22.5k.pt')

    # Access and print some attributes of the dataset
    print(dataset_per, dataset_con)
    print("Number of samples in the dataset:", len(dataset_per))
    print("One sample in the dataset:", dataset_per[98])
    print(dataset_per[99])

    # You can also access and print features, edge_index, labels of the first graph in the dataset like this
    # print("Features of the first graph: ", dataset[1].x)
    # print("Edge index of the first graph: ", dataset[0].edge_index)
    # print("Labels of the graph: ", dataset[198].y, dataset[199].y)
    # print(dataset_con[198].y, dataset_per[198].y)
    # print(dataset_con[199].y, dataset_per[199].y)

    # print("------------------------------------------")
    #
    # test_con = 1e-3
    # mapped = map_con(test_con)
    # demapped = demap_con(mapped)
    # print(f"Original con: {test_con}, Mapped: {mapped}, Demapped: {demapped}")


if __name__ == '__main__':
    main()
