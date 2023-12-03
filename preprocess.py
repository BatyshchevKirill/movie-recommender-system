import pandas as pd
import os
import itertools
import collections
import networkx as nx
import torch
from sklearn.preprocessing import OneHotEncoder

movie_col = [
    "movie_id", "movie_title", "release_date", "video_release_date", "imbd_url", "unknown", "action", "adventure",
    "animation", "childrens", "comedy", "crime", "documentary", "drama", "fantasy", "film_noir", "horror", "musical",
    "mystery", "romance", "sci_fi", "thriller", "war", "western"
]

user_col = ["user_id", "age", "gender", "occupation", "zip_code"]


def preprocess(data_folder: str = 'data/raw/ml-100k/', save_folder: str = 'data/interim', train_name: str = "u1.base",
               test_name: str = "u1.test"):
    items = pd.read_csv(os.path.join(data_folder, "u.item"), sep="|", names=movie_col, index_col=movie_col[0],
    encoding="latin-1")
    items.release_date = pd.to_datetime(items.release_date, format='%d-%b-%Y')
    items.release_date = (items.release_date - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    first_film_date = items.release_date.min()
    items.release_date = (items.release_date - first_film_date) / (items.release_date.max() - first_film_date)
    items.drop(['video_release_date', 'imbd_url'], axis=1, inplace=True)

    users = pd.read_csv(os.path.join(data_folder, "u.user"), sep="|", names=user_col, index_col=user_col[0], encoding="latin-1")
    users.reset_index(inplace=True)
    users.age = users.age / users.age.max()
    users.gender = users.gender.str.replace("M", "1").replace("F", "0")
    encoder = OneHotEncoder(sparse_output=False)
    encoded_data = encoder.fit_transform(users[['occupation']])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.categories_[0])
    users = pd.concat([users, encoded_df], axis=1)
    users.drop(["occupation", "zip_code", "user_id"], axis=1, inplace=True)

    train_data = pd.read_csv(os.path.join(data_folder, train_name), sep='\t', names=["user_id", "item_id", "rating", "timestamp"])
    test_data = pd.read_csv(os.path.join(data_folder, test_name), sep='\t', names=["user_id", "item_id", "rating", "timestamp"])
    train_data.user_id = train_data.user_id.apply(lambda x: x-1)
    test_data.user_id = test_data.user_id.apply(lambda x: x - 1)
    train_data.drop("timestamp", axis=1, inplace=True)
    test_data.drop("timestamp", axis=1, inplace=True)

    items.to_csv(os.path.join(save_folder, "films.csv"))
    users.to_csv(os.path.join(save_folder, "users.csv"))
    train_data.to_csv(os.path.join(save_folder, "train.csv"))
    test_data.to_csv(os.path.join(save_folder, "test.csv"))


def create_loader(dirpath: str = 'data/interim/', alpha=0.01, batch_size: int = 16):
    users = pd.read_csv(os.path.join(dirpath, 'users.csv'), index_col=0)
    rates = pd.read_csv(os.path.join(dirpath, 'train.csv'), index_col=0)
    pairs = []
    grouped = rates.groupby(['item_id', 'rating'])

    for key, group in grouped:
        pairs.extend(list(itertools.combinations(group['user_id'], 2)))
    counter = collections.Counter(pairs)
    alpha *= len(rates.item_id.unique())

    edge_list = map(list, collections.Counter(el for el in counter.elements() if counter[el] >= alpha).keys())

    G = nx.Graph()

    for i in edge_list:
        G.add_edge(i[0], i[1], weight=1)

    pr = nx.pagerank(G.to_directed())
    users['pagerank'] = users.index.map(pr)
    users['pagerank'] /= float(users['pagerank'].max())
    dc = nx.degree_centrality(G)
    users['degree_centrality'] = users.index.map(dc)
    users['degree_centrality'] /= float(users['degree_centrality'].max())
    cc = nx.closeness_centrality(G)
    users['closeness_centrality'] = users.index.map(cc)
    users['closeness_centrality'] /= float(users['closeness_centrality'].max())
    bc = nx.betweenness_centrality(G)
    users['betweenness_centrality'] = users.index.map(bc)
    users['betweenness_centrality'] /= float(users['betweenness_centrality'].max())
    lc = nx.load_centrality(G)
    users['load_centrality'] = users.index.map(lc)
    users['load_centrality'] /= float(users['load_centrality'].max())
    nd = nx.average_neighbor_degree(G, weight='weight')
    users['average_neighbor_degree'] = users.index.map(nd)
    users['average_neighbor_degree'] /= float(users['average_neighbor_degree'].max())
    x_train = users[users.columns[1:]].fillna(0)
    x_train = torch.tensor(x_train.values, dtype=torch.float32)
    x_train = torch.utils.data.TensorDataset(x_train)
    g = torch.Generator()
    g.manual_seed(42)
    return torch.utils.data.DataLoader(x_train, batch_size=batch_size, shuffle=True, pin_memory=True, generator=g)


if __name__ == "__main__":
    preprocess()
