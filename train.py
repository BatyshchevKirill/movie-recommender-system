import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

from autoencoder import train_and_encode

def train_kmeans(
        encoded_users_names: str,
        data_folder: str = "data/interim/",
        save_path: str = "data/interim/ratings.csv",
        n_clusters: int = 10,
):
    users = pd.read_csv(data_folder + encoded_users_names)
    clust = KMeans(n_clusters=n_clusters, random_state=11, n_init="auto").fit_predict(users)

    users_clustered = pd.DataFrame(clust, index=users.index)
    name = encoded_users_names.replace("encoded", "clustered")
    users_clustered.to_csv(data_folder + name)

    ratings = pd.read_csv(data_folder + "train.csv", index_col=0)
    ratings['cluster'] = None
    for i in range(len(ratings)):
        ratings['cluster'][i] = clust[ratings['user_id'][i]]

    rating = ratings.groupby(['cluster', 'item_id']).agg({'rating': ['mean']}).reset_index()
    rating.columns = ["cluster", "item_id", "rating"]
    rating = rating.pivot(index="item_id", columns="cluster", values="rating")
    rating.fillna(0, inplace=True)

    movies = pd.read_csv(data_folder + "films.csv", index_col=0)

    def similar_movies(id: int):
        row_to_compare = movies.loc[id]

        time_diff = movies.iloc[:, 1].sub(row_to_compare.iloc[1]).abs()
        time_diff = time_diff.fillna(1.5)
        genre_sim = (movies.iloc[:, 2:] == row_to_compare.iloc[2:]).sum(axis=1)

        # Create a Series with the results
        result_series = pd.Series(genre_sim - time_diff, index=movies.index).sort_values(ascending=False)
        return result_series[1:6].index.tolist()

    clustered_rating = pd.DataFrame(index=movies.index, columns=rating.columns)
    for index, row in clustered_rating.iterrows():
        data = rating.loc[index].values if index in rating.index else np.zeros(n_clusters)
        for j, cluster in enumerate(rating.columns):
            clustered_rating.at[index, cluster] = data[j]
    for index, row in clustered_rating.iterrows():
        sim = None
        for cluster in rating.columns:
            if clustered_rating.at[index, cluster] < 1:
                if sim is None:
                    sim = similar_movies(index)
                mean = 0
                count = 0
                for m in sim:
                    if m in rating.index and rating.at[m, cluster] > 0.1:
                        mean += rating.at[m, cluster]
                        count += 1
                if count > 0:
                    mean /= count
                else:
                    mean = 3.0

                clustered_rating.at[index, cluster] = mean
    clustered_rating.to_csv(save_path)

# if __name__ == "__main__":
#     train_and_encode(loader: torch.utils.data.DataLoader,
#     input_dim: int = 28,
#     hidden_dim: int = 16,
#     enc_dim: int = 8,
#     epochs: int = 100,
#     noise: float = 0.0,
#     l1_weight: float = 0.0,
#     l2_weight: float = 0.0,
#     cpt_folder: str = "models/",
#     data_folder: str = "data/interim/"
#     ):
