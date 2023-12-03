import argparse

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from autoencoder import train_and_encode
from preprocess import create_loader

np.random.seed(42)


def train_kmeans(
    encoded_users_names: str,
    data_folder: str = "data/interim/",
    save_path: str = "data/interim/ratings.csv",
    n_clusters: int = 10,
):
    """
    Trains a KMeans clustering model on encoded user data, assigns user clusters, and calculates
    film ratings for each cluster. The results are saved to a CSV file.

    Parameters:
    - encoded_users_names (str): File name of the encoded user data.
    - data_folder (str): Folder containing input data.
    - save_path (str): File path to save the resulting clustered ratings.
    - n_clusters (int): Number of clusters for KMeans clustering.

    Returns:
    None
    """
    # Split users to clusters
    users = pd.read_csv(data_folder + encoded_users_names)
    clust = KMeans(n_clusters=n_clusters, random_state=11, n_init="auto").fit_predict(
        users
    )

    # Save user clusters
    users_clustered = pd.DataFrame(clust, index=users.index)
    name = encoded_users_names.replace("encoded", "clustered")
    users_clustered.to_csv(data_folder + name)

    # Prepare the rating table based on train data
    ratings = pd.read_csv(data_folder + "train.csv", index_col=0)
    ratings["cluster"] = None
    for i in range(len(ratings)):
        ratings["cluster"][i] = clust[ratings["user_id"][i]]

    # Calculate mean rating of a film for each cluster
    rating = (
        ratings.groupby(["cluster", "item_id"]).agg({"rating": ["mean"]}).reset_index()
    )
    rating.columns = ["cluster", "item_id", "rating"]
    rating = rating.pivot(index="item_id", columns="cluster", values="rating")
    rating.fillna(0, inplace=True)

    movies = pd.read_csv(data_folder + "films.csv", index_col=0)

    def similar_movies(id: int):
        # Find ids of similar movies
        row_to_compare = movies.loc[id]

        # Calculate release time difference
        time_diff = movies.iloc[:, 1].sub(row_to_compare.iloc[1]).abs()
        time_diff = time_diff.fillna(1.5)
        # Calculate genre similarity
        genre_sim = (movies.iloc[:, 2:] == row_to_compare.iloc[2:]).sum(axis=1)

        # Create a Series with the results
        result_series = pd.Series(
            genre_sim - time_diff, index=movies.index
        ).sort_values(ascending=False)
        return result_series[1:6].index.tolist()

    # Calulate film ratings by clusters using mean
    clustered_rating = pd.DataFrame(index=movies.index, columns=rating.columns)
    for index, row in clustered_rating.iterrows():
        data = (
            rating.loc[index].values if index in rating.index else np.zeros(n_clusters)
        )
        for j, cluster in enumerate(rating.columns):
            clustered_rating.at[index, cluster] = data[j]

    # Calculate film ratings for unknown films based on similar films
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

    # Save the final mapping table
    clustered_rating.to_csv(save_path)
    print("Saved results to", save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dirpath", default="data/interim/")
    parser.add_argument("-a", "--alpha", default=0.01)
    parser.add_argument("-b", "--batch_size", default=16)
    parser.add_argument("-id", "--input_dim", default=28)
    parser.add_argument("-hd", "--hidden_dim", default=16)
    parser.add_argument("-ed", "--enc_dim", default=8)
    parser.add_argument("-e", "--epochs", default=100)
    parser.add_argument("-n", "--noise", default=0.0)
    parser.add_argument("-l1", "--l1_weight", default=0.0)
    parser.add_argument("-l2", "--l2_weight", default=0.0)
    parser.add_argument("-c", "--cpt_folder", default="models/")
    parser.add_argument("-n", "-cluster_number", default=10)

    args = parser.parse_args()

    # Prepare dataloaders
    x_train = create_loader(
        dirpath=args.dirpath, alpha=float(args.alpha), batch_size=int(args.batch_size)
    )

    # Train autoencoder
    train_and_encode(
        x_train,
        input_dim=int(args.input_dim),
        hidden_dim=int(args.hidden_dim),
        enc_dim=int(args.enc_dim),
        epochs=int(args.epochs),
        noise=float(args.noise),
        l1_weight=float(args.l1_weight),
        l2_weight=float(args.l2_weight),
        cpt_folder=args.cpt_folder,
        data_folder=args.dirpath,
    )

    # Train the final model
    name = f"encoded_ae_{args.hidden_dim}_{args.enc_dim}_{args.epochs}_{args.noise}_{args.l1_weight:.4f}_{args.l2_weight:.4f}".replace(
        ".", "_"
    )
    train_kmeans(name, n_clusters=int(args.cluster_number))
