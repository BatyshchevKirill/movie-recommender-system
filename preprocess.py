import argparse
import os
import pandas as pd
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
    items.release_date = (items.release_date - first_film_date) / items.release_date.max()
    items.reset_index(inplace=True)
    items.drop(['video_release_date', 'imbd_url', 'movie_id'], axis=1, inplace=True)

    users = pd.read_csv(os.path.join(data_folder, "u.user"), sep="|", names=user_col, index_col=user_col[0], encoding="latin-1")
    users.reset_index(inplace=True)
    users.age = users.age / users.age.max()
    users.gender = users.gender.str.replace("M", "1").replace("F", "-1")
    encoder = OneHotEncoder(sparse_output=False)
    encoded_data = encoder.fit_transform(users[['occupation']])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.categories_[0])
    users = pd.concat([users, encoded_df], axis=1)
    users.drop(["occupation", "zip_code", "user_id"], axis=1, inplace=True)

    train_data = pd.read_csv(os.path.join(data_folder, train_name), sep='\t', names=["user_id", "item_id", "rating", "timestamp"])
    test_data = pd.read_csv(os.path.join(data_folder, test_name), sep='\t', names=["user_id", "item_id", "rating", "timestamp"])
    train_data.user_id = train_data.user_id.apply(lambda x: x-1)
    test_data.user_id = test_data.user_id.apply(lambda x: x - 1)
    train_data.item_id = train_data.item_id.apply(lambda x: x-1)
    test_data.item_id = test_data.item_id.apply(lambda x: x - 1)
    train_data.drop("timestamp", axis=1, inplace=True)
    test_data.drop("timestamp", axis=1, inplace=True)

    items.to_csv(os.path.join(save_folder, "films.csv"))
    users.to_csv(os.path.join(save_folder, "users.csv"))
    train_data.to_csv(os.path.join(save_folder, "train.csv"))
    test_data.to_csv(os.path.join(save_folder, "test.csv"))


if __name__ == "__main__":
    preprocess()
