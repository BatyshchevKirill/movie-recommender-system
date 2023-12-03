import argparse

import pandas as pd


def rmse(test: pd.DataFrame, ratings: pd.DataFrame, users: pd.DataFrame):
    """
    Calculate the Root Mean Squared Error (RMSE) between predicted and true ratings.

    Parameters:
    - test (pd.DataFrame): Test dataset containing user-item interactions.
    - ratings (pd.DataFrame): Predicted ratings matrix.
    - users (pd.DataFrame): DataFrame with user-cluster mapping.

    Returns:
    - float: RMSE value.
    """
    error = 0

    for i, item in test.iterrows():
        cluster = str(users.iloc[item["user_id"], 0])
        film = item["item_id"]
        true = item["rating"]
        pred = ratings.at[film, cluster]
        error += (true - pred) ** 2

    error /= len(test)
    return error**0.5


def metrics_single_user(test: pd.DataFrame, ratings: pd.Series):
    """
    Calculate precision and recall for a single user.

    Parameters:
    - test (pd.DataFrame): Test dataset for a single user.
    - ratings (pd.Series): Predicted ratings for items.

    Returns:
    - Tuple[float, float]: Precision and Recall values.
    """
    tp = 0
    fp = 0
    fn = 0

    # Calculate the hits and misses
    for i, row in test.iterrows():
        movie = row["item_id"]
        rating = row["rating"]
        rec = ratings[movie]

        if rating >= 4 and rec >= 3.5:
            tp += 1
        elif rating < 4 and rec >= 3.5:
            fp += 1
        elif rating >= 4 and rec < 3.5:
            fn += 1

    if tp == 0:
        return 0, 0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return precision, recall


def prec_rec(test: pd.DataFrame, users: pd.DataFrame, ratings: pd.DataFrame):
    """
    Calculate average precision and recall over all users.

    Parameters:
    - test (pd.DataFrame): Test dataset containing user-item interactions.
    - users (pd.DataFrame): DataFrame with user-cluster mapping.
    - ratings (pd.DataFrame): Predicted ratings matrix.

    Returns:
    - Tuple[float, float]: Average precision and recall.
    """
    tot_prec = 0
    tot_rec = 0
    tot = 0

    # Calculate metrics for all users present in the test set
    for uid in test["user_id"].unique():
        df = test[test.user_id == uid]
        cluster = users.iloc[uid, 0]
        df_map = ratings.iloc[:, cluster]

        precision, recall = metrics_single_user(df, df_map)

        tot_prec += precision
        tot_rec += recall
        tot += 1

    return tot_prec / tot, tot_rec / tot


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("metric", choices=["rmse", "pr"])
    parser.add_argument(
        "-u",
        "--users",
        default="data/interim/clustered_ae_16_8_100_0_0_0_0000_0_0000.csv",
    )
    parser.add_argument("-t", "--test", default="data/interim/test.csv")
    parser.add_argument("-r", "--ratings", default="data/interim/ratings.csv")
    args = parser.parse_args()

    # Read the data
    users = pd.read_csv(args.users, index_col=0)
    test = pd.read_csv(args.test, index_col=0)
    ratings = pd.read_csv(args.ratings, index_col=0)

    if args.metric == "rmse":
        # Calculate RMSE
        print(
            "Root mean squared error of predictions is:",
            rmse(users=users, test=test, ratings=ratings),
        )
    else:
        # Calculate precision and recall
        prec, rec = prec_rec(users=users, test=test, ratings=ratings)
        print("Metrics of your data:")
        print(f"Precision - {prec: .4f}")
        print(f"Recall    - {rec: .4f}")
