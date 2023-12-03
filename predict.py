import argparse

import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["eval", "rec"])
    parser.add_argument("user_id")
    parser.add_argument("-u", "--users", default="data/interim/clustered_ae_16_8_100_0_0_0_0000_0_0000.csv")
    parser.add_argument("-r", "--ratings", default="data/interim/ratings.csv")
    parser.add_argument("-m", "--movies", default="data/interim/films.csv")
    parser.add_argument("-f", "--film_id", default=None)
    parser.add_argument("-k", "--top_k", default=5)
    parser.add_argument("-mn", "--min_rating", default=3)
    parser.add_argument("-mx", "--max_rating", default=4.8)
    args = parser.parse_args()

    # Read the data
    users = pd.read_csv(args.users, index_col=0)
    ratings = pd.read_csv(args.ratings, index_col=0)
    cluster = users.iloc[int(args.user_id), 0]
    ratings = ratings.iloc[:, cluster]

    if args.mode == "eval":
        # Predict the rating of a movie
        if args.film_id is None:
            parser.error("Film id should be present for rating prediction mode")
        rating = ratings[int(args.film_id)]
        print(f"The predicted rating for the film is: {rating: .4f}")
    else:
        # Make k recommendations
        ratings = ratings.sort_values(ascending=False)
        ratings = ratings[float(args.min_rating) <= ratings.iloc[:]]
        ratings = ratings[ratings.iloc[:] <= float(args.max_rating)]
        topk = list(ratings.index[:int(args.top_k)])
        films = pd.read_csv(args.movies, index_col=0).loc[topk, 'movie_title'].values.tolist()
        print(f"Here are your recommendations:", *films, sep='\n')



