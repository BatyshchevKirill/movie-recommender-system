# Introduction
Recommendation systems play a pivotal role in enhancing user experience 
and engagement across various online platforms. In the field of recommendation
systems the MovieLens dataset stands as a benchmark for evaluating and
developing advanced recommendation algorithms. This dataset comprises 
a rich collection of user ratings for movies, offering insights into 
user preferences and behaviors. As an integral part of the broader
field of machine learning and data science, recommendation systems 
aim to predict user preferences and suggest personalized content, 
fostering a more tailored and enjoyable experience. In this assignment,
we used the MovieLens 100K dataset [1], exploring its structure, 
characteristics, and the application of recommendation algorithms.
Through this analysis, we aim to gain a deeper understanding of the 
challenges and opportunities inherent in recommendation systems.
# Data analysis
The movie lens dataset consists of 943 users, 1682 movies, and
100000 ratings. <br>
The movies are represented by id, title,
release date, video_release_date, IMDb_URL, and 19 encoded genres.
The video_release_date was None for all films, and url does not have
any helpful information, so I dropped them. I decided to leave the title to
create a more user-friendly representation of recommendations. The dates were
distributed as following: <br>![Fig. 1. Release dates distribution](figures/release_date.png)
<br>In the genre distribution we can see a shift towards drama and comedy genres:
<br>![Fig. 2 Genre distribution](figures/genredistribution.png)<br>
Now let us explore a little bit the users' demographic info: it consists of 
id, age, gender, occupation (one of 21), and zipcode. The zipcode column primary 
contains unique values, so it is not helpful for our model, and it was dropped.
The gender consists of 2 genders (in correspondence with Russian laws),
male and female, with domination of male raters: <br>
![Fig. 3. Gender distribution](figures/gender.png)<br>This information was encoded 
binary, 1 for male, 0 for female. Now let us take a look at occupation distribution
<br>![Fig. 4. Occupation distribution](figures/occupation%20distribution.png)<br>
We can see that several occupations dominate over others, while some of them 
are underrepresented and can negatively affect the quality of learning. The occupation
info was encoded using OneHotEncoder. The next
thing to consider is gender distribution. 
<br>![Fig. 5. Age distribution](figures/age%20distribution.png)<br>
The age is distributed more or less evenly in range from ~15 to ~55 with an outlier
in ~30 years, ages less and more are represented poorly.

Finally, let's take a look at rating distribution.
<br>![Fig. 6. Rating distribution](figures/rating%20distribution.png)<br>
We can see that most of the ratings are somehow positive, so users tend to
like the movies they watch (the task becomes a bit simpler). These pictures show
how many movies were rated by each user and how many grades has received each movie:
<br>![Fig. 7. Users rated](figures/user_ratings.png)
![Fig. 8. Movies rated](figures/movie_raters.png)
<br>
The pictures show, that there are many movies that were rated by a small amount of users.
This may become (it has become, actually) a bottleneck, because we do not have enough
info about the movies to know what kind of users like them

# Model Implementation
To choose the model for the task I used the benchmarking website [2]. I
have chosen the current state-of-the-art model [3]. It was implemented in [4], 
so some of the elements were taken from there. Also, I compared the metrics with
this implementation, because the authors did not share their code, and how they achieved their
metrics is unclear. <br>
The model scheme is the following:
1. The users are connected with each other in a graph if they have more than
alpha similar ratings for different movies.
2. The following metrics of the received graph are calculated for each user:
- Pagerank
- Degree centrality
- Closeness centrality
- Betweenness centrality
- Load centrality
- Average neighbour degree
3. The data is concatenated with the users' demographic information.
4. The user information is passed to a 5 layer autoencoder to reduce the
dimensionality of the data.
5. The resulting user data is clustered using k-means algorithm
6. For each film its average rating is calculated within each cluster.
If no users watched a film within a cluster, then we search for similar
films. Authors do not specify the way how they define the similarity of
movies, but I assumed they used no complex algorithm and measured the 
similarity based on the genre and release date information.
7. Users rating of a movie is measured as a rating of the user's cluster
in the generated lookup table. 
8. If recommendations are to be made, we sort the movies by rating within
a cluster, and recommend top k movies (excluding ones with too high rating
because these ratings are likely to be made by too few users)

# Training Process
The training process included the following steps:
- Generating the user graph data. For this I used alpha=0.01 proposed in the
article and networkx library to extract the features. The demographic features
were encoded or scaled. 
- Training the autoencoder for dimensionality reduction. The architecture
of the encoder part was the following 32 -> 16 -> ReLU -> 8 -> ReLU. I
trained it with the recommended hyperparameters: 100 epochs (even though the
loss stopped dropping after the first one), Adam optimizer with starting
learning rate of 0.01, and MeanSquaredError as a loss. 
- After that k-means algorithm with k = 10 was trained on the encoded data.
- Then the ratings were calculated for each movie for each cluster. The table
was saved and used for predictions in the future.
# Model Advantages and Disadvantages
## Advantages:
- The model is the current state-of-the-art for the dataset
- It utilizes the demographic information about users as well as the information about ratings
- The model can deal with unknown users and items
- The training complexity is low
## Disadvantages:
- The actual results are not perfect
- Insertion of new users is costly as it requires recalculating metrics on the graph
- The model poorly deals with the movies that are not well rated by the people within a cluster
# Evaluation
To measure the goodness of my model's predictions I used the following metrics:
- **Root Mean Squared Error** for evaluating rating prediction
- **Precision** and **recall** for measuring the quality of recommendation. Note
that as test set does not allow us to measure the true recall and precision, 
I measured the metrics based on whether test items are likely to be recommended to
a user or not. 

My results for the metrics are the following:<br>

|           | RMSE  | Precision | Recall |
|-----------|-------|-----------|--------|
| Train     | 0.922 | 0.7202    | 0.7575 |
| Test      | 1.106 | 0.6749    | 0.6945 |

Approximate RMSE for an average guess was 1.35, so the model is performing good enough.
However, we can see that the result for training set is much better than the result
in testing set. The main difference between the training and testing sets is that
there are films that were not rated by a cluster in a train set, but occur in a test set.
So, I suggest that the problem of the metric gap is that the approach of computing
similarity of unrated films with rated ones is not efficient enough. Maybe we should
use more complex methods (embeddings, for example) for finding similar movies.
# Results
The results are good, they are significantly better than random, approximately
the same with another implementation [4] (1.10, 0.67, 0.72), and reasonably worse than ones
stated in the article [3] (0.89, 0.77, and 0.8). So, overall the implementation of recommendation system
on the movielens dataset can be considered successfull

Future work:
- Implementing a more efficient way of dealing with unknown movies, including not recommending them at all
- Trying different methods for dimensionality reduction. Trying to run the clustering
algorithm without dimensionality reduction
- Implementing my own GridSearch, not using the suggested hyperparameters

Summing up, in this work I successfully implemented a movie recommendation system
based on the previous ratings and user demographic data. 
The dynamic nature of user interactions with movie content highlights the need for 
adaptive approaches to recommendation systems. Moving forward, leveraging
advancements in ML and DL will be crucial in further enhancing the accuracy and
personalization capabilities of RecSys, ensuring they remain helpful
in delivering curated and enjoyable content experiences for users.
