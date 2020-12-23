import numpy as np


class PMF(object):
    """PMF

    :param object:
    """

    def __init__(self, num_factors, num_users, num_movies):
        """__init__

        :param num_factors:
        :param num_users:
        :param num_movies:
        """
        # note that you should not modify this function
        np.random.seed(11)
        self.U = np.random.normal(size=(num_factors, num_users))
        self.V = np.random.normal(size=(num_factors, num_movies))
        self.num_users = num_users
        self.num_movies = num_movies

    def predict(self, user, movie):
        """predict

        :param user:
        :param movie:
        """
        # note that you should not modify this function
        return (self.U[:, user] * self.V[:, movie]).sum()

    def train(self, users, movies, ratings, alpha, lambda_u, lambda_v,
              batch_size, num_iterations):
        """train

        :param users: np.array of shape [N], type = np.int64
        :param movies: np.array of shape [N], type = np.int64
        :param ratings: np.array of shape [N], type = np.float32
        :param alpha: learning rate
        :param lambda_u:
        :param lambda_v:
        :param batch_size:
        :param num_iterations: how many SGD iterations to run
        """
        # modify this function to implement mini-batch SGD
        # for the i-th training instance,
        # user `users[i]` rates the movie `movies[i]`
        # with a rating `ratings[i]`.

        total_training_cases = users.shape[0]
        for i in range(num_iterations):
            start_idx = (i * batch_size) % total_training_cases
            users_batch = users[start_idx:start_idx + batch_size]
            movies_batch = movies[start_idx:start_idx + batch_size]
            ratings_batch = ratings[start_idx:start_idx + batch_size]
            curr_size = ratings_batch.shape[0]

            # TODO: implement your SGD here!!
            for i in range(len(users_batch)):
                user = users_batch[i]
                rating = ratings_batch[i]
                movie = movies_batch[i]
                diff = rating - (self.U[:, user] * self.V[:, movie]).sum()
                self.U[:, user] -= alpha * (-diff * self.V[:, movie] + lambda_u * self.U[:, user])
                self.V[:, movie] -= alpha * (-diff * self.U[:, user] + lambda_v * self.V[:, movie])


# if __name__ == '__main__':
#     user_array = np.array([1, 1, 2, 2, 2, 3, 3, 3, 3])
#     movie_array = np.array([0, 1, 3, 2, 1, 1, 0, 3, 2])
#     ratings_array = np.array([2., 3., 5., 1., 4., 3., 2., 3., 4.])
#     original = np.array([[0, 0, 0, 0], [2., 3., 0, 0], [0, 4., 1., 5.], [2., 3., 4., 3.]])
#     print(original)
#     pmf = PMF(num_factors=2, num_users=4, num_movies=4)
#     pmf.train(user_array, movie_array, ratings_array, alpha=0.1, lambda_u=0.001, lambda_v=0.001, batch_size=3, num_iterations=100)
#     UV = pmf.U.transpose() @ pmf.V
#     print(UV)