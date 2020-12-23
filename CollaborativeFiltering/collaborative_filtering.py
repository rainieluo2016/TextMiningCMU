import numpy as np
import os
from torch.optim.lr_scheduler import StepLR
import torch
import datetime
from scipy.sparse import csr_matrix


# read ratings (training set) from file
def read_ratings(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    movie_ids = [int(line.split(',')[0]) for line in lines]
    user_ids = [int(line.split(',')[1]) for line in lines]
    ratings = [int(line.split(',')[2]) - 3 for line in lines]  # includes imputation Option 2
    m = max(movie_ids) + 1
    n = max(user_ids) + 1
    sparse_matrix = csr_matrix((ratings, (movie_ids, user_ids)), shape=(m, n))
    return sparse_matrix


# get user dot similarity matrix using rating matrix (m * n)
def user_similarity_matrix_dot(rating_matrix):
    A = rating_matrix.transpose().dot(rating_matrix)
    return A.todense()


# get movie dot similarity matrix using rating matrix (m * n)
def item_similarity_matrix_dot(rating_matrix):
    A = rating_matrix.dot(rating_matrix.transpose())
    return A.todense()


# get cosine similarity matrix from dot similarity matrix (divide by norm)
def get_cosine_similarity(dot_similarity_matrix, rating_matrix):
    col_norms = np.linalg.norm(rating_matrix, axis=0)
    col_norms = np.where(col_norms == 0, 1, col_norms)  # avoid zero norms
    return dot_similarity_matrix / col_norms


# read movie-user pairs from file
def read_pairs(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    return [(int(line.split(',')[0]), int(line.split(',')[1])) for line in lines]


# get ratings using user-user similarity
def get_predicted_ratings_memory_based(input_pairs, k, similarity_matrix, rating_matrix, filename, scheme):
    print('Start getting ratings for', filename[16:-4] + '... ', end='')
    start = datetime.datetime.now()
    with open(filename, 'w') as f:
        for movie_id, user_id in input_pairs:
            # get all other users to select KNN
            similarities = similarity_matrix[user_id, :]
            # get all ratings for this movie
            row = rating_matrix[movie_id, :]
            if np.all((similarities == 0)):
                movie_rating = np.mean(row, dtype=np.float64)
            else:
                knn_ids = np.argsort(similarities)[::-1][0, :k]
                # calculate based on different schemes
                if scheme == 'mean':
                    movie_rating = np.mean(row[0, knn_ids], dtype=np.float64)
                else:
                    # get knn weights
                    knn_weights = np.sort(similarities)[::-1][0, :k]
                    knn_weights_sum = np.sum(np.abs(knn_weights))
                    if knn_weights_sum != 0:
                        knn_weights /= knn_weights_sum
                    movie_rating = np.mean((np.multiply(row[0, knn_ids], knn_weights)), dtype=np.float64)
            f.write(str(movie_rating + 3) + '\n')
    end = datetime.datetime.now()
    print('Ratings finished in', (end - start).total_seconds())
    print('Dev RMSE: ', end='')


# get ratings using item-item similarity
def get_predicted_ratings_item_based(input_pairs, k, similarity_matrix, rating_matrix, filename, scheme):
    print('Start getting ratings for', filename[16:-4] + '... ', end='')
    start = datetime.datetime.now()
    with open(filename, 'w') as f:
        for movie_id, user_id in input_pairs:
            # get all other movies to select KNN
            similarities = similarity_matrix[movie_id, :]
            # get all ratings by this user
            row = rating_matrix[user_id, :]
            if np.all((similarities == 0)):
                movie_rating = np.mean(row, dtype=np.float64)
            else:
                knn_ids = np.argsort(similarities)[::-1][0, :k]
                # calculate based on different schemes
                if scheme == 'mean':
                    movie_rating = np.mean(row[0, knn_ids], dtype=np.float64)
                else:
                    # get knn weights
                    knn_weights = np.sort(similarities)[::-1][0, :k]
                    knn_weights_sum = np.sum(np.abs(knn_weights))
                    if knn_weights_sum != 0:
                        knn_weights /= knn_weights_sum
                    movie_rating = np.mean((np.multiply(row[0, knn_ids], knn_weights)), dtype=np.float64)
            f.write(str(movie_rating + 3) + '\n')
    end = datetime.datetime.now()
    print('Ratings finished in', (end - start).total_seconds())
    print('Dev RMSE: ', end='')


# standardization of matrix: subtract by mean and divide by norm
def normalize_matrix(rating_matrix):
    rating_matrix = rating_matrix.astype('float64')
    means = np.mean(rating_matrix, axis=1)
    row_norms = np.linalg.norm(rating_matrix, axis=0)
    row_norms = np.where(row_norms == 0, 1, row_norms)  # avoid zero norms
    rating_matrix -= means
    rating_matrix /= row_norms
    return rating_matrix


# initialize U and V for PMF
def initialize_parameters(n, m, n_latent):
    print('Start for', n_latent, 'latent factors... ', end='')
    U = torch.normal(0.0, 0.01, (n_latent, n))
    V = torch.normal(0.0, 0.01, (n_latent, m))
    return U, V


# loss function in the PMF paper
def pmf_loss(R, U, V, lambda_U, lambda_V):
    UV = torch.matmul(U.t(), V)  # U^T V
    R_UV = (R[R != 0] - UV[R != 0])  # times I_ij: only update those with ratings
    return 0.5 * (torch.sum(torch.matmul(R_UV, R_UV.t())) + lambda_U * torch.sum(torch.matmul(U, U.t()))
                  + lambda_V * torch.sum(torch.matmul(V, V.t())))


# get ratings based on trained U and V
def get_predicted_ratings_pmf(U, V, input_pairs, filename):
    print('Start getting ratings... ', end='')
    start = datetime.datetime.now()
    with open(filename, 'w') as f:
        for movie_id, user_id in input_pairs:
            rating = predict(U, V, user_id, movie_id)
            f.write(str(rating) + '\n')
    end = datetime.datetime.now()
    print('Ratings finished in', (end - start).total_seconds())
    print('Dev RMSE: ', end='')


# train PMF
def train(R, U, V, lambda_U, lambda_V, lr, epoch):
    print('Start training... ', end='')
    start = datetime.datetime.now()
    U.requires_grad_()
    V.requires_grad_()
    # SGD and learning rating scheduling
    optimizer = torch.optim.SGD([U, V], lr, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.55)
    # reduce pmf loss; evaluation is not included because the model is already tuned
    for i in range(epoch):
        optimizer.zero_grad()
        loss = pmf_loss(R, U, V, lambda_U, lambda_V)
        loss.backward()
        optimizer.step()
        scheduler.step()
    end = datetime.datetime.now()
    print('Training finished in', (end - start).total_seconds())
    return U, V


# predict rating using PMF
def predict(U, V, user_id, movie_id):
    r_ij = U[:, user_id].T.reshape(1, -1) @ V[:, movie_id].reshape(-1, 1)
    r_ij = r_ij
    r_ij += 3
    return max(min(r_ij.item(), 5), 1)


if __name__ == '__main__':
    # read from files
    rating_matrix = read_ratings('data/train.csv')  # m * n
    dev_pairs = read_pairs('data/dev.csv')
    test_pairs = read_pairs('data/test.csv')

    dev_users = set([user for movie, user in dev_pairs])
    dev_movies = set([movie for movie, user in dev_pairs])

    # dot product similarity
    user_similarity_matrix = user_similarity_matrix_dot(rating_matrix)
    movie_similarity_matrix = item_similarity_matrix_dot(rating_matrix)

    # dense matrix is faster for getting predictions
    rating_matrix = rating_matrix.todense()

    # cosine similarity
    user_cos_similarity_matrix = get_cosine_similarity(user_similarity_matrix, rating_matrix)
    movie_cos_similarity_matrix = get_cosine_similarity(movie_similarity_matrix, rating_matrix.transpose())

    # Experiment 1: Memory-based
    print('Experiment 1:')
    # mean, dot, k = 10
    get_predicted_ratings_memory_based(dev_pairs, 10, user_similarity_matrix, rating_matrix, 'dev-predictions-dot-10.txt', scheme='mean')
    os.system('python eval_rmse.py data/dev.golden dev-predictions-dot-10.txt')

    # mean, dot, k = 100
    get_predicted_ratings_memory_based(dev_pairs, 100, user_similarity_matrix, rating_matrix, 'dev-predictions-dot-100.txt', scheme='mean')
    os.system('python eval_rmse.py data/dev.golden dev-predictions-dot-100.txt')

    # mean, dot, k = 500
    get_predicted_ratings_memory_based(dev_pairs, 500, user_similarity_matrix, rating_matrix, 'dev-predictions-dot-500.txt', scheme='mean')
    os.system('python eval_rmse.py data/dev.golden dev-predictions-dot-500.txt')

    # mean, cosine, k = 10
    get_predicted_ratings_memory_based(dev_pairs, 10, user_cos_similarity_matrix, rating_matrix, 'dev-predictions-cos-10.txt', scheme='mean')
    os.system('python eval_rmse.py data/dev.golden dev-predictions-cos-10.txt')

    # mean, cosine, k = 100
    get_predicted_ratings_memory_based(dev_pairs, 100, user_cos_similarity_matrix, rating_matrix, 'dev-predictions-cos-100.txt', scheme='mean')
    os.system('python eval_rmse.py data/dev.golden dev-predictions-cos-100.txt')

    # mean, cosine, k = 500
    get_predicted_ratings_memory_based(dev_pairs, 500, user_cos_similarity_matrix, rating_matrix, 'dev-predictions-cos-500.txt', scheme='mean')
    os.system('python eval_rmse.py data/dev.golden dev-predictions-cos-500.txt')

    # weighted, cosine, k = 10
    get_predicted_ratings_memory_based(dev_pairs, 10, user_cos_similarity_matrix, rating_matrix, 'dev-predictions-cos-ws-10.txt', scheme='ws')
    os.system('python eval_rmse.py data/dev.golden dev-predictions-cos-ws-10.txt')

    # weighted, cosine, k = 100
    get_predicted_ratings_memory_based(dev_pairs, 100, user_cos_similarity_matrix, rating_matrix, 'dev-predictions-cos-ws-100.txt', scheme='ws')
    os.system('python eval_rmse.py data/dev.golden dev-predictions-cos-ws-100.txt')

    # weighted, cosine, k = 500
    get_predicted_ratings_memory_based(dev_pairs, 500, user_cos_similarity_matrix, rating_matrix, 'dev-predictions-cos-ws-500.txt', scheme='ws')
    os.system('python eval_rmse.py data/dev.golden dev-predictions-cos-ws-500.txt')

    '''
    Experiment 2: Item-based
    '''
    print('\nExperiment 2:')
    # transpose to n * m for item-based
    rating_matrix = rating_matrix.transpose()
    # mean, dot, k = 10
    get_predicted_ratings_item_based(dev_pairs, 10, movie_similarity_matrix, rating_matrix, 'dev-predictions-dot-10-item.txt', scheme='mean')
    os.system('python eval_rmse.py data/dev.golden dev-predictions-dot-10-item.txt')

    # mean, dot, k = 100
    get_predicted_ratings_item_based(dev_pairs, 100, movie_similarity_matrix, rating_matrix, 'dev-predictions-dot-100-item.txt', scheme='mean')
    os.system('python eval_rmse.py data/dev.golden dev-predictions-dot-100-item.txt')

    # mean, dot, k = 500
    get_predicted_ratings_item_based(dev_pairs, 500, movie_similarity_matrix, rating_matrix, 'dev-predictions-dot-500-item.txt', scheme='mean')
    os.system('python eval_rmse.py data/dev.golden dev-predictions-dot-500-item.txt')

    # mean, cosine, k = 10
    get_predicted_ratings_item_based(dev_pairs, 10, movie_cos_similarity_matrix, rating_matrix, 'dev-predictions-cos-10-item.txt', scheme='mean')
    os.system('python eval_rmse.py data/dev.golden dev-predictions-cos-10-item.txt')

    # mean, cosine, k = 100
    get_predicted_ratings_item_based(dev_pairs, 100, movie_cos_similarity_matrix, rating_matrix, 'dev-predictions-cos-100-item.txt', scheme='mean')
    os.system('python eval_rmse.py data/dev.golden dev-predictions-cos-100-item.txt')

    # mean, cosine, k = 500
    get_predicted_ratings_item_based(dev_pairs, 500, movie_cos_similarity_matrix, rating_matrix, 'dev-predictions-cos-500-item.txt', scheme='mean')
    os.system('python eval_rmse.py data/dev.golden dev-predictions-cos-500-item.txt')

    # weighted, cosine, k = 10
    get_predicted_ratings_item_based(dev_pairs, 10, movie_cos_similarity_matrix, rating_matrix, 'dev-predictions-cos-ws-10-item.txt', scheme='ws')
    os.system('python eval_rmse.py data/dev.golden dev-predictions-cos-ws-10-item.txt')

    # weighted, cosine, k = 100
    get_predicted_ratings_item_based(dev_pairs, 100, movie_cos_similarity_matrix, rating_matrix, 'dev-predictions-cos-ws-100-item.txt', scheme='ws')
    os.system('python eval_rmse.py data/dev.golden dev-predictions-cos-ws-100-item.txt')

    # weighted, cosine, k = 500
    get_predicted_ratings_item_based(dev_pairs, 500, movie_cos_similarity_matrix, rating_matrix, 'dev-predictions-cos-ws-500-item.txt', scheme='ws')
    os.system('python eval_rmse.py data/dev.golden dev-predictions-cos-ws-500-item.txt')

    '''
    Experiment 3: PCC + Item-based
    '''
    print('\nExperiment 3:')
    # transpose back to original (m * n)
    rating_matrix = rating_matrix.transpose()
    # do row-wise normalization (for each movie)
    rating_matrix_pcc = normalize_matrix(rating_matrix)
    # dot similarity
    movie_similarity_matrix_pcc = rating_matrix_pcc.dot(rating_matrix_pcc.transpose())
    # cosine similarity
    movie_cos_similarity_matrix_pcc = get_cosine_similarity(movie_similarity_matrix_pcc, rating_matrix_pcc.transpose())

    # for item-based, transpose to n * m before getting the ratings
    rating_matrix_pcc = rating_matrix_pcc.transpose()
    # mean, dot, k = 10
    get_predicted_ratings_item_based(dev_pairs, 10, movie_similarity_matrix_pcc, rating_matrix_pcc, 'dev-predictions-dot-10-pcc.txt', scheme='mean')
    os.system('python eval_rmse.py data/dev.golden dev-predictions-dot-10-pcc.txt')

    # mean, dot, k = 100
    get_predicted_ratings_item_based(dev_pairs, 100, movie_similarity_matrix_pcc, rating_matrix_pcc, 'dev-predictions-dot-100-pcc.txt', scheme='mean')
    os.system('python eval_rmse.py data/dev.golden dev-predictions-dot-100-pcc.txt')

    # mean, dot, k = 500
    get_predicted_ratings_item_based(dev_pairs, 500, movie_similarity_matrix_pcc, rating_matrix_pcc, 'dev-predictions-dot-500-pcc.txt', scheme='mean')
    os.system('python eval_rmse.py data/dev.golden dev-predictions-dot-500-pcc.txt')

    # mean, cosine, k = 10
    get_predicted_ratings_item_based(dev_pairs, 10, movie_cos_similarity_matrix_pcc, rating_matrix_pcc, 'dev-predictions-cos-10-pcc.txt', scheme='mean')
    os.system('python eval_rmse.py data/dev.golden dev-predictions-cos-10-pcc.txt')

    # mean, cosine, k = 100
    get_predicted_ratings_item_based(dev_pairs, 100, movie_cos_similarity_matrix_pcc, rating_matrix_pcc, 'dev-predictions-cos-100-pcc.txt', scheme='mean')
    os.system('python eval_rmse.py data/dev.golden dev-predictions-cos-100-pcc.txt')

    # mean, cosine, k = 500
    get_predicted_ratings_item_based(dev_pairs, 500, movie_cos_similarity_matrix_pcc, rating_matrix_pcc, 'dev-predictions-cos-500-pcc.txt', scheme='mean')
    os.system('python eval_rmse.py data/dev.golden dev-predictions-cos-500-pcc.txt')

    # weighted, cosine, k = 10
    get_predicted_ratings_item_based(dev_pairs, 10, movie_cos_similarity_matrix_pcc, rating_matrix_pcc, 'dev-predictions-cos-ws-10-pcc.txt', scheme='ws')
    os.system('python eval_rmse.py data/dev.golden dev-predictions-cos-ws-10-pcc.txt')

    # weighted, cosine, k = 100
    get_predicted_ratings_item_based(dev_pairs, 100, movie_cos_similarity_matrix_pcc, rating_matrix_pcc, 'dev-predictions-cos-ws-100-pcc.txt', scheme='ws')
    os.system('python eval_rmse.py data/dev.golden dev-predictions-cos-ws-100-pcc.txt')

    # weighted, cosine, k = 500
    get_predicted_ratings_item_based(dev_pairs, 500, movie_cos_similarity_matrix_pcc, rating_matrix_pcc, 'dev-predictions-cos-ws-500-pcc.txt', scheme='ws')
    os.system('python eval_rmse.py data/dev.golden dev-predictions-cos-ws-500-pcc.txt')

    '''
    Experiment 4: PMF
    '''
    print('\nExperiment 4:')
    # make shape n * m and convert to Tensor
    R = torch.Tensor(rating_matrix.transpose())
    n, m = R.shape

    # latent = 2
    U, V = initialize_parameters(n=n, m=m, n_latent=2)
    U_trained, V_trained = train(R, U, V, lambda_U=0.001, lambda_V=0.001, lr=0.001, epoch=50)
    get_predicted_ratings_pmf(U_trained, V_trained, dev_pairs, 'dev-predictions-pmf-2.txt')
    os.system('python eval_rmse.py data/dev.golden dev-predictions-pmf-2.txt')

    # latent = 5
    U, V = initialize_parameters(n=n, m=m, n_latent=5)
    U_trained, V_trained = train(R, U, V, lambda_U=0.001, lambda_V=0.001, lr=0.001, epoch=50)
    get_predicted_ratings_pmf(U_trained, V_trained, dev_pairs, 'dev-predictions-pmf-5.txt')
    os.system('python eval_rmse.py data/dev.golden dev-predictions-pmf-5.txt')

    # latent = 10
    U, V = initialize_parameters(n=n, m=m, n_latent=10)
    U_trained, V_trained = train(R, U, V, lambda_U=0.001, lambda_V=0.001, lr=0.001, epoch=50)
    get_predicted_ratings_pmf(U_trained, V_trained, dev_pairs, 'dev-predictions-pmf-10.txt')
    os.system('python eval_rmse.py data/dev.golden dev-predictions-pmf-10.txt')

    # latent = 20
    U, V = initialize_parameters(n=n, m=m, n_latent=20)
    U_trained, V_trained = train(R, U, V, lambda_U=0.001, lambda_V=0.001, lr=0.001, epoch=50)
    get_predicted_ratings_pmf(U_trained, V_trained, dev_pairs, 'dev-predictions-pmf-20.txt')
    os.system('python eval_rmse.py data/dev.golden dev-predictions-pmf-20.txt')

    # latent = 50
    print('This one could take a while')
    U, V = initialize_parameters(n=n, m=m, n_latent=50)
    U_trained, V_trained = train(R, U, V, lambda_U=0.001, lambda_V=0.001, lr=0.001, epoch=100)
    get_predicted_ratings_pmf(U_trained, V_trained, dev_pairs, 'dev-predictions-pmf-50.txt')
    os.system('python eval_rmse.py data/dev.golden dev-predictions-pmf-50.txt')
    get_predicted_ratings_pmf(U_trained, V_trained, test_pairs, 'test-predictions.txt')