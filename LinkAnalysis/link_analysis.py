import numpy as np
from collections import Counter
from scipy.sparse import csr_matrix
import os
import glob
import datetime


# get p_0 vector
def get_p0(n):
    return np.ones((n, 1)) / n


# read transition matrix from file
# returns a sparse matrix with all connections and indices of dead-end nodes
def read_transition_raw(filename):
    print('Reading Transition matrix...')
    start = datetime.datetime.now()
    # read connections
    with open(filename, 'r') as f:
        lines = f.readlines()
    from_indices = [int(line.split(' ')[0]) - 1 for line in lines]
    to_indices = [int(line.split(' ')[1]) - 1 for line in lines]
    from_index_dict = dict(Counter(from_indices))
    # get n
    max_row = max(from_indices)
    max_col = max(to_indices)
    n = max(max_row, max_col) + 1
    # get transition matrix (without dead-end nodes)
    data = []
    for i in range(len(from_indices)):
        data.append(1 / from_index_dict[from_indices[i]])
    sparse_M = csr_matrix((data, (from_indices, to_indices)), shape=(n, n))
    # record dead-end row indices for efficient computation
    no_link = np.array([i for i in range(n) if sparse_M[i, :].getnnz() == 0])
    end = datetime.datetime.now()
    print('Transition matrix processing completed. Execution time (second):', (end - start).total_seconds())
    return sparse_M, np.array(no_link)


# global page rank algorithm
def global_page_rank(M, no_link, alpha=0.8):
    print('\nStarting Global Page Rank...')
    start = datetime.datetime.now()
    # parameters
    n = M.shape[0]
    p_0 = get_p0(n)
    r = np.random.rand(n, 1)
    r_next = np.ones((n, 1)) / n
    # power iteration
    while np.linalg.norm(r_next - r) > 1e-8:
        r = r_next
        # update r
        r_next = alpha * M.transpose().dot(r) + (1 - alpha) * p_0
        # update calculations from those dead-end nodes separately
        r_next += alpha * np.sum(r[no_link]) * p_0
    end = datetime.datetime.now()
    print('GRP completed. PageRank time (second):', (end - start).total_seconds())
    return r


# page rank for one topic
def topic_page_rank_single(M, no_link, p_t, alpha=0.8, beta=0.19, gamma=0.01):
    # parameters
    n = M.shape[0]
    p_0 = get_p0(n)
    r = np.random.rand(n, 1)
    r_next = np.ones((n, 1)) / n
    # power iteration
    while np.linalg.norm(r_next - r) > 1e-8:
        r = r_next
        # update r
        r_next = alpha * M.transpose().dot(r) + beta * p_t + gamma * p_0
        # update calculations from those dead-end nodes separately
        r_next += alpha * np.sum(r[no_link]) * p_0
    return r


# get page rank vectors for all topics
def topic_page_rank_all(M, no_link, topic_vector):
    r_t_dict = {}
    # for each topics compute its r_t
    for topic in topic_vector.keys():
        p_t = topic_vector[topic]
        r_t = topic_page_rank_single(M, no_link, p_t)
        r_t_dict[topic] = r_t
    return r_t_dict


# weighted sum of each topic PageRank vector based on user preferences
def user_topic_page_rank(M, no_link, topic_vector, user_dist):
    print('\nStarting Personalized Topic Specific Page Rank...')
    start = datetime.datetime.now()
    r_t_dict = topic_page_rank_all(M, no_link, topic_vector)
    r_t_user_dict = {}
    for query_id in user_dist.keys():
        r_u = np.sum([r_t_dict[topic] * user_dist[query_id][topic] for topic in r_t_dict.keys()], axis=0)
        r_t_user_dict[query_id] = r_u
    end = datetime.datetime.now()
    print('PTSPR completed. PageRank time per query (second):', (end - start).total_seconds() / len(user_dist.keys()))
    return r_t_user_dict


# weighted sum of each topic PageRank vector based on queries
def query_topic_page_rank(M, no_link, topic_vector, query_dist):
    print('\nStarting Query-Based Topic Specific Page Rank...')
    start = datetime.datetime.now()
    r_t_dict = topic_page_rank_all(M, no_link, topic_vector)
    r_t_query_dict = {}
    for query_id in query_dist.keys():
        r_q = np.sum([r_t_dict[topic] * query_dist[query_id][topic] for topic in r_t_dict.keys()], axis=0)
        r_t_query_dict[query_id] = r_q
    end = datetime.datetime.now()
    print('QTSPR completed. PageRank time per query (second):', (end - start).total_seconds() / len(query_dist.keys()))
    return r_t_query_dict


# read document-topic connections from file
def read_topic_vector(filename, n):
    with open(filename, 'r') as f:
        lines = f.readlines()
    topic_indices = [int(line.split(' ')[1]) - 1 for line in lines]
    topic_count_dict = Counter(topic_indices)
    topic_vector_dict = {topic: np.zeros((n, 1)) for topic in topic_count_dict.keys()}
    for line in lines:
        doc = int(line.split(' ')[0]) - 1
        topic = int(line.split(' ')[1]) - 1
        p_t = topic_vector_dict[topic]
        p_t[doc, 0] = 1 / topic_count_dict[topic]
        topic_vector_dict[topic] = p_t
    return topic_vector_dict


# read topic distributions based on users or queries
def read_topic_dist(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    user_query_dist = {}
    for line in lines:
        info = line.split(' ')
        u_q = str(info[0]) + '-' + str(info[1])
        dist = {int(info[i].split(':')[0]) - 1: float(info[i].split(':')[1]) for i in range(2, len(info))}
        user_query_dist[u_q] = dist
    return user_query_dist


# real all google-search files
def read_search_file(path):
    list_of_files = glob.glob(path + '/*.txt')
    file_dict = {}
    for file_name in list_of_files:
        query_id = os.path.basename(file_name).replace('.results.txt', '')
        with open(file_name, 'r') as f:
            lines = f.readlines()
            info = {int(line.split(' ')[2]): float(line.split(' ')[4]) for line in lines}
            file_dict[query_id] = info
    return file_dict


# get NS score (only PageRank score) for each document
def get_NS_score(r_vec, doc_id_vec):
    return [(doc_id, float(r_vec[doc_id])) for doc_id in doc_id_vec]


# get WS score (weighted sum between PageRank and Google search scores) for each document
def get_WS_score(r_vec, doc_score_dict, alpha=0.99):
    return [(doc_id, alpha * float(r_vec[doc_id]) + (1 - alpha) * doc_score_dict[doc_id]) for doc_id in doc_score_dict]


# get CM score (customized formula of PageRank and Google search scores) for each document
# weighted sum of the log of PageRank scores and search scores
def get_CM_score(r_vec, doc_score_dict, alpha=0.3):
    return [(doc_id, alpha * float(np.log(r_vec[doc_id])) + (1 - alpha) * (doc_score_dict[doc_id])) for doc_id in doc_score_dict]


# retrieval for GPR: supports three different scoring schemes
def GPR_retrieval(r_global, search_result_dict, mode):
    print('Start computing:', mode)
    start = datetime.datetime.now()
    all_scores = ''
    count = 0
    # sort query ids by ascending order
    query_ids = sorted(list(search_result_dict.keys()), key=lambda x: int(x.replace('-', '')))
    for query_id in query_ids:
        docs = search_result_dict[query_id]
        # compute scores based on different modes and sort by score first and then doc_id
        if mode == 'GPR-NS':
            sorted_doc_scores = sorted(get_NS_score(r_global, docs.keys()), key=lambda x: (-x[1], x[0]))
        elif mode == 'GPR-WS':
            sorted_doc_scores = sorted(get_WS_score(r_global, docs), key=lambda x: (-x[1], x[0]))
        else:
            sorted_doc_scores = sorted(get_CM_score(r_global, docs), key=lambda x: (-x[1], x[0]))
        for rank, doc_tuple in enumerate(sorted_doc_scores):
            doc_id, score = doc_tuple
            score_entry = str(query_id) + ' Q0 ' + str(doc_id) + ' ' + str(rank + 1) + ' ' + str(score) + ' ' + mode + '\n'
            count += 1
            all_scores += score_entry
    with open(mode + '.txt', 'w') as f:
        f.write(all_scores)
    end = datetime.datetime.now()
    print(mode, 'completed. Retrieval time per query (second):', (end - start).total_seconds() / 17885)


# retrieval for QTSPR: supports three different scoring schemes
def QTSPR_retrieval(r_t_query_dict, search_result_dict, mode):
    print('Start computing:', mode)
    start = datetime.datetime.now()
    all_scores = ''
    count = 0
    # sort query ids by ascending order
    query_ids = sorted(list(search_result_dict.keys()), key=lambda x: int(x.replace('-', '')))
    for query_id in query_ids:
        docs = search_result_dict[query_id]
        # compute scores based on different modes and sort by score first and then doc_id
        if mode == 'QTSPR-NS':
            sorted_doc_scores = sorted(get_NS_score(r_t_query_dict[query_id], docs.keys()), key=lambda x: (-x[1], x[0]))
        elif mode == 'QTSPR-WS':
            sorted_doc_scores = sorted(get_WS_score(r_t_query_dict[query_id], docs), key=lambda x: (-x[1], x[0]))
        else:
            sorted_doc_scores = sorted(get_CM_score(r_t_query_dict[query_id], docs), key=lambda x: (-x[1], x[0]))
        for rank, doc_tuple in enumerate(sorted_doc_scores):
            doc_id, score = doc_tuple
            score_entry = str(query_id) + ' Q0 ' + str(doc_id) + ' ' + str(rank + 1) + ' ' + str(score) + ' ' + mode + '\n'
            count += 1
            all_scores += score_entry
    with open(mode + '.txt', 'w') as f:
        f.write(all_scores)
    end = datetime.datetime.now()
    print(mode, 'completed. Retrieval time per query (second):', (end - start).total_seconds() / 17885)


# retrieval for PTSPR: supports three different scoring schemes
def PTSPR_retrieval(r_t_user_dict, search_result_dict, mode):
    print('Start computing:', mode)
    start = datetime.datetime.now()
    all_scores = ''
    count = 0
    # sort query ids by ascending order
    query_ids = sorted(list(search_result_dict.keys()), key=lambda x: int(x.replace('-', '')))
    for query_id in query_ids:
        docs = search_result_dict[query_id]
        # compute scores based on different modes and sort by score first and then doc_id
        if mode == 'PTSPR-NS':
            sorted_doc_scores = sorted(get_NS_score(r_t_user_dict[query_id], docs.keys()), key=lambda x: (-x[1], x[0]))
        elif mode == 'PTSPR-WS':
            sorted_doc_scores = sorted(get_WS_score(r_t_user_dict[query_id], docs), key=lambda x: (-x[1], x[0]))
        else:
            sorted_doc_scores = sorted(get_CM_score(r_t_user_dict[query_id], docs), key=lambda x: (-x[1], x[0]))
        for rank, doc_tuple in enumerate(sorted_doc_scores):
            doc_id, score = doc_tuple
            score_entry = str(query_id) + ' Q0 ' + str(doc_id) + ' ' + str(rank + 1) + ' ' + str(score) + ' ' + mode + '\n'
            count += 1
            all_scores += score_entry
    with open(mode + '.txt', 'w') as f:
        f.write(all_scores)
    end = datetime.datetime.now()
    print(mode, 'completed. Retrieval time per query (second):', (end - start).total_seconds() / 17885)


# save vector values to file
def vector_to_file(r, filename):
    with open(filename, 'w') as f:
        for i, value in enumerate(r, 1):
            f.write(str(i) + ' ' + str(float(value)) + '\n')


# generate sample files
def generate_sample_files(r_global, r_t_query_dict, r_t_user_dict):
    print('\nGenerating sample files...')
    vector_to_file(r_global, 'GPR.txt')
    vector_to_file(r_t_query_dict['2-1'], 'QTSPR-U2Q1.txt')
    vector_to_file(r_t_user_dict['2-1'], 'PTSPR-U2Q1.txt')
    print('Program complete.')


if __name__ == '__main__':
    # get transition matrix (without dead-end nodes) and indices for dead-end nodes
    M, no_link = read_transition_raw('data/transition.txt')
    # search results
    search_result_dict = read_search_file('data/indri-lists')
    # doc-topics
    topic_vector_dict = read_topic_vector('data/doc_topics.txt', M.shape[0])

    # GPR
    r_global = global_page_rank(M, no_link)
    # GPR-NS
    GPR_retrieval(r_global, search_result_dict, mode='GPR-NS')
    # GPR-WS
    GPR_retrieval(r_global, search_result_dict, mode='GPR-WS')
    # GPR-CM
    GPR_retrieval(r_global, search_result_dict, mode='GPR-CM')

    # query based topic distribution
    query_topic_dist = read_topic_dist('data/query-topic-distro.txt')
    # QTSPR
    r_t_query_dict = query_topic_page_rank(M, no_link, topic_vector_dict, query_topic_dist)
    # QTSPR-NS
    QTSPR_retrieval(r_t_query_dict, search_result_dict, mode='QTSPR-NS')
    # QTSPR-WS
    QTSPR_retrieval(r_t_query_dict, search_result_dict, mode='QTSPR-WS')
    # QTSPR-CM
    QTSPR_retrieval(r_t_query_dict, search_result_dict, mode='QTSPR-CM')

    # personalized topic distribution
    user_topic_dist = read_topic_dist('data/user-topic-distro.txt')
    # PTSPR
    r_t_user_dict = user_topic_page_rank(M, no_link, topic_vector_dict, query_topic_dist)
    # PTSPR-NS
    PTSPR_retrieval(r_t_user_dict, search_result_dict, mode='PTSPR-NS')
    # PTSPR-WS
    PTSPR_retrieval(r_t_user_dict, search_result_dict, mode='PTSPR-WS')
    # PTSPR-CM
    PTSPR_retrieval(r_t_user_dict, search_result_dict, mode='PTSPR-CM')

    # generate sample files
    generate_sample_files(r_global, r_t_query_dict, r_t_user_dict)