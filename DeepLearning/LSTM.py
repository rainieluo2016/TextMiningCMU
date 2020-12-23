import datetime
import os
import numpy as np
from collections import Counter
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import nltk

nltk.download('punkt')
nltk.download('stopwords')
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# preprocessing the texts
def preprocess_text(filename):
    f = open(filename, 'r')
    text = f.read()
    text = text.lower()
    # remove punctuations
    text_p = "".join([char for char in text if char not in string.punctuation])
    # tokenization
    words = word_tokenize(text_p)
    # removing stop words
    stop_words = stopwords.words('english')
    filtered_words = [word for word in words if word not in stop_words]
    # stemming
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in filtered_words]
    return stemmed


# read input texts and returns index mappings for words in documents
def read_input(path, word_dict=None, preprocess=True):
    # get paths
    pos_path = path + '/positive/'
    neg_path = path + '/negative/'
    pos_files = os.listdir(pos_path)
    neg_files = os.listdir(neg_path)
    pos_file_list, neg_file_list = [], []
    lengths, words = [], []
    # positive files
    for pos_file in pos_files:
        # option of preprocessing
        if preprocess:
            lst = preprocess_text(pos_path + pos_file)
        else:
            lst = open(pos_path + pos_file, 'r').read().split(' ')
        # get first 128 words
        lst = lst[:128]
        pos_file_list.append(lst)
        lengths.append(len(lst))
        words += lst
    # negative files
    for neg_file in neg_files:
        if preprocess:
            lst = preprocess_text(neg_path + neg_file)
        else:
            lst = open(neg_path + neg_file, 'r').read().split(' ')
        lst = lst[:128]
        neg_file_list.append(lst)
        lengths.append(len(lst))
        words += lst
    # option of using word dictionary (for test set)
    if word_dict is None:
        word_count = Counter(words)
        word_count = [('UNK', 0)] + sorted([(key, value) for key, value in word_count.items()],
                                           key=lambda x: (-x[1], x[0]))[:9999]
        word_dict = {key[0]: i for i, key in enumerate(word_count)}
    # map words to indices for positive and negative
    pos_file_index, neg_file_index = [], []
    for lst in pos_file_list:
        lst = [word_dict.get(word, 0) for word in lst] + [0] * (128 - len(lst))
        pos_file_index.append(lst)
    for lst in neg_file_list:
        lst = [word_dict.get(word, 0) for word in lst] + [0] * (128 - len(lst))
        neg_file_index.append(lst)
    return pos_file_index, neg_file_index, word_dict


# read pre-trained embedding into a dictionary
def read_embedding(embed_file):
    vec_dict = {}
    with open(embed_file, 'r') as f:
        n_words, n_dim = f.readline().split(' ')
        for line in f.readlines():
            lst = line.strip('\n').split(' ')
            word = lst[0]
            vec = [float(num) for num in lst[1:] if len(num) > 0]
            vec_dict[word] = np.array(vec)
    return vec_dict


# Pytorch Dataset
class ReviewDataset(Dataset):
    def __init__(self, lst):
        self.lst = lst

    def __len__(self):
        return len(self.lst)

    def __getitem__(self, idx):
        return self.lst[idx]


# get DataLoader for train or test data
def get_data_loader(pos_data, neg_data, batch_size, mode='train'):
    # positive and negative
    lst = [(torch.LongTensor(data), 1) for data in pos_data]
    lst += [(torch.LongTensor(data), 0) for data in neg_data]
    dataset = ReviewDataset(lst)
    # shuffle for train but not for test
    if mode == 'train':
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# get accuracy to validate
def validate(testloader):
    num_correct = 0
    for i, (batch_data, batch_labels) in enumerate(testloader):
        outputs = model(batch_data)
        pred = torch.argmax(outputs, dim=1)
        current_correct = len([pred[i] for i in range(len(pred)) if pred[i] == batch_labels[i]])
        num_correct += current_correct
    accuracy = num_correct / len(testloader)
    return accuracy


# the model: used bidirectional LSTM
class Net(nn.Module):

    def __init__(self, embedding):
        super(Net, self).__init__()
        self.embedding = embedding
        self.flatten = nn.Flatten()
        self.lstm = nn.LSTM(input_size=100, hidden_size=100, num_layers=1, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(in_features=100 * 2, out_features=2)

    def forward(self, x):
        y = self.embedding(x)
        y, (hidden, cell) = self.lstm(y)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        y = self.fc1(hidden)
        return y


if __name__ == "__main__":
    start = datetime.datetime.now()
    # read files and get data
    train_pos, train_neg, word_dict = read_input('data/train')
    test_pos, test_neg, _ = read_input('data/test', word_dict=word_dict)
    train_loader = get_data_loader(train_pos, train_neg, batch_size=1, mode='train')
    test_loader = get_data_loader(test_pos, test_neg, batch_size=1, mode='test')

    # get embedding
    embedding = torch.nn.Embedding(num_embeddings=10000, embedding_dim=100)

    # model parameters
    model = Net(embedding=embedding)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    criterion = nn.CrossEntropyLoss()
    n_epochs = 3

    # training
    train_accuracies, test_accuracies = [], []
    train_losses, test_losses = [], []
    print('Start training')
    for epoch in range(n_epochs):
        train_loss = 0.0
        for i, (batch_data, batch_labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        # get test loss
        with torch.no_grad():
            test_loss = 0.0
            for i, (batch_data, batch_labels) in enumerate(test_loader):
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                test_loss += loss.item()
            test_losses.append(test_loss / 2000)
            test_loss = 0.0
        # get accuracy
        train_accu = validate(train_loader)
        test_accu = validate(test_loader)
        train_losses.append(train_loss / 2000)
        train_loss = 0.0
        train_accuracies.append(train_accu)
        test_accuracies.append(test_accu)
        scheduler.step()
        epoch_end = datetime.datetime.now()
        print('Epoch {0}: train_accu {1}, test_accu {2}, accumulative time: {3}s'.format((epoch + 1), train_accu, test_accu,
                                                                            (epoch_end - start).total_seconds()))
    print('Finished training')

    # pre-trained
    start = datetime.datetime.now()
    # read files and get data
    train_pos_raw, train_neg_raw, word_dict = read_input('data/train', word_dict=None, preprocess=False)
    test_pos_raw, test_neg_raw, _ = read_input('data/test', word_dict=word_dict, preprocess=False)
    train_loader = get_data_loader(train_pos_raw, train_neg_raw, batch_size=1, mode='train')
    test_loader = get_data_loader(test_pos_raw, test_neg_raw, batch_size=1, mode='test')

    # pretrained embedding
    embedding_dict = read_embedding('data/all.review.vec.txt')
    embedding_dict['</s>'] = np.zeros(embedding_dict['</s>'].shape)
    weights = torch.FloatTensor([embedding_dict.get(word, embedding_dict['</s>']) for word in word_dict.keys()])
    embedding = nn.Embedding.from_pretrained(weights)

    # model parameters
    model = Net(embedding=embedding)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.75)
    criterion = nn.CrossEntropyLoss()
    n_epochs = 3

    # training
    pt_train_accuracies, pt_test_accuracies = [], []
    pt_train_losses, pt_test_losses = [], []
    print('Start training: pre-trained')
    for epoch in range(n_epochs):
        train_loss = 0.0
        for i, (batch_data, batch_labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        # get test loss
        with torch.no_grad():
            test_loss = 0.0
            for i, (batch_data, batch_labels) in enumerate(test_loader):
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                test_loss += loss.item()
            pt_test_losses.append(test_loss / 2000)
            test_loss = 0.0
        pt_train_losses.append(train_loss / 2000)
        train_loss = 0.0
        # get accuracy
        train_accu = validate(train_loader)
        test_accu = validate(test_loader)
        pt_train_accuracies.append(train_accu)
        pt_test_accuracies.append(test_accu)
        scheduler.step()
        epoch_end = datetime.datetime.now()
        print('Epoch {0}: train_accu {1}, test_accu {2}, accumulative time: {3}s'.format((epoch + 1), train_accu, test_accu,
                                                                             (epoch_end - start).total_seconds()))
    print('Finished training: pre-trained')

