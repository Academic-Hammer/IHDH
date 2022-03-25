import numpy as np
import os
import scipy.io
from dotmap import DotMap
from tqdm import tqdm

from collections import Counter
import math
import torch
################################################################################################################
class MedianHashing(object):
    
    def __init__(self):
        self.threshold = None
        self.latent_dim = None
    
    def fit(self, X):
        self.threshold = np.median(X, axis=0)
        self.latent_dim = X.shape[1]
        
    def transform(self, X):
        assert(X.shape[1] == self.latent_dim)
        binary_code = np.zeros(X.shape)
        for i in range(self.latent_dim):
            binary_code[np.nonzero(X[:,i] < self.threshold[i]),i] = 0
            binary_code[np.nonzero(X[:,i] >= self.threshold[i]),i] = 1
        return binary_code.astype(int)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

def transform_sign(X, threshold):
    latent_dim = X.shape[1]
    binary_code = np.zeros(X.shape)
    for i in range(latent_dim):
        binary_code[np.nonzero(X[:,i] < threshold),i] = 0
        binary_code[np.nonzero(X[:,i] >= threshold),i] = 1
    return binary_code.astype(int)


def compute_similarity(test_categories, train_categories): 

    n_test = test_categories.shape[0]
    n_train = train_categories.shape[0]

    # compute jaccard
    test_categories_bin = np.sign(test_categories)
    train_categories_bin = np.sign(train_categories)

    test_and_train = test_categories_bin.dot(train_categories_bin.T) 
    
    test_sum = test_categories_bin.sum(1)
    test_array = test_sum[:, np.newaxis].repeat(n_train, axis=1)
    train_sum = train_categories_bin.sum(1)
    train_array = train_sum[np.newaxis, :].repeat(n_test, axis=0)

    test_or_train = test_array + train_array - test_and_train

    Jaccard = test_and_train / test_or_train

    # print("1. Compute Jaccard finished !")

    return Jaccard

################################################################################################################
# def Load_Dataset(filename):
#     dataset = scipy.io.loadmat(filename)
#     x_train = dataset['train']
#     x_test = dataset['test']
#     # x_cv = dataset['cv']
#     y_train = dataset['gnd_train']
#     y_test = dataset['gnd_test']
#     # y_cv = dataset['gnd_cv']
    
#     data = DotMap()
#     data.n_trains = y_train.shape[0]
#     data.n_tests = y_test.shape[0]
#     # data.n_cv = y_cv.shape[0]
#     data.n_tags = y_train.shape[1]
#     data.n_feas = x_train.shape[1]

#     ## Convert sparse to dense matricesimport numpy as np
#     train = x_train
#     nz_indices = np.where(np.sum(train, axis=1) > 0)[0]
#     train = train[nz_indices, :]
#     train_len = np.sum(train > 0, axis=1)
#     train_len = np.squeeze(np.asarray(train_len))

#     test = x_test
#     test_len = np.sum(test > 0, axis=1)
#     test_len = np.squeeze(np.asarray(test_len))

#     # if x_cv is not None:
#     #     cv = x_cv
#     #     cv_len = np.sum(cv > 0, axis=1)
#     #     cv_len = np.squeeze(np.asarray(cv_len))
#     # else:
#     #     cv = None
#     #     cv_len = None
        
#     gnd_train = y_train[nz_indices, :]
#     gnd_test = y_test
#     # gnd_cv = y_cv

#     data.train = train
#     data.test = test
#     # data.cv = cv
#     data.train_len = train_len
#     data.test_len = test_len
#     # data.cv_len = cv_len
#     data.gnd_train = gnd_train
#     data.gnd_test = gnd_test
#     # data.gnd_cv = gnd_cv
    
#     return data



def Load_Dataset(filename):
    dataset = scipy.io.loadmat(filename)
    x_train = dataset['train']
    x_test = dataset['test']
    y_train = dataset['gnd_train']
    y_train_1l = dataset['gnd_train1l']
    y_train_2l = dataset['gnd_train2l']
    y_test = dataset['gnd_test']
    y_test_1l = dataset['gnd_test1l']
    y_test_2l = dataset['gnd_test2l']
    
    data = DotMap()
    data.n_trains = y_train.shape[0]
    data.n_tests = y_test.shape[0]
    data.n_tags = y_train.shape[1]
    data.n_tags_1l = y_train_1l.shape[1]
    data.n_tags_2l = y_train_2l.shape[1]
    data.n_feas = x_train.shape[1]

    ## Convert sparse to dense matricesimport numpy as np
    train = x_train
    nz_indices = np.where(np.sum(train, axis=1) > 0)[0]
    train = train[nz_indices, :]
    train_len = np.sum(train > 0, axis=1)
    train_len = np.squeeze(np.asarray(train_len))

    test = x_test
    test_len = np.sum(test > 0, axis=1)
    test_len = np.squeeze(np.asarray(test_len))
        
    gnd_train = y_train[nz_indices, :]
    gnd_train_1l = y_train_1l[nz_indices, :]
    gnd_train_2l = y_train_2l[nz_indices, :]

    gnd_test = y_test
    gnd_test_1l = y_test_1l
    gnd_test_2l = y_test_2l

    data.train = train
    data.test = test

    data.train_len = train_len
    data.test_len = test_len
    
    data.gnd_train = gnd_train
    data.gnd_train_1l = gnd_train_1l
    data.gnd_train_2l = gnd_train_2l

    data.gnd_test = gnd_test
    data.gnd_test_1l = gnd_test_1l
    data.gnd_test_2l = gnd_test_2l
    
    return data

################################################################################################################

class TopDoc(object):
    def __init__(self, data_fn, is_train=False):
        self.data_fn = data_fn
        self.is_train = is_train
        self.db = self.load(data_fn, is_train)
        
    def load(self, fn, is_train):
        db = {}
        with open(fn) as in_data:
            for line in in_data:
                line = line.strip()
                first, rest = line.split(':')

                topk = list(map(int, rest.split(',')))
                
                docId = int(first)
                if is_train:
                    db[docId] = topk[1:]
                else:
                    db[docId] = topk
        return db
    
    def getTopK(self, docId, topK):
        return self.db[docId][:topK]

    def getTopK_Noisy(self, docId, topK, topCandidates):
        candidates = self.db[docId][:topCandidates]
        candidates = np.random.permutation(candidates)
        return candidates[:topK]

###############################################################################################################


def Prec(query_TopK_indeces, gnd_train, gnd_test, TopK):
    
    query_TopK_indeces = query_TopK_indeces.tolist()
    n_test = len(query_TopK_indeces)
    # print(n_test)

    gnd_train = np.sign(gnd_train) 
    gnd_test = np.sign(gnd_test)

    prec = []
    # pbar = tqdm(total=n_test, ncols=0)
    for i in range(n_test):
        received_cate = gnd_test[i] #(7,)
        # print(received_cate.shape)
        # print(received_cate)

        total_cate = gnd_train[list(query_TopK_indeces[i])] #(100,7)

        flag = np.matmul(total_cate, received_cate)

        prec.append(flag.nonzero()[0].shape[0]/flag.shape[0])
    #     pbar.set_description("prec iteration {}".format(i))
    #     pbar.update(1)
    # pbar.close()

    return sum(prec)/len(prec)

def MS(query_TopK_indeces, gnd_train, gnd_test, TopK):
    
    query_TopK_indeces = query_TopK_indeces.tolist()
    n_test = len(query_TopK_indeces)
    # print(n_test)

    gnd_train = np.sign(gnd_train) 
    gnd_test = np.sign(gnd_test)

    mistake = []
    # pbar = tqdm(total=n_test, ncols=0)
    for i in range(n_test):
        received_cate = gnd_test[i]
        # print(received_cate.shape)
        # print(received_cate)

        total_cate = gnd_train[list(query_TopK_indeces[i])]
        # print(total_cate.shape)
        # print(total_cate)

        flag = np.matmul(total_cate, received_cate)

        mistake.append(flag.shape[0] - flag.nonzero()[0].shape[0])
    #     pbar.set_description("prec iteration {}".format(i))
    #     pbar.update(1)
    # pbar.close()

    return mistake


def NDCG(query_TopK_indeces, gnd_train, gnd_test, TopK, weighted=False):
    n_test = len(query_TopK_indeces)
    # print(n_test)

    if not weighted:
        gnd_train = np.sign(gnd_train) 
        gnd_test = np.sign(gnd_test) 

    weight = np.array([math.log(i+1, 2) for i in range(1,TopK+1)])
    weight = weight[::-1].copy()

    gnd_train = torch.cuda.FloatTensor(gnd_train)
    weight = torch.cuda.FloatTensor(weight)
    gnd_test1 = torch.cuda.FloatTensor(gnd_test)

    NDCG = []
    pbar = tqdm(total=n_test, ncols=0)
    for i in range(n_test):
        received_cate = gnd_test[i]
        
        gnd_train1 = torch.min(gnd_test1[i], gnd_train) 
        
        # print(received_cate)
        received_cate = list(np.nonzero(received_cate)[0])
        # print(received_cate)

        gnd_train_this_test = gnd_train1[:,received_cate].sum(1) 

        gnd_TopK = gnd_train_this_test[list(query_TopK_indeces[i])]
        # print(gnd_TopK)
        gnd_train_this_test = gnd_train_this_test.sort()[0]
        gnd_bestK = gnd_train_this_test[-TopK:]
        # print(gnd_bestK)

        DCG = torch.div(gnd_TopK, weight).sum()
        # print(DCG)
        IDCG = torch.div(gnd_bestK, weight).sum()

        NDCG.append(float((DCG/IDCG).cpu().data))
        pbar.set_description("ndcg iteration {}".format(i))
        pbar.update(1)
    pbar.close()

    return sum(NDCG)/len(NDCG)
################################################################################################################

def run_topK_retrieval_experiment_GPU_batch_train(codeTrain, codeTest, gnd_train, gnd_test, batchSize, TopK, mode):
    

    #from tqdm import tqdm_notebook as tqdm
    assert (codeTrain.shape[1] == codeTest.shape[1])
    assert (gnd_train.shape[1] == gnd_test.shape[1])
    assert (codeTrain.shape[0] == gnd_train.shape[0])
    assert (codeTest.shape[0] == gnd_test.shape[0])
    
    n_bits = codeTrain.shape[1]
    n_train = codeTrain.shape[0]
    n_test = codeTest.shape[0]

    topScores = torch.cuda.ByteTensor(n_test, TopK + batchSize).fill_(n_bits+1)
    topIndices = torch.cuda.LongTensor(n_test, TopK + batchSize).zero_()

    testBinmat = torch.cuda.ByteTensor(codeTest).unsqueeze_(2)
    for batchIdx in tqdm(range(0, n_train, batchSize), ncols=0):
        s_idx = batchIdx
        e_idx = min(batchIdx + batchSize, n_train)
        numCandidates = e_idx - s_idx

        batch_codeTrain = codeTrain[s_idx:e_idx].T
        trainBinmat = torch.cuda.ByteTensor(batch_codeTrain).unsqueeze_(0)
        trainBinmat = trainBinmat.expand(testBinmat.size(0), n_bits, trainBinmat.size(2))

        testBinmatExpand = testBinmat.expand_as(trainBinmat)

        scores = (trainBinmat ^ testBinmatExpand).sum(dim=1) #.type(torch.cuda.FloatTensor)
        indices = torch.from_numpy(np.arange(s_idx, e_idx)).cuda().unsqueeze_(0).expand(n_test, numCandidates)

        topScores[:, -numCandidates:] = scores
        topIndices[:, -numCandidates:] = indices

        topScores, newIndices = topScores.sort(dim=1)
        topIndices = torch.gather(topIndices, 1, newIndices)

    # Compute Precision
    Indices = topIndices[:,:TopK]

    if mode == "p":
        mistake_p = MS(Indices, gnd_train, gnd_test, TopK)
        return mistake_p
    elif mode == "l":
        mistake_l = MS(Indices, gnd_train, gnd_test, TopK)
        prec = Prec(Indices, gnd_train, gnd_test, TopK)
        ndcg = NDCG(Indices, gnd_train, gnd_test, TopK)
        return mistake_l, prec, ndcg


def topk_results(codeTrain, codeTest, gnd_train_p, gnd_test_p, gnd_train_l, gnd_test_l, batchSize, TopK):

    mistake_p = run_topK_retrieval_experiment_GPU_batch_train(codeTrain, codeTest, gnd_train_p, gnd_test_p, batchSize, TopK, "p")
    mistake_l, prec, ndcg = run_topK_retrieval_experiment_GPU_batch_train(codeTrain, codeTest, gnd_train_l, gnd_test_l, batchSize, TopK, "l")
    
    ms = (np.array(mistake_p) / (np.array(mistake_l) + 1E-6)).mean()

    print("Prec", prec)
    print("NDCG", ndcg)
    print("MS", ms)
    
    return prec, ndcg, ms
