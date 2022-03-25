# IHDH for wos 2021/1/8
# @author Jia-Nan Guo

from dotmap import DotMap
import numpy as np
import scipy.io
import pickle
import os
from utils import *
from tqdm import tqdm
import sklearn.preprocessing
from scipy import sparse
import argparse
import random 
from scipy.sparse import coo_matrix
##################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpunum", help="GPU number to train the model.")
parser.add_argument("-d", "--dataset", help="Name of the dataset.")
parser.add_argument("-b", "--nbits", help="Number of bits of the embedded vector.", type=int)
parser.add_argument("--train_batch_size", default=100, type=int)
parser.add_argument("--test_batch_size", default=100, type=int)
parser.add_argument("--transform_batch_size", default=30, type=int)
parser.add_argument("--num_epochs", default=100, type=int)
parser.add_argument("--lr", default=0.0005, type=float)

args = parser.parse_args()

if not args.gpunum:
    parser.error("Need to provide the GPU number.")
    
if not args.dataset:
    parser.error("Need to provide the dataset.")

if not args.nbits:
    parser.error("Need to provide the dataset.")
        
DATASET = args.dataset
data = Load_Dataset("data/{}.mat".format(DATASET))

##################################################################################################

label_binarizer = sklearn.preprocessing.LabelBinarizer()
label_binarizer.fit(range(data.n_tags))

gnd_train = data.gnd_train
gnd_test = data.gnd_test

##################################################################################################

print(gnd_train.shape)
print(gnd_test.shape)
print('num train:{}'.format(data.n_trains))
print('num test:{}'.format(data.n_tests))
# print(data.gnd_test_1l.shape)
# print(data.gnd_test_2l.shape)
# print(data.gnd_train_1l.shape)
# print(data.gnd_train_2l.shape)
# print(data.n_tags_1l)
# print(data.n_tags_2l)

##################################################################################################

import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter

class IHDH(nn.Module):
    
    def __init__(self, vocabSize, tags, tags_1l, tags_2l, latentDim, dropoutProb=0.):
        super(IHDH, self).__init__()
        
        self.hidden_dim = 1000
        self.vocabSize = vocabSize
        self.latentDim = latentDim
        self.tags = tags
        self.tags_1l = tags_1l
        self.tags_2l = tags_2l
        
        self.dtype = torch.cuda.FloatTensor

        self.fc1 = nn.Linear(self.vocabSize, self.hidden_dim)
        torch.nn.init.xavier_normal_(self.fc1.weight, gain=1)

        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        torch.nn.init.xavier_normal_(self.fc2.weight, gain=1)

        self.fc3 = nn.Linear(self.hidden_dim, self.latentDim)
        torch.nn.init.xavier_normal_(self.fc3.weight, gain=1)

        self.dropout = nn.Dropout(p=dropoutProb)
        
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.tanh = nn.Tanhshrink()
        self.eps = 1e-10

        self.fc41 = nn.Linear(self.latentDim, self.vocabSize)
        torch.nn.init.xavier_normal_(self.fc41.weight, gain=1)

        #/ reconst tag 1l and 2l /# 
        self.fc42 = nn.Linear(self.latentDim, self.tags_2l) # 2l
        torch.nn.init.xavier_normal_(self.fc42.weight, gain=1)

        self.fc = nn.Linear(self.latentDim, self.latentDim)
        nn.init.constant_(self.fc.weight, 0.0)

    def encode(self, document_mat, reference_mat, drop=True):
        documents = Variable(torch.from_numpy(document_mat).type(self.dtype))
        references = Variable(torch.from_numpy(reference_mat).type(self.dtype))

        h1_d = self.relu(self.fc1(documents))
        h1_r = self.relu(self.fc1(references))

        h2_d = self.relu(self.fc2(h1_d))
        h2_r = self.relu(self.fc2(h1_r))

        if drop:
            h3_d = self.dropout(h2_d)
            h3_r = self.dropout(h2_r)
        else:
            h3_d = h2_d
            h3_r = h2_r

        x_d0 = self.fc3(h3_d)
        x_r0 = self.fc3(h3_r)

        x_d = self.refer(x_d0, x_r0)
        x_r = self.refer(x_r0, x_d0)

        h_d = torch.sign(x_d)
        h_r = torch.sign(x_r)
        # print(x)
        return x_d, h_d, x_r, h_r 
    
    def decode(self, x):
        word_prob = self.fc41(x)
        y_2l = self.fc42(x)
        return self.log_softmax(word_prob), self.sigmoid(y_2l)

    #/-- update documents according to references --/#
    def refer(self, documents, references): 
        # # cos similarity
        # scores = torch.cosine_similarity(documents, references)
        
        # # scores = torch.exp(documents.mm(references.t()))
        # # scores = F.normalize(scores, p=1, dim=1).diag() 

        # scores_weight = scores.unsqueeze(-1)
        # scores_weight = scores_weight.repeat(1, references.shape[1])
        # # updata documents
        # documents = documents + self.fc(scores_weight * references)

        # documents = documents + self.fc(references)
        
        # attention = torch.div((references-documents), torch.norm((references-documents), 2, 1, True) + self.eps)
        attention = (references-documents)
        documents = documents + self.tanh(self.fc(attention * references))

        return documents

    def union(self, elements):
        
        # temp = elements.sum(1)

        temp = Variable(torch.zeros(elements.shape[0], 1).type(self.dtype))
        num_child = elements.shape[1]
        for i in range(num_child):
            temp[:,0] = temp[:,0] + (1 - temp[:,0]) * elements[:,i]

        return temp[:,0]

    # for wos
    def comp_prob_y_1l(self, prob_y_2l):
        computed_1l  = Variable(torch.ones(prob_y_2l.shape[0], self.tags_1l).type(self.dtype))
        # print(computed_1l.shape)
        computed_1l[:,0] = self.union(prob_y_2l[:,0:17])
        computed_1l[:,1] = self.union(prob_y_2l[:,17:33])
        computed_1l[:,2] = self.union(prob_y_2l[:,33:52])
        computed_1l[:,3] = self.union(prob_y_2l[:,52:61])
        computed_1l[:,4] = self.union(prob_y_2l[:,61:72])
        computed_1l[:,5] = self.union(prob_y_2l[:,72:125])
        computed_1l[:,6] = self.union(prob_y_2l[:,125:])
        return computed_1l

    def forward(self, document_mat, gnd_mat):
        x_d, h_d, x_r, h_r = self.encode(document_mat, gnd_mat)
        prob_w, prob_y_2l = self.decode(x_d)
        prob_y_1l = self.comp_prob_y_1l(prob_y_2l)
        return prob_w, prob_y_1l, prob_y_2l, x_d, h_d, x_r, h_r


def compute_reconstr_loss(log_word_prob, document_mat):
    loss = None
    
    for idx, doc_vec in enumerate(document_mat):
        word_indices = doc_vec.nonzero()
        word_indices = Variable(torch.from_numpy(word_indices[0]).type(torch.cuda.LongTensor))
        pred_logprob = torch.gather(log_word_prob[idx], 0, word_indices)

        if loss is None:
            loss = -torch.sum(pred_logprob) 
        else:
            loss.add_(-torch.sum(pred_logprob))

    return loss / document_mat.shape[0]

def compute_pred_loss(log_word_prob, document_mat):
    document_mat = Variable(torch.from_numpy(document_mat).type(torch.cuda.FloatTensor))

    loss = torch.norm(log_word_prob - document_mat, p=2, dim=1).sum()
    return loss / document_mat.shape[0]

def compute_depend_loss(tag_prob_1l, computed_1l):
    computed_1l = Variable(torch.from_numpy(computed_1l).type(torch.cuda.FloatTensor))
    zeros = Variable(torch.zeros_like(computed_1l).type(torch.cuda.FloatTensor))

    loss = torch.max(zeros, computed_1l - tag_prob_1l).sum()
    return loss / computed_1l.shape[0]

def compute_hash_loss(x, s, k):
    s = Variable(torch.from_numpy(s).type(torch.cuda.FloatTensor))
    loss = torch.norm(torch.mm(x, x.t()) - 2 * k * s + k, p=2, dim=1).sum() # s \in (0, 1)
    return loss / s.shape[0]

def update_references(up_part = True):
    references = np.zeros([data.n_tags, data.n_feas])
    flag = np.array([0 for i in range(data.n_tags)])
    indeies = np.array([i for i in range(data.n_trains)]) 
    np.random.shuffle(indeies)
    if up_part:
        indeies = indeies[0:100] 
    for idx in indeies:
        batch_train = data.train[idx]
        batch_train_gnd = data.gnd_train[idx]

        # cate_index = np.argmax(batch_train_gnd) 
        cate_index = batch_train_gnd.nonzero()[1]

        if flag[cate_index].any() == 0:
            batch_train = batch_train.toarray()
            references[cate_index] = batch_train[0]
            flag[cate_index] = 1
        if min(flag) == 1:
            break
    return references




##################################################################################################

GPU_NUM = args.gpunum
NUM_BITS = args.nbits
TEST_BATCH_SIZE = args.test_batch_size

os.environ["CUDA_VISIBLE_DEVICES"]=GPU_NUM

model = IHDH(data.n_feas, data.n_tags, data.n_tags_1l, data.n_tags_2l, NUM_BITS, dropoutProb=0.1)
print(model)
model.cuda()

def transform(doc_mat, batch_size=500):
    Z = None
    model.eval()
    for idx in range(0, doc_mat.shape[0], batch_size):
        if idx + batch_size < doc_mat.shape[0]:
            batch_train = doc_mat[idx:idx+batch_size]
        else:
            batch_train = doc_mat[idx:]
            
        x, _, _, _ = model.encode(batch_train, batch_train, drop=False) 
        if Z is None:
            Z = x.cpu().data.numpy()
        else:
            Z = np.concatenate((Z, x.cpu().data.numpy()), axis=0)
    return Z

TopK = 100
def run_test():
    model.eval()
    test_loss = 0

    batch_size = args.transform_batch_size
    z_train = transform(data.train.toarray())
    z_test = transform(data.test.toarray())
    
    cbTrain = transform_sign(z_train,0)
    cbTest = transform_sign(z_test,0)
    
    gnd_train = data.gnd_train.toarray()
    gnd_test = data.gnd_test.toarray()

    gnd_train_1l = data.gnd_train_1l.toarray()
    gnd_test_1l = data.gnd_test_1l.toarray()
    
    return topk_results(cbTrain, cbTest, gnd_train_1l, gnd_test_1l, gnd_train, gnd_test, batchSize=TEST_BATCH_SIZE, TopK=100)

##################################################################################################

optimizer = optim.Adam(model.parameters(), lr=args.lr)
# scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma = 0.8) 

BATCH_SIZE = args.train_batch_size
NUM_EPOCHS = args.num_epochs

# quan weight annealing
quanWeight = 0.
quanStepSize = 1 / 1000 
maxQuanWeight = 5.

l1weight = 1.2
l2Weight = 1.

hashWeight = 0.05

drWeight = 1.

predWeight = 0. 
predInc = 0.1
maxPredWeight = 400

BestPrec = 0.
BestRound = 0

references = update_references(False)

for iteration in range(1, NUM_EPOCHS + 1):
    model.train()
    train_loss = []
    # scheduler.step()

    pbar = tqdm(total=data.n_trains, ncols=0)
    for idx in range(0, data.n_trains, BATCH_SIZE):
        if idx + BATCH_SIZE < data.n_trains:
            batch_train = data.train[idx:idx+BATCH_SIZE]
            batch_train_gnd = data.gnd_train[idx:idx+BATCH_SIZE]
            batch_train_gnd_1l = data.gnd_train_1l[idx:idx+BATCH_SIZE]
            batch_train_gnd_2l = data.gnd_train_2l[idx:idx+BATCH_SIZE]
        else:
            batch_train = data.train[idx:]
            batch_train_gnd = data.gnd_train[idx:]
            batch_train_gnd_1l = data.gnd_train_1l[idx:]
            batch_train_gnd_2l = data.gnd_train_2l[idx:]

        batch_train = batch_train.toarray()
        batch_train_gnd = batch_train_gnd.toarray()
        batch_train_gnd_1l = batch_train_gnd_1l.toarray()
        batch_train_gnd_2l = batch_train_gnd_2l.toarray()
        
        optimizer.zero_grad()

        #/-- updata reference --/#
        if random.random() > 0.:
            references = update_references() 

        #/-- according to gnd, building refer_train --/#
        batch_train_gnd_index = np.argmax(batch_train_gnd, axis=1)
        refer_train = np.zeros_like(batch_train)
        for i in range(batch_train.shape[0]):
            index = batch_train_gnd_index[i]
            refer_train[i] = references[index]

        word_prob, tag_prob_1l, tag_prob_2l, x_d, h_d, x_r, h_r = model(batch_train, refer_train)

        s = compute_similarity(batch_train_gnd, batch_train_gnd)

        hash_loss = compute_hash_loss(x_d, s, NUM_BITS) + compute_hash_loss(x_r, s, NUM_BITS)
        reconstr_loss = compute_reconstr_loss(word_prob, batch_train)
        quan_loss = torch.norm(x_d - h_d, p=2, dim=1).sum() / h_d.shape[0] + torch.norm(x_r - h_r, p=2, dim=1).sum() / h_r.shape[0]

        reconstr_loss_gnd_1l = compute_pred_loss(tag_prob_1l, batch_train_gnd_1l)
        reconstr_loss_gnd_2l = compute_pred_loss(tag_prob_2l, batch_train_gnd_2l)

        dr_loss = torch.norm(x_d - x_r, p=2, dim=1).sum() / x_d.shape[0]

        loss = reconstr_loss + quanWeight * quan_loss + predWeight * (l1weight * reconstr_loss_gnd_1l + l2Weight * reconstr_loss_gnd_2l) + hashWeight * hash_loss + drWeight * dr_loss
        
        loss.backward()
        optimizer.step()

        quanWeight = min(quanWeight + quanStepSize, maxQuanWeight)
        predWeight = min(predWeight + predInc, maxPredWeight)

        train_loss.append(loss.item())

        pbar.set_description("{}: IHDH Best Round:{} Prec:{:.4f} AvgLoss:{:.3f} quanWeight:{:.4f} predWeight:{:.1f}"
                             .format(iteration, BestRound, BestPrec, np.mean(train_loss), quanWeight, predWeight))
        pbar.update(len(batch_train))

    pbar.close()
    
    prec, ndcg, ms = run_test()
    BestPrec = max(BestPrec, prec)
    
    if BestPrec == prec:
        BestRound = iteration
