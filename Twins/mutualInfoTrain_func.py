import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
import json
import os
import pickle

def read_csv_data():
    dataset_dict = pd.read_csv('Final_data_twins.csv', index_col=[0])
    T = dataset_dict['T'].values
    Y = dataset_dict['yf'].values
    YCF = dataset_dict['y_cf'].values
    all_covariates = pd.DataFrame(dataset_dict.drop(columns=['T', 'y0', 'y1', 'yf', 'y_cf']))
    X = all_covariates.values
    #proxies = all_covariates[[col for col in all_covariates if col.startswith('gestat')]]
    #true_confounders = all_covariates[[col for col in all_covariates if not col.startswith('gestat')]]

    y0 = Y * (1 - T) + YCF * T
    y1 = Y * T + YCF * (1 - T)
    return X, T, Y, y0, y1

def create_sets(X, T, Y):
    nSamples = X.shape[0]
    nTrainSamples = int(np.round(0.8 * nSamples))
    trainIndexes = np.sort(np.random.permutation(nSamples)[:nTrainSamples])
    testIndexes = np.setdiff1d(np.arange(0, nSamples), trainIndexes)

    X_train, T_train, Y_train = X[trainIndexes, :], T[trainIndexes], Y[trainIndexes]
    X_test, T_test, Y_test = X[testIndexes, :], T[testIndexes], Y[testIndexes]

    # scale each covariate:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    print(
        f'If I use the standardScaler correcly then the std of a feature should be 1. The std of a feature: {X_train_scaled[:, 50].std()}')
    '''
    # remove the tails outside 5std's
    goodIndexes = np.arange(X_train.shape[0])
    maxStd = 5
    for featureIdx in range(X_train_scaled.shape[1]):
        featureValues = X_train_scaled[:, featureIdx]
        goodFeatureIndexes = np.intersect1d(np.where(-maxStd < featureValues), np.where(featureValues < maxStd))
        goodIndexes = np.intersect1d(goodIndexes, goodFeatureIndexes)

    print(f'out of {X_train.shape[0]} samples we remained with {goodIndexes.shape[0]} after removing outlayers')
    X_train, T_train, Y_train = X_train[goodIndexes, :], T_train[goodIndexes], Y_train[goodIndexes]
    '''
    # scale each covariate:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return scaler, X_train_scaled, X_test_scaled, X_train, T_train, Y_train, T_test, Y_test

########################################## S-learner #######################################################
class propensityModel(nn.Module):
    def __init__(self, covariateDim):
        super(propensityModel, self).__init__()

        self.covariateDim = covariateDim
        self.internalDim = 1
        # encoder:
        self.fc21 = nn.Linear(self.covariateDim, self.internalDim)
        self.fc22 = nn.Linear(self.internalDim, 1)
        self.fc23 = nn.Linear(self.internalDim, 1)
        #self.fc24 = nn.Linear(self.covariateDim, self.covariateDim)
        #self.fc25 = nn.Linear(self.covariateDim, 1)

        # general:
        self.logSoftMax = nn.LogSoftmax(dim=1)
        self.LeakyReLU = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x0):
        x1 = self.LeakyReLU(self.fc21(x0))
        x2 = self.fc22(x1)
        #x3 = self.fc23(x2)
        #x4 = self.LeakyReLU(self.fc24(x3))
        #x5 = self.fc25(x4)
        return self.fc21(x0)#x2

    def forward(self, x):
        return self.sigmoid(self.encode(x))

def loss_function_PS(recon_x, x):
    BCE = F.binary_cross_entropy(recon_x, x)
    return BCE

def calc_ATT_sLearner(X_train_scaled, X_test_scaled, T_test, Y_test, T_train, Y_train, scaler, X, T):
    model_S_learner = propensityModel(covariateDim=X_train_scaled.shape[1] + 1).cuda()
    trainable_params = filter(lambda p: p.requires_grad, model_S_learner.parameters())
    config = json.load(open('my-config.json'))
    opt_name = config['optimizer']['type']
    opt_args = config['optimizer']['args']
    # opt_args['lr'] = 1e-4
    optimizer = getattr(torch.optim, opt_name)(trainable_params, **opt_args)
    # optimizer = optim.Adam(model_S_learner.parameters(), lr=1e-3)

    lr_name = config['lr_scheduler']['type']
    lr_args = config['lr_scheduler']['args']
    lr_args['step_size'] = 200
    if lr_name == 'None':
        lr_scheduler = None
    else:
        lr_scheduler = getattr(torch.optim.lr_scheduler, lr_name)(optimizer, **lr_args)

    nEpochs = 30 + 1
    trainLoss, trainProbLoss, testLoss, testProbLoss = np.zeros(nEpochs), np.zeros(nEpochs), np.zeros(nEpochs), np.zeros(nEpochs)
    X_test_scaled, T_test, Y_test = torch.tensor(X_test_scaled, dtype=torch.float).cuda(), torch.tensor(T_test, dtype=torch.float).unsqueeze_(-1).cuda(), torch.tensor(Y_test, dtype=torch.float).cuda()
    X_train_scaled, T_train, Y_train = torch.tensor(X_train_scaled, dtype=torch.float).cuda(), torch.tensor(T_train, dtype=torch.float).unsqueeze_(-1).cuda(), torch.tensor(Y_train, dtype=torch.float).cuda()
    nTrainSamples = X_train_scaled.shape[0]

    minTestLoss = np.inf
    for epochIdx in range(nEpochs):
        model_S_learner.train()
        total_loss = 0
        batchSize = 50
        nBatches = int(np.ceil(nTrainSamples / batchSize))

        inputIndexes = torch.randperm(nTrainSamples)
        X_train_scaled, T_train, Y_train = X_train_scaled[inputIndexes], T_train[inputIndexes], Y_train[inputIndexes]

        for batchIdx in range(nBatches):
            batchStartIdx, batchStopIdx = batchIdx * batchSize, min(nTrainSamples, (batchIdx + 1) * batchSize)
            # if batchIdx == 0: print('epoch %d: starting batch %d out of %d' % (epochIdx, batchIdx, nBatches))
            data = torch.cat((X_train_scaled[batchStartIdx:batchStopIdx], T_train[batchStartIdx:batchStopIdx]), dim=1)
            label = Y_train[batchStartIdx:batchStopIdx]  # .cuda()

            optimizer.zero_grad()
            t_recon = model_S_learner(data)
            loss = loss_function_PS(t_recon[:, 0], label)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()

        trainLoss[epochIdx] = total_loss / nBatches
        lr_scheduler.step()

        model_S_learner.eval()
        t_recon = model_S_learner(torch.cat((X_test_scaled, T_test), dim=1))
        loss = loss_function_PS (t_recon[:, 0], Y_test)
        testLoss[epochIdx] = loss.item()

        if testLoss[epochIdx] < minTestLoss:
            minTestLoss = testLoss[epochIdx]
            torch.save(model_S_learner.state_dict(), 'S_learner.pt')

        # print(f'epoch: {epochIdx}: trainLoss: {trainLoss[epochIdx]}; testLoss: {testLoss[epochIdx]}')
        nTotalDifferent = (t_recon[:, 0] - Y_test).abs().sum()
        nTotal = T_test.numel()
        testProbLoss[epochIdx] = nTotalDifferent / nTotal

        t_recon = model_S_learner(torch.cat((X_train_scaled, T_train), dim=1))
        nTotalDifferent = (t_recon[:, 0] - Y_train).abs().sum()
        nTotal = T_train.numel()
        trainProbLoss[epochIdx] = nTotalDifferent / nTotal

    n_bins = 50
    n, bins, patches = plt.hist(t_recon.detach().cpu().numpy(), n_bins, density=True, histtype='step', cumulative=False, label='est_outcome')
    n, bins, patches = plt.hist(Y_test.detach().cpu().numpy(), n_bins, density=True, histtype='step', cumulative=False, label='true_outcome')
    plt.legend()
    plt.grid(True)
    plt.title('S-learner: True and est outcome hist dataset')
    plt.savefig('Slearner_True_and_est_outcome_hist_dataset')
    plt.close()
    #plt.show()

    plt.plot(trainLoss, label='train')
    plt.plot(testLoss, label='test')
    plt.plot(testProbLoss, label='testProbLoss')
    plt.plot(trainProbLoss, label='trainProbLoss')
    plt.legend()
    plt.grid(True)
    plt.xlabel('epoch')
    plt.title('S-learner: L1-loss train')
    plt.savefig('Slearner_L1loss_train')
    plt.close()
    # plt.show()

    model_S_learner.load_state_dict(torch.load('S_learner.pt'))
    model_S_learner.eval()
    X_scaled = scaler.transform(X)
    X_scaled_treated = X_scaled[np.where(T)]

    treatmentEstOutcome = model_S_learner(torch.cat((torch.tensor(X_scaled_treated, dtype=torch.float), torch.ones(X_scaled_treated.shape[0], 1)), dim=1).cuda())
    controlEstOutcome = model_S_learner(torch.cat((torch.tensor(X_scaled_treated, dtype=torch.float), torch.zeros(X_scaled_treated.shape[0], 1)), dim=1).cuda())

    # mu_ATT_sLearner = treatmentEstOutcome.detach().cpu().numpy().mean() - controlEstOutcome.detach().cpu().numpy().mean()

    mu_ATT_sLearner = 0
    return mu_ATT_sLearner