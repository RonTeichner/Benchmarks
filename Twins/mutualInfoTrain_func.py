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
    Propensity = dataset_dict['Propensity'].values
    all_covariates = pd.DataFrame(dataset_dict.drop(columns=['T', 'y0', 'y1', 'yf', 'y_cf', 'Propensity']))
    X = all_covariates.values
    #proxies = all_covariates[[col for col in all_covariates if col.startswith('gestat')]]
    #true_confounders = all_covariates[[col for col in all_covariates if not col.startswith('gestat')]]

    y0 = Y * (1 - T) + YCF * T
    y1 = Y * T + YCF * (1 - T)
    return X, T, Y, y0, y1, Propensity

def create_sets(X, T, Y, Propensity):
    nSamples = X.shape[0]
    nTrainSamples = int(np.round(0.8 * nSamples))
    trainIndexes = np.sort(np.random.permutation(nSamples)[:nTrainSamples])
    testIndexes = np.setdiff1d(np.arange(0, nSamples), trainIndexes)

    X_train, T_train, Y_train, Propensity_train = X[trainIndexes, :], T[trainIndexes], Y[trainIndexes], Propensity[trainIndexes]
    X_test, T_test, Y_test, Propensity_test  = X[testIndexes, :], T[testIndexes], Y[testIndexes], Propensity[testIndexes]

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

    return scaler, X_train_scaled, X_test_scaled, X_train, T_train, Y_train, T_test, Y_test, Propensity_train, Propensity_test

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
        #x1 = self.LeakyReLU(self.fc21(x0))
        #x2 = self.fc22(x1)
        #x3 = self.fc23(x2)
        #x4 = self.LeakyReLU(self.fc24(x3))
        #x5 = self.fc25(x4)
        return self.fc21(x0)#x2

    def forward(self, x):
        X = x[:, :x.shape[1]-1]
        T = x[:, x.shape[1]-1:x.shape[1]]
        posT, negT = torch.ones_like(T), torch.zeros_like(T)
        posTreatmentOutcomeEst = self.sigmoid(self.encode(torch.cat((X, posT), dim=1)))
        negTreatmentOutcomeEst = self.sigmoid(self.encode(torch.cat((X, negT), dim=1)))
        trueTreatmentOutcomeEst = self.sigmoid(self.encode(torch.cat((X, T), dim=1)))
        return trueTreatmentOutcomeEst, posTreatmentOutcomeEst, negTreatmentOutcomeEst

def loss_function_PS(theta_x_t, theta_x_1, theta_x_0, y, Propensity_batch, enableMi):
    BCE = F.binary_cross_entropy(theta_x_t, y)

    log_sum_x_theta_x_1 = torch.log(theta_x_1.sum())
    log_sum_x_1_minus_theta_x_1 = torch.log((1-theta_x_1).sum())

    log_sum_x_theta_x_0 = torch.log(theta_x_0.sum())
    log_sum_x_1_minus_theta_x_0 = torch.log((1 - theta_x_0).sum())

    a_x_1 = torch.mul(theta_x_1, log_sum_x_theta_x_1) + torch.mul(1 - theta_x_1, log_sum_x_1_minus_theta_x_1)
    a_x_0 = torch.mul(theta_x_0, log_sum_x_theta_x_0) + torch.mul(1 - theta_x_0, log_sum_x_1_minus_theta_x_0)

    alpha_x = torch.mul(Propensity_batch, theta_x_1) + torch.mul(1 - Propensity_batch, theta_x_0)

    log_sum_x_alpha = torch.log(alpha_x.sum())
    log_sum_x_1_minus_alpha = torch.log((1-alpha_x).sum())

    gamma_x_1 = torch.mul(theta_x_1, log_sum_x_alpha) + torch.mul(1-theta_x_1, log_sum_x_1_minus_alpha)
    gamma_x_0 = torch.mul(theta_x_0, log_sum_x_alpha) + torch.mul(1 - theta_x_0, log_sum_x_1_minus_alpha)

    mutualInfo = (torch.mul(a_x_1 - gamma_x_1, Propensity_batch) + torch.mul(a_x_0 - gamma_x_0, 1 - Propensity_batch)).mean()
    #mutualInfo = 0.1*(F.sigmoid(10000*(torch.mul(a_x_1 - gamma_x_1, Propensity_batch) + torch.mul(a_x_0 - gamma_x_0, 1 - Propensity_batch)))).mean()
    #mutualInfo = 0.1*F.sigmoid(220*((torch.mul(a_x_1 - gamma_x_1, Propensity_batch) + torch.mul(a_x_0 - gamma_x_0, 1 - Propensity_batch)).mean()))
    if enableMi:
        loss = BCE - mutualInfo
    else:
        loss = BCE
    return loss, mutualInfo, BCE

def calc_ATT_sLearner(X_train_scaled, X_test_scaled, T_test, Y_test, T_train, Y_train, Propensity_train, Propensity_test, enableMi):
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

    nEpochs = 200 + 1
    trainLoss, trainProbLoss, trainMi, trainBCE, testLoss, testProbLoss = np.zeros(nEpochs), np.zeros(nEpochs), np.zeros(nEpochs), np.zeros(nEpochs), np.zeros(nEpochs), np.zeros(nEpochs)
    X_test_scaled, T_test, Y_test, Propensity_test = torch.tensor(X_test_scaled, dtype=torch.float).cuda(), torch.tensor(T_test, dtype=torch.float).unsqueeze_(-1).cuda(), torch.tensor(Y_test, dtype=torch.float).cuda(), torch.tensor(Propensity_test, dtype=torch.float).cuda()
    X_train_scaled, T_train, Y_train, Propensity_train = torch.tensor(X_train_scaled, dtype=torch.float).cuda(), torch.tensor(T_train, dtype=torch.float).unsqueeze_(-1).cuda(), torch.tensor(Y_train, dtype=torch.float).cuda(), torch.tensor(Propensity_train, dtype=torch.float).cuda()
    nTrainSamples = X_train_scaled.shape[0]

    minTestLoss = np.inf
    for epochIdx in range(nEpochs):
        model_S_learner.train()
        total_loss, total_mI, total_BCE = 0, 0, 0
        batchSize = 50
        nBatches = int(np.ceil(nTrainSamples / batchSize))

        inputIndexes = torch.randperm(nTrainSamples)
        X_train_scaled, T_train, Y_train, Propensity_train = X_train_scaled[inputIndexes], T_train[inputIndexes], Y_train[inputIndexes], Propensity_train[inputIndexes]

        for batchIdx in range(nBatches):
            batchStartIdx, batchStopIdx = batchIdx * batchSize, min(nTrainSamples, (batchIdx + 1) * batchSize)
            # if batchIdx == 0: print('epoch %d: starting batch %d out of %d' % (epochIdx, batchIdx, nBatches))
            data = torch.cat((X_train_scaled[batchStartIdx:batchStopIdx], T_train[batchStartIdx:batchStopIdx]), dim=1)
            label = Y_train[batchStartIdx:batchStopIdx]  # .cuda()
            Propensity_batch = Propensity_train[batchStartIdx:batchStopIdx]

            optimizer.zero_grad()
            trueTreatmentOutcomeEst_train, posTreatmentOutcomeEst_train, negTreatmentOutcomeEst_train = model_S_learner(data)
            loss, mutualInfo, BCE = loss_function_PS(trueTreatmentOutcomeEst_train[:, 0], posTreatmentOutcomeEst_train[:, 0], negTreatmentOutcomeEst_train[:, 0], label, Propensity_batch, enableMi)
            loss.backward()
            total_loss += loss.item()
            total_mI += mutualInfo.item()
            total_BCE += BCE.item()

            optimizer.step()

        trainLoss[epochIdx], trainMi[epochIdx], trainBCE[epochIdx] = total_loss / nBatches, total_mI / nBatches, total_BCE / nBatches
        lr_scheduler.step()

        model_S_learner.eval()
        trueTreatmentOutcomeEst_test, posTreatmentOutcomeEst_test, negTreatmentOutcomeEst_test = model_S_learner(torch.cat((X_test_scaled, T_test), dim=1))
        loss, _, _ = loss_function_PS(trueTreatmentOutcomeEst_test[:, 0], posTreatmentOutcomeEst_test[:, 0], negTreatmentOutcomeEst_test[:,0], Y_test, Propensity_test, enableMi)
        testLoss[epochIdx] = loss.item()

        if testLoss[epochIdx] < minTestLoss:
            minTestLoss = testLoss[epochIdx]
            torch.save(model_S_learner.state_dict(), 'S_learner.pt')

        # print(f'epoch: {epochIdx}: trainLoss: {trainLoss[epochIdx]}; testLoss: {testLoss[epochIdx]}')
        nTotalDifferent = (trueTreatmentOutcomeEst_test[:, 0] - Y_test).abs().sum()
        nTotal = T_test.numel()
        testProbLoss[epochIdx] = nTotalDifferent / nTotal

        trueTreatmentOutcomeEst_train, posTreatmentOutcomeEst_train, negTreatmentOutcomeEst_train = model_S_learner(torch.cat((X_train_scaled, T_train), dim=1))
        nTotalDifferent = (trueTreatmentOutcomeEst_train[:, 0] - Y_train).abs().sum()
        nTotal = T_train.numel()
        trainProbLoss[epochIdx] = nTotalDifferent / nTotal

    n_bins = 50
    plt.figure()
    n, bins, patches = plt.hist(trueTreatmentOutcomeEst_test.detach().cpu().numpy(), n_bins, density=True, histtype='step', cumulative=False, label='est_outcome')
    n, bins, patches = plt.hist(Y_test.detach().cpu().numpy(), n_bins, density=True, histtype='step', cumulative=False, label='true_outcome')
    plt.legend()
    plt.grid(True)
    plt.title('S-learner: True and est outcome hist dataset')
    plt.savefig('Slearner_True_and_est_outcome_hist_dataset')
    #plt.close()
    #plt.show()

    plt.figure()
    plt.plot(trainMi, label='mutual-information')  # train
    plt.plot(trainLoss, label='total loss')   # train
    plt.plot(trainBCE, label='BCE')  # train
    #plt.plot(testLoss, label='test')
    #plt.plot(testProbLoss, label='testProbLoss')
    #plt.plot(trainProbLoss, label='trainProbLoss')
    plt.legend()
    plt.grid(True)
    plt.xlabel('epoch')
    if enableMi:
        plt.title('S-learner: train loss (enabled mutual-info)')
    else:
        plt.title('S-learner: train loss (disabled mutual-info)')
    plt.savefig('Slearner_L1loss_train')
    #plt.close()
    #plt.show()

    #model_S_learner.load_state_dict(torch.load('S_learner.pt'))
    return model_S_learner