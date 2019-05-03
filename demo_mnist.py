#!/bin/sh
# Pytorch implementation of the paper Bayesian Generative Active Deep Learning based on a VAE-ACGAN model.
# We thank for the help from
# https://github.com/znxlwm/pytorch-generative-model-collections
# and
# https://github.com/fahadm/Bayesian-Active-Learning-Pytorch

# Requirements: Python 2.7 & Pytorch 0.3

# To execute the code:
# python demo_mnist.py VAR_RATIOS


from __future__ import print_function
import sys
import argparse, os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from scipy.misc import imread

from scipy import linalg
from torch.autograd import Variable
from torch.nn.functional import adaptive_avg_pool2d

import torch.utils.data as data_utils
from scipy.stats import mode
import utils, time, os, pickle
from torch.optim import lr_scheduler

cuda = True

batch_size = 100
input_dim, input_height, input_width = 1, 28, 28
nb_classes = 10

nb_filters = 32
nb_pool = 2
nb_conv = 4

lr = 0.01
momentum = 0.9
log_interval = 100
epochs = 50

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

# Load training data (60,000 samples)
train_loader_all = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)

# Load test data (10,000 samples)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)


# The whole training data set is split into original and pool data sets

def prepare_data():
    train_data_all = train_loader_all.dataset.train_data
    train_target_all = train_loader_all.dataset.train_labels
    shuffler_idx = torch.randperm(train_target_all.size(0))
    train_data_all = train_data_all[shuffler_idx]
    train_target_all = train_target_all[shuffler_idx]

    test_data = test_loader.dataset.test_data
    test_target = test_loader.dataset.test_labels


    train_data = []
    train_target = []

    train_data_pool = train_data_all[15000:60000, :, :]
    train_target_pool = train_target_all[15000:60000]

    # train_data_all = train_data_all[0:10000,:,:]
    # train_target_all = train_target_all[0:10000]

    train_data_pool.unsqueeze_(1)
    train_data_all.unsqueeze_(1)
    test_data.unsqueeze_(1)

    train_data_pool = train_data_pool.float()

    train_data_all = train_data_all.float()

    test_data = test_data.float()

    for i in range(0, 10):
        arr = np.array(np.where(train_target_all.numpy() == i))
        idx = np.random.permutation(arr)
        data_i = train_data_all.numpy()[idx[0][0:10], :, :, :]  # pick the first 10 elements of the shuffled idx array
        target_i = train_target_all.numpy()[idx[0][0:10]]
        train_data.append(data_i)
        train_target.append(target_i)

    train_data = np.concatenate(train_data, axis=0).astype("float32")
    train_target = np.concatenate(train_target, axis=0)

    return torch.from_numpy(train_data / 255).float(), torch.from_numpy(train_target), \
           train_data_pool / 255, train_target_pool, \
           test_data / 255, test_target

train_data, train_target, pool_data, pool_target, test_data, test_target = prepare_data()


train_loader = None


# Initialize the training data

def initialize_train_set():
    # Training Data set
    global train_loader
    global train_data
    train = data_utils.TensorDataset(train_data, train_target)
    train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)

initialize_train_set()


# Build a classifier

class Net_Correct(nn.Module):
    def __init__(self, input_shape=(input_dim, input_width, input_height)):
        super(Net_Correct, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, nb_filters, kernel_size=nb_conv),
            nn.ReLU(),
            nn.Conv2d(nb_filters, nb_filters, kernel_size=nb_conv),
            nn.ReLU(),
            nn.MaxPool2d(nb_pool),
            nn.Dropout2d(0.25),
        )

        input_size = self._get_conv_output_size(input_shape)

        self.dense = nn.Sequential(nn.Linear(input_size, 128))

        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, nb_classes)
        )

    def _get_conv_output_size(self, shape):
        bs = batch_size
        input = Variable(torch.rand(bs, *shape))
        output_feat = self.conv(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(self.dense(x))
        return x


class lenet(nn.Module):  
    def __init__(self):
        super(lenet, self).__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.input_dim = input_dim
        self.class_num = 10

        self.conv1 = nn.Conv2d(self.input_dim, 6, (5, 5), padding=2)
        self.conv2 = nn.Conv2d(6, 16, (5, 5))
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.class_num)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


model = None
optimizer = None
model_scheduler = None


def train(epoch):
    model.train()
    loss = None
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        output = model(data)
        criterion = nn.CrossEntropyLoss()

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    if epoch or epochs:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))

    return loss.item()


def evaluate(input_data, stochastic=False, predict_classes=False):

    if stochastic:
        model.train()  # we use dropout at test time
    else:
        model.eval()

    predictions = []
    test_loss = 0
    correct = 0
    for data, target in input_data:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)

        output = model(data)

        softmaxed = F.softmax(output.cpu())

        if predict_classes:
            predictions.extend(np.argmax(softmaxed.data.numpy(), axis=-1))
        else:
            predictions.extend(softmaxed.data.numpy())
        criterion = nn.CrossEntropyLoss()

        loss = criterion(output, target)

        test_loss += loss.item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        pred = pred.eq(target.data).cpu().data.float()
        correct += pred.sum()
    return test_loss, correct, predictions


best_acc = 0

def test(epoch):
    global train_loader
    global train_data
    global best_acc
    test = data_utils.TensorDataset(test_data, test_target)
    test_loader = data_utils.DataLoader(test, batch_size=batch_size, shuffle=True)

    test_loss = 0
    correct = 0

    test_loss, correct, _ = evaluate(test_loader, stochastic=False)

    test_loss /= len(test_loader)  # loss function already averages over batch size
    test_acc = 100. * correct / len(test_loader.dataset)

    if epoch or epochs:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_acc))

    if test_acc > best_acc:
        print('Saving...')
        state = {
            'net': model.state_dict(),
            'acc': test_acc,
            'epoch': epoch,
        }
        best_acc = test_acc

    return test_loss, best_acc


def getAcquisitionFunction(name):
    if name == "BALD":
        return bald_acquisition
    elif name == "VAR_RATIOS":
        return variation_ratios_acquisition
    elif name == "MAX_ENTROPY":
        return max_entroy_acquisition
    elif name == "MEAN_STD":
        return mean_std_acquisition
    else:
        print ("ACQUSITION FUNCTION NOT IMPLEMENTED")
        sys.exit(-1)


def acquire_points(argument, random_sample=False):
    global train_data
    global train_target
    global model
    global optimizer
    global model_scheduler

    acquisition_iterations = 100
    dropout_iterations = 20  # [50, 100, 500, 1000]
    Queries = 100
    nb_samples = 100
    pool_all = np.zeros(shape=(1))

    if argument == "RANDOM":
        random_sample = True
    else:
        acquisition_function = getAcquisitionFunction(argument)

    test_acc_hist = []

    for i in range(acquisition_iterations):
        pool_subset = 2000
        if random_sample:
            pool_subset = Queries
        print('---------------------------------')
        print ("Acquisition Iteration " + str(i))
        pool_subset_dropout = torch.from_numpy(np.asarray(random.sample(range(0, pool_data.size(0)), pool_subset)))
        pool_data_dropout = pool_data[pool_subset_dropout]
        pool_target_dropout = pool_target[pool_subset_dropout]
        if random_sample is True:
            pool_index = np.array(range(0, Queries))

        else:
            points_of_interest = acquisition_function(dropout_iterations, pool_data_dropout, pool_target_dropout)
            pool_index = points_of_interest.argsort()[-Queries:][::-1]

        pool_index = torch.from_numpy(np.flip(pool_index, axis=0).copy())

        pool_all = np.append(pool_all, pool_index)

        pooled_data = pool_data_dropout[pool_index]  # LongTensor
        pooled_target = pool_target_dropout[pool_index]  # LongTensor

        train_data = torch.cat((train_data, pooled_data), 0)
        train_target = torch.cat((train_target, pooled_target), 0)

        #remove from pool set
        remove_pooled_points(pool_subset, pool_data_dropout, pool_target_dropout, pool_index)

        # Train the ACGAN here

        gan = VAEACGAN()
        test_acc = gan.train()
        test_acc_hist.append(test_acc)
        # gan.visualize_results(epochs)

    np.save("./test_acc_VAEACGAN_MNIST" + argument + ".npy", np.asarray(test_acc_hist))


def bald_acquisition(dropout_iterations, pool_data_dropout, pool_target_dropout):
    print ("BALD ACQUISITION FUNCTION")
    score_all = np.zeros(shape=(pool_data_dropout.size(0), nb_classes))
    all_entropy = np.zeros(shape=pool_data_dropout.size(0))

    # Validation Dataset
    pool = data_utils.TensorDataset(pool_data_dropout, pool_target_dropout)
    pool_loader = data_utils.DataLoader(pool, batch_size=batch_size, shuffle=True)
    start_time = time.time()
    for d in range(dropout_iterations):
        _, _, scores = evaluate(pool_loader, stochastic=True)

        scores = np.array(scores)
        #predictions = np.expand_dims(predictions, axis=1)
        score_all = score_all + scores

        log_score = np.log2(scores)
        entropy = - np.multiply(scores, log_score)
        entropy_per_dropout = np.sum(entropy, axis =1)
        all_entropy = all_entropy + entropy_per_dropout

    print("Dropout Iterations took --- %s seconds ---" % (time.time() - start_time))
    # print (All_Dropout_Classes)
    avg_pi = np.divide(score_all, dropout_iterations)
    log_avg_pi = np.log2(avg_pi)
    entropy_avg_pi = - np.multiply(avg_pi, log_avg_pi)
    entropy_average_pi = np.sum(entropy_avg_pi, axis=1)

    g_x = entropy_average_pi
    average_entropy = np.divide(all_entropy, dropout_iterations)
    f_x = average_entropy

    u_x = g_x - f_x


    # THIS FINDS THE MINIMUM INDEX
    # a_1d = U_X.flatten()
    # x_pool_index = a_1d.argsort()[-Queries:]

    points_of_interest = u_x.flatten()
    return points_of_interest


def max_entroy_acquisition(dropout_iterations, pool_data_dropout, pool_target_dropout):
    print("MAX ENTROPY FUNCTION")
    score_All = np.zeros(shape=(pool_data_dropout.size(0), nb_classes))

    # Validation Dataset
    pool = data_utils.TensorDataset(pool_data_dropout, pool_target_dropout)
    pool_loader = data_utils.DataLoader(pool, batch_size=batch_size, shuffle=True)
    start_time = time.time()
    for d in range(dropout_iterations):
        _, _, predictions = evaluate(pool_loader, stochastic=True)

        predictions = np.array(predictions)
        #predictions = np.expand_dims(predictions, axis=1)
        score_All = score_All + predictions
    print("Dropout Iterations took --- %s seconds ---" % (time.time() - start_time))
    # print (All_Dropout_Classes)
    Avg_Pi = np.divide(score_All, dropout_iterations)
    Log_Avg_Pi = np.log2(Avg_Pi)
    Entropy_Avg_Pi = - np.multiply(Avg_Pi, Log_Avg_Pi)
    Entropy_Average_Pi = np.sum(Entropy_Avg_Pi, axis=1)

    U_X = Entropy_Average_Pi

    points_of_interest = U_X.flatten()
    return  points_of_interest


def mean_std_acquisition(dropout_iterations, pool_data_dropout, pool_target_dropout):
    print("MEAN STD ACQUISITION FUNCTION")
    all_dropout_scores = np.zeros(shape=(pool_data_dropout.size(0), 1))
    # Validation Dataset
    pool = data_utils.TensorDataset(pool_data_dropout, pool_target_dropout)
    pool_loader = data_utils.DataLoader(pool, batch_size=batch_size, shuffle=True)
    start_time = time.time()
    for d in range(dropout_iterations):
        _, _, scores = evaluate(pool_loader, stochastic=True)

        scores = np.array(scores)
        all_dropout_scores = np.append(all_dropout_scores, scores, axis=1)
    print("Dropout Iterations took --- %s seconds ---" % (time.time() - start_time))
    std_devs= np.zeros(shape = (pool_data_dropout.size(0), nb_classes))
    sigma = np.zeros(shape = (pool_data_dropout.size(0)))
    for t in range(pool_data_dropout.size(0)):
        for r in range( nb_classes ):
            L = np.array([0])
            for k in range(r + 1, all_dropout_scores.shape[1], 10):
                L = np.append(L, all_dropout_scores[t, k])

            L_std = np.std(L[1:])
            std_devs[t, r] = L_std
        E = std_devs[t, :]
        sigma[t] = sum(E)/nb_classes


    points_of_interest = sigma.flatten()
    return points_of_interest


def variation_ratios_acquisition(dropout_iterations, pool_data_dropout, pool_target_dropout):
    # print("VARIATIONAL RATIOS ACQUSITION FUNCTION")
    All_Dropout_Classes = np.zeros(shape=(pool_data_dropout.size(0), 1))
    # Validation Dataset
    pool = data_utils.TensorDataset(pool_data_dropout, pool_target_dropout)
    pool_loader = data_utils.DataLoader(pool, batch_size=batch_size, shuffle=True)
    start_time = time.time()
    for d in range(dropout_iterations):
        _, _, predictions = evaluate(pool_loader, stochastic=True, predict_classes=True)

        predictions = np.array(predictions)
        predictions = np.expand_dims(predictions, axis=1)
        All_Dropout_Classes = np.append(All_Dropout_Classes, predictions, axis=1)
    # print("Dropout Iterations took --- %s seconds ---" % (time.time() - start_time))
    # print (All_Dropout_Classes)
    Variation = np.zeros(shape=(pool_data_dropout.size(0)))
    for t in range(pool_data_dropout.size(0)):
        L = np.array([0])
        for d_iter in range(dropout_iterations):
            L = np.append(L, All_Dropout_Classes[t, d_iter + 1])
        Predicted_Class, Mode = mode(L[1:])
        v = np.array([1 - Mode / float(dropout_iterations)])
        Variation[t] = v
    points_of_interest = Variation.flatten()
    return points_of_interest


def remove_pooled_points(pool_subset, pool_data_dropout, pool_target_dropout, pool_index):
    global pool_data
    global pool_target
    np_data = pool_data.numpy()
    np_target = pool_target.numpy()
    pool_data_dropout = pool_data_dropout.numpy()
    pool_target_dropout = pool_target_dropout.numpy()
    np_index = pool_index.numpy()
    np.delete(np_data, pool_subset, axis=0)
    np.delete(np_target, pool_subset, axis=0)

    np.delete(pool_data_dropout, np_index, axis=0)
    np.delete(pool_target_dropout, np_index, axis=0)

    np_data = np.concatenate((np_data, pool_data_dropout), axis=0)
    np_target = np.concatenate((np_target, pool_target_dropout), axis=0)

    pool_data = torch.from_numpy(np_data)
    pool_target = torch.from_numpy(np_target)



# Build an encoder

class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        self.output_dim = 100

        self.input_height = 28
        self.input_width = 28
        self.input_dim = 1

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, stride=2, padding=1),  # 64 x 14 x 14
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2, stride=2),  # 64 x 7 x 7

            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 128 x 3 x 3
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 1, 4, stride=2, padding=1),  # 1 x 1 x 1
            nn.ReLU(),

        )

        self.fc1 = nn.Sequential(
            nn.Linear(1, 1024),
            nn.ReLU(),
        )

        self.fc21 = nn.Sequential(
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )

        self.fc22 = nn.Sequential(
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )

        utils.initialize_weights(self)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc1(x)
        mu = self.fc21(x)
        log_var = self.fc22(x)
        return mu, log_var


def reparameterize(mu, log_var):
    std = log_var.mul(0.5).exp_()
    eps = Variable(std.data.new(std.size()).normal_())

    return eps.mul(std).add_(mu)

    # # Alternative
    # std = torch.exp(0.5 * log_var)
    # eps = torch.randn_like(std)
    # return eps.mul(std).add_(mu)


def latent_loss(mu, log_var):

    std = log_var.mul(0.5).exp_()
    mean_sq = mu * mu
    stddev_sq = std * std
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

    # # Alternative
    # return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())


# Build a generator/decoder

class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self):
        super(generator, self).__init__()

        self.input_height = 28
        self.input_width = 28
        self.input_dim = 100 + 10
        self.output_dim = 1

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_height // 4) * (self.input_width // 4)),
            nn.BatchNorm1d(128 * (self.input_height // 4) * (self.input_width // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Sigmoid(),
        )
        utils.initialize_weights(self)

    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = self.fc(x)
        x = x.view(-1, 128, (self.input_height // 4), (self.input_width // 4))
        x = self.deconv(x)

        return x


# Build a discriminator

class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self):
        super(discriminator, self).__init__()
        self.input_height = 28
        self.input_width = 28
        self.input_dim = 1
        self.output_dim = 1

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128 * (self.input_height // 4) * (self.input_width // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
        )
        self.dc = nn.Sequential(
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_height // 4) * (self.input_width // 4))
        x = self.fc1(x)
        d = self.dc(x)

        return d


# Initialize the model

def init_model():
    global model
    global optimizer
    global model_scheduler

    # model = Net_Correct()
    model = lenet()

    if cuda:
        model.cuda()

    decay = 3.5 / train_data.size(0)

    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5, weight_decay=decay)

    # optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.5, 0.999), weight_decay=decay)

    optimizer = optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-6, weight_decay=decay)  
    model_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5, last_epoch=-1)


# VAE-ACGAN training

class VAEACGAN(object):
    def __init__(self):
        # parameters
        self.epoch = 50
        self.sample_num = 100
        self.batch_size = 100
        self.save_dir = 'models'
        self.result_dir = 'results'
        self.log_dir = 'logs'
        self.gpu_mode = True
        self.model_name = 'VAEACGAN'

        # networks init
        init_model()                # Classifier
        self.E = encoder()          # Encoder
        self.G = generator()        # Generator/Decoder
        self.D = discriminator()    # Discriminator

        self.E_optimizer = optim.Adadelta(self.E.parameters(), lr=1.0, rho=0.9, eps=1e-6)
        self.G_optimizer = optim.Adadelta(self.G.parameters(), lr=1.0, rho=0.9, eps=1e-6)
        self.D_optimizer = optim.Adadelta(self.D.parameters(), lr=1.0, rho=0.9, eps=1e-6)

        self.E_scheduler = lr_scheduler.StepLR(self.E_optimizer, step_size=8, gamma=0.5, last_epoch=-1)
        self.G_scheduler = lr_scheduler.StepLR(self.G_optimizer, step_size=8, gamma=0.5, last_epoch=-1)
        self.D_scheduler = lr_scheduler.StepLR(self.D_optimizer, step_size=8, gamma=0.5, last_epoch=-1)


        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.E.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
            # self.BCE_loss = nn.BCELoss().cuda()
            self.CE_loss = nn.CrossEntropyLoss().cuda()
            self.MSE_loss = nn.MSELoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()
            self.CE_loss = nn.CrossEntropyLoss()
            self.MSE_loss = nn.MSELoss()

        self.data_X = train_data

        print(self.data_X.size())

        y_train = np.asarray(train_target).astype(np.int)
        y_train_vec = np.zeros((len(y_train), 10), dtype=np.float)
        for i, label in enumerate(y_train):
            y_train_vec[i, y_train[i]] = 1

        self.data_Y = torch.from_numpy(y_train_vec).type(torch.FloatTensor)

        self.y_train = y_train
        self.X_test = test_data

        y_test = np.asarray(test_target).astype(np.int)
        y_test_vec = np.zeros((len(y_test), 10), dtype=np.float)
        for i, label in enumerate(y_test):
            y_test_vec[i, y_test[i]] = 1

        self.y_test_vec = torch.from_numpy(y_test_vec).type(torch.FloatTensor)

        self.z_dim = 100
        self.y_dim = 10

        # fixed noise & condition
        self.sample_z_ = torch.zeros((self.sample_num, self.z_dim))
        for i in range(10):
            self.sample_z_[i*self.y_dim] = torch.rand(1, self.z_dim)
            for j in range(1, self.y_dim):
                self.sample_z_[i*self.y_dim + j] = self.sample_z_[i*self.y_dim]

        temp = torch.zeros((10, 1))
        for i in range(self.y_dim):
            temp[i, 0] = i

        temp_y = torch.zeros((self.sample_num, 1))
        for i in range(10):
            temp_y[i*self.y_dim: (i+1)*self.y_dim] = temp

        self.sample_y_ = torch.zeros((self.sample_num, self.y_dim))
        self.sample_y_.scatter_(1, temp_y.type(torch.LongTensor), 1)
        if self.gpu_mode:
            self.sample_z_, self.sample_y_ = Variable(self.sample_z_.cuda(), volatile=True), Variable(self.sample_y_.cuda(), volatile=True)
        else:
            self.sample_z_, self.sample_y_ = Variable(self.sample_z_, volatile=True), Variable(self.sample_y_, volatile=True)

    def train(self):
        self.train_hist = {}
        self.train_hist['E_loss'] = []
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['C_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        if self.gpu_mode:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1).cuda()), Variable(torch.zeros(self.batch_size, 1).cuda())
        else:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1)), Variable(torch.zeros(self.batch_size, 1))

        self.E.train()
        self.D.train()
        model.train()
        print('training start!!')
        start_time = time.time()

        for epoch in range(self.epoch):
            self.G.train()
            model_scheduler.step()
            self.E_scheduler.step()
            self.G_scheduler.step()
            self.D_scheduler.step()
            epoch_start_time = time.time()
            for iter in range(len(self.data_X) // self.batch_size):
                x_ = self.data_X[iter*self.batch_size:(iter+1)*self.batch_size]
                y_vec_ = self.data_Y[iter*self.batch_size:(iter+1)*self.batch_size]
                # z_ = torch.Tensor(self.batch_size, self.z_dim).normal_(0, 1)
                z_ = torch.randn((self.batch_size, self.z_dim))
                if self.gpu_mode:
                    x_, y_vec_, z_ = Variable(x_.cuda()), Variable(y_vec_.cuda()), Variable(z_.cuda())
                else:
                    x_, y_vec_, z_ = Variable(x_), Variable(y_vec_), Variable(z_)


                # Fix G, update E network

                self.E_optimizer.zero_grad()

                mu, log_var = self.E(x_)
                noise = reparameterize(mu, log_var)
                noise = noise.view(self.batch_size, 100)
                output = self.G(noise, y_vec_)

                # Compute the decoder loss that will be added to network E
                ll = latent_loss(mu, log_var)
                E_loss = self.MSE_loss(output, x_)  # / self.batch_size
                E_loss += ll

                self.train_hist['E_loss'].append(E_loss.item())

                E_loss.backward(retain_graph=True)
                self.E_optimizer.step()

                # Fix E, D, C, update G network
                self.G_optimizer.zero_grad()

                # Compute the GAN loss that will be added to the Generator G
                G_ = self.G(z_, y_vec_)

                D_fake = self.D(G_)
                C_fake = model(G_)

                G_loss= self.BCE_loss(D_fake, self.y_real_)
                C_fake_loss = self.CE_loss(C_fake, torch.max(y_vec_, 1)[1])

                G_loss += C_fake_loss

                # Compute the decoder loss that will be added to the Generator G

                mu, log_var = self.E(x_)
                noise = reparameterize(mu, log_var)
                noise = noise.view(self.batch_size, 100)
                G_dec = self.G(noise, y_vec_)


                G_dec_loss = self.MSE_loss(G_dec, x_)  # / self.batch_size
                # G_dec_loss = F.binary_cross_entropy(G_dec, x_, size_average=False) / self.batch_size

                G_loss += 0.75*G_dec_loss

                self.train_hist['G_loss'].append(G_loss.item())

                G_loss.backward(retain_graph=True)
                self.G_optimizer.step()

                # Fix G, update D, C network

                self.D_optimizer.zero_grad()
                optimizer.zero_grad()

                D_real = self.D(x_)
                C_real = model(x_)

                D_real_loss = self.BCE_loss(D_real, self.y_real_)
                C_real_loss = self.CE_loss(C_real, torch.max(y_vec_, 1)[1])

                G_ = self.G(z_, y_vec_)

                D_fake = self.D(G_)
                C_fake = model(G_)
                D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)
                C_fake_loss = self.CE_loss(C_fake, torch.max(y_vec_, 1)[1])


                mu, log_var = self.E(x_)
                noise = reparameterize(mu, log_var)
                noise = noise.view(self.batch_size, 100)
                # output = self.G(noise, y_vec_)

                G_dec = self.G(noise, y_vec_)

                D_dec = self.D(G_dec)
                C_dec = model(G_dec)
                D_dec_loss = self.BCE_loss(D_dec, self.y_fake_)
                C_dec_loss = self.CE_loss(C_dec, torch.max(y_vec_, 1)[1])


                D_loss = D_real_loss + D_fake_loss + D_dec_loss
                C_loss = C_real_loss + C_fake_loss + C_dec_loss

                self.train_hist['D_loss'].append(D_loss.item())
                self.train_hist['C_loss'].append(C_loss.item())

                D_loss.backward(retain_graph=True)
                self.D_optimizer.step()

                C_loss.backward(retain_graph=True)
                optimizer.step()


                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f, C_loss: %.8f, E_loss: %.8f" %
                          ((epoch + 1), (iter + 1), len(self.data_X) // self.batch_size, D_loss.item(), G_loss.item()
                           , C_loss.item(), E_loss.item()))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            self.visualize_results((epoch+1))

            if epoch==self.epoch-1:
                model.eval()
                _, test_acc_ = test(epoch)

        print("Training finish!... save training results")

        self.save()
        utils.generate_animation(self.result_dir + '/'  + self.model_name + '/' + self.model_name,
                                 self.epoch)
        utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.model_name), self.model_name)

        return test_acc_

    def visualize_results(self, epoch, fix=True):
        self.G.eval()

        if not os.path.exists(self.result_dir + '/'  + self.model_name):
            os.makedirs(self.result_dir + '/' + self.model_name)

        image_frame_dim = int(np.floor(np.sqrt(self.sample_num)))

        if fix:
            """ fixed noise """
            samples = self.G(self.sample_z_, self.sample_y_)
        else:
            """ random noise """
            temp = torch.LongTensor(self.batch_size, 1).random_() % 10
            sample_y_ = torch.FloatTensor(self.batch_size, 10)
            sample_y_.zero_()
            sample_y_.scatter_(1, temp, 1)
            if self.gpu_mode:
                sample_z_, sample_y_ = Variable(torch.randn((self.batch_size, self.z_dim)).cuda(), volatile=True), \
                                       Variable(sample_y_.cuda(), volatile=True)
            else:
                sample_z_, sample_y_ = Variable(torch.randn((self.batch_size, self.z_dim)), volatile=True), \
                                       Variable(sample_y_, volatile=True)

            samples = self.G(sample_z_, sample_y_)

        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_dir + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png')

    def save(self):
        save_dir = os.path.join(self.save_dir, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))
        torch.save(model.state_dict(), os.path.join(save_dir, self.model_name + '_C.pkl'))
        torch.save(self.E.state_dict(), os.path.join(save_dir, self.model_name + '_E.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))
        model.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_C.pkl')))
        self.E.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_E.pkl')))


def main(argv):
    start_time = time.time()
    print (str(argv[0]))

    initialize_train_set()

    init_model()
    print ("Training without acquisition")
    for epoch in range(1, epochs + 1):
        train(epoch)
        test(epoch)

    print ("acquiring points")
    acquire_points(str(argv[0]))

    print ("Training again")
    gan = VAEACGAN()
    _, test_acc = gan.train()

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main(sys.argv[1:])

