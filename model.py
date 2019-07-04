import chainer.functions as F
from chainer import optimizers, Variable, serializers
from collections import deque
import gym
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys

from network import NN

SEED_NUMPY = 12345
np.random.seed(SEED_NUMPY)


class Model(object):
    """Model for predicting next state based on current state and action.

    Args:
        state_n:   dimension of state
        action_n:  dimension of action

    Result:
        next state = f(state, action) = NN(state + action)

    """

    def __init__(self, state_n, action_n):
        super(Model, self).__init__()
        self.model = NN(state_n + action_n, state_n)
        self.optimizer = optimizers.MomentumSGD(lr=1e-4)
        self.optimizer.setup(self.model)
        self.train_data = deque()
        self.train_data_size_max = 2000

    def predict(self, state, action):
        state_action = np.concatenate((state, action), axis=0).astype(np.float32)
        state_action = Variable(state_action.reshape((1, state_action.shape[0])))
        next_state = self.model(state_action)
        return next_state

    def store_data(self, state, action, next_state):
        state_action = np.concatenate((state, action), axis=0)
        self.train_data.append((state_action, next_state))
        if len(self.train_data) > self.train_data_size_max:
            self.train_data.popleft()

    def shuffle_data(self):
        data = np.array(self.train_data)
        return np.random.permutation(data)

    def train(self, n_epoch, batch_size):
        print('Train start!')
        for epoch in range(n_epoch):
            # print(f'epoch: {epoch}')

            perm = self.shuffle_data()
            sum_loss = 0.

            # Train
            for i in range(0, len(perm), batch_size):
                batch_data = perm[i:i + batch_size]

                x_batch = np.array(list(batch_data[:, 0]), dtype=np.float32)
                t_batch = np.array(list(batch_data[:, 1]), dtype=np.float32)

                x_batch, t_batch = Variable(x_batch), Variable(t_batch)

                y = self.model(x_batch)
                loss = F.mean_squared_error(y, t_batch)

                self.model.cleargrads()
                loss.backward()
                self.optimizer.update()
                sum_loss += loss.data

            # print(f'train loss: {sum_loss}')

        self.save_model()

    @property
    def train_data_size(self):
        return len(self.train_data)

    def dump_data(self, file='train_data/train_data.txt'):
        with open(file, 'wb') as f:
            pickle.dump(self.train_data, f)

    def load_data(self, file='train_data/train_data.txt'):
        with open(file, 'rb') as f:
            self.train_data = pickle.load(f)

    @staticmethod
    def exist_data(file='train_data/train_data.txt'):
        return os.path.exists(file)

    def save_model(self, file='model/model.model'):
        serializers.save_npz(file, self.model)

    def load_model(self, file='model/model.model'):
        serializers.load_npz(file, self.model)

    @staticmethod
    def exist_model(file='model/model.model'):
        return os.path.exists(file)


class RandomController(object):
    """Pick action by random"""

    def __init__(self):
        super(RandomController, self).__init__()
        self.name = 'random'

    def get_action(self, action_list):
        return np.random.choice(action_list)


class MPCController(object):
    """Pick action by MPCController"""

    def __init__(self, rollouts_num, rollout_length):
        super(MPCController, self).__init__()
        self.name = 'mpc'
        self.rollouts_num = rollouts_num
        self.rollout_length = rollout_length
        self.randomcontroller = RandomController()

    def get_action(self, state, action_list, model):
        rollouts_1st_action = []
        cost_result = np.zeros(self.rollouts_num)

        for rollout in range(self.rollouts_num):
            state_tmp = state.copy()

            for step in range(self.rollout_length):
                action = self.randomcontroller.get_action(action_list)
                # save 1st action of every rollout
                if step == 0:
                    rollouts_1st_action.append(action)
                state_tmp = model.predict(state_tmp, [action])[0].data

            # save current cartpole_angle as cost (cartpole_angle->0, result->better)
            cost_result[rollout] = abs(state_tmp[2])

        best_result_rollout = cost_result.argmin()
        return rollouts_1st_action[best_result_rollout]
