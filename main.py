import gym
import matplotlib.pyplot as plt
import numpy as np

from model import RandomController, MPCController, Model


SEED_NUMPY = 12345
np.random.seed(SEED_NUMPY)


def store_data(controller, model=None):
    env_name = "CartPole-v0"
    env = gym.make(env_name)

    n_st = env.observation_space.shape[0]

    if type(env.action_space) == gym.spaces.discrete.Discrete:
        # CartPole-v0, Acrobot-v0, MountainCar-v0
        n_act = env.action_space.n
        action_list = np.arange(0, n_act)
    elif type(env.action_space) == gym.spaces.box.Box:
        # Pendulum-v0
        action_list = [np.array([a]) for a in [-2.0, 2.0]]
        n_act = len(action_list)

    # RandomController
    if controller.name == 'random':
        model = Model(n_st, 1)

        if model.exist_data():
            model.load_data()
            print(f'Exist enough data! Get data size: {model.train_data_size}')
        else:
            for i_episode in range(100):
                print("Episode_num" + str(i_episode))
                observation = env.reset()
                for t in range(200):
                    # env.render()
                    state = observation.astype(np.float32)
                    action = controller.get_action(action_list)

                    observation, reward, ep_end, _ = env.step(action)
                    state_dash = observation.astype(np.float32)

                    model.store_data(state, [action], state_dash)
                    if model.train_data_size >= model.train_data_size_max:
                        print(f'Get enough data! Data size: {model.train_data_size}')
                        model.dump_data()
                        return model

                    if ep_end:
                        print(f'Current data size: {model.train_data_size}')
                        break
            env.close()

    # MPCController
    else:
        observation = env.reset()
        for t in range(200):
            # env.render()
            state = observation.astype(np.float32)

            action = controller.get_action(state, action_list, model)

            observation, reward, ep_end, _ = env.step(action)
            state_dash = observation.astype(np.float32)

            model.store_data(state, [action], state_dash)

            if ep_end:
                print(f'Length: {t}')
                break
        model.dump_data()
        env.close()

    return model


def train():
    # use random_controller to collect data
    random_controller = RandomController()

    model = store_data(random_controller)

    if model.exist_model():
        model.load_model()
        print(f'Exist model! Load model!')
    else:
        model.train(n_epoch=5000, batch_size=64)

    # use mpc_controller to collect more data and update current dataset
    mpc_controller = MPCController(rollouts_num=10, rollout_length=10)

    train_times = 1000
    for i in range(train_times):
        model = store_data(mpc_controller, model=model)
        model.train(n_epoch=200, batch_size=64)


if __name__ == "__main__":
    agent = train()
