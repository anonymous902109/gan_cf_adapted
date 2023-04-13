
import numpy as np
import torch
from stable_baselines3 import DQN
from torch.utils.data import DataLoader

from gridworld import Gridworld
from src.dataset_generation import create_dataset, create_unique_dataset, under_sample, split_dataset
from src.star_gan.main import DiscreteDataset
from src.star_gan.model import Generator
from src.train import train_star_gan
from src.util import generate_counterfactual


def run():
    # Initialize params
    env_name = 'gridworld'
    env_type = 'Discrete'

    agent_path = '{}.zip'.format(env_name)
    env = Gridworld()

    # agent = DQN("MlpPolicy", env, verbose=1)
    # agent.learn(total_timesteps=int(2e5), progress_bar=True)
    # agent.save(agent_path)

    agent = DQN.load(agent_path)

    # Generate datasets
    nb_domains = env.action_space.n
    nb_samples = 400000
    dataset_path = "res/datasets/{}".format(env_name)
    unique_dataset_path = dataset_path + "_Unique"
    domains = list(map(str, np.arange(nb_domains)))

    create_dataset(env_name, env, nb_samples, dataset_path, agent, agent_type='dqn', dataset=env_type, seed=42, epsilon=0.2, domains=domains)
    under_sample(dataset_path, min_size=nb_samples / nb_domains, domains=domains, dataset=env_type)
    create_unique_dataset(unique_dataset_path, dataset_path, env_type)
    under_sample(unique_dataset_path, dataset=env_type, domains=domains)
    split_dataset(unique_dataset_path, 0.1, domains, env_type)

    train_star_gan("{}_Unique".format(env_name),
                   env_name,
                   agent_type='dqn',
                   env_type=env_type,
                   image_size=32,
                   image_channels=1,
                   c_dim=6,
                   batch_size=128,
                   agent_file=agent_path)

    generator = Generator(c_dim=nb_domains, channels=1)
    generator.load_state_dict(torch.load("res/models/{}/models/200-G.ckpt".format(env_name),
                                         map_location=lambda storage, loc: storage))

    fact_path = "res/datasets/{}_Unique/test".format(env_name)

    ds = DiscreteDataset(fact_path)
    dl = DataLoader(ds, batch_size=10, shuffle=True)

    x, y = next(iter(dl))
    for i in range(len(y)):
        fact = x[i]
        fact = fact.unsqueeze(0)
        target = y[i].item()
        for i in range(nb_domains):
            if i != target:
                cf, _ = generate_counterfactual(generator, fact, target, nb_domains, 32)
                cf = cf.squeeze().tolist()
                cf = [int(j) for j in cf]
                print('FACT = {} CF = {} TARGET = {}'.format(fact.squeeze().tolist(), cf, i))


if __name__ == '__main__':
    run()