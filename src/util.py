import time

import gym
import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F
from PIL import Image
import keras
from torch.autograd import Variable

from src.atari_wrapper import AtariWrapper
from src.star_gan.data_loader import get_star_gan_transform
from src.star_gan.model import Generator


try:
    from mpi4py import MPI
except ImportError:
    MPI = None



def run_agent(max_steps, agent, env_name, seed=None, max_noop=1, render=True, power_pill_objective=False,
              max_episodes=None, agent_type="keras", ablate_agent=False):
    """
    Runs the given agent on the given environment. This is mainly used to measure the performance of agents and for
    debugging. The accumulated reward is printed after every episode and the mean and std reward over all episodes is
    printed at the end.

    :param max_steps: Maximum amount of total steps until termination.
    :param agent: A trained agent (Pytorch and Keras are supported).
    :param env_name: The Gym environment name.
    :param seed: A random number generation seed.
    :param max_noop: A maximum amount of NOOPs that are executed at the start of each episode.
    :param render: Whether to render frames.
    :param power_pill_objective: Whether the Power-Pill objective is used on Pac-Man.
    :param max_episodes: Maximum amount of episodes until termination.
    :param agent_type: the type of Pacman Agent, ignored with Space Invader. Accepts "keras" or "acer".
    :param ablate_agent: Whether the laser canon should be hidden from the frames that are input to the agent.
    :return: None
    """
    if seed is not None:
        np.random.seed(seed)

    wrapper, skip_frames = init_environment(env_name, power_pill_objective, agent_type=agent_type,
                                            ablate_agent=ablate_agent)
    stacked_frames = wrapper.reset(noop_max=max_noop)

    total_reward = 0
    reward_list = []
    nb_episodes = 0

    for i in range(max_steps):
        if i < 4:
            action = wrapper.env.action_space.sample()
        else:
            output = get_agent_prediction_from_stacked_frames(agent, stacked_frames)
            action = np.argmax(np.squeeze(output))

        stacked_frames, observations, reward, done, info = wrapper.step(action, skip_frames=skip_frames)
        total_reward += reward
        if done:
            print('total_reward', total_reward)
            reward_list.append(total_reward)
            total_reward = 0
            nb_episodes += 1
            if max_episodes is not None and nb_episodes >= max_episodes:
                break

        if render:
            wrapper.env.render()

    wrapper.env.close()

    reward_list.append(total_reward)
    average_reward = np.mean(reward_list)
    std_reward = np.std(reward_list)

    print("AVG Reward:", average_reward)
    print("STD Reward:", std_reward)


def init_environment(env_name, power_pill_objective, agent_type, ablate_agent=False):
    """
    Initializes a wrapped Gym environment for atari games. Only supported for Ms. Pac-Man and Space Invaders

    :param env_name: The Gym environment name.
    :param power_pill_objective: Whether the Power-Pill objective is used on Pac-Man.
    :param agent_type: the type of Pacman Agent, ignored with Space Invader. Accepts "keras" or "acer".
    :param ablate_agent: Whether the laser canon should be hidden from frames that are input to the agent.
    :return: (wrapper, skip_frames) - The environment wrapper and the amount of skipped frames that are used for the
        given environment.
    """
    if env_name.startswith("MsPacman") and (agent_type == "acer"):
        wrapper = AtariWrapper(env_name, power_pill_objective=power_pill_objective, deepq_preprocessing=False)
    else:
        wrapper = AtariWrapper(env_name, power_pill_objective=power_pill_objective, ablate_agent=ablate_agent)

    if env_name.startswith("MsPacman"):
        skip_frames = 4
    elif env_name.startswith("SpaceInvaders"):
        skip_frames = 7
    else:
        raise NotImplementedError("Only implemented for PacMan and SpaceInvaders.")

    return wrapper, skip_frames


def get_action_names(env_name):
    """
    Returns all a list of the action names of the given Gym environment.

    :param env_name: The Gym environment name.
    :return: List of action names.
    """
    env = gym.make(env_name)
    return env.unwrapped.get_action_meanings()


def restrict_tf_memory():
    """
    Restricts the tensorflow memory usage to be dynamic. If not used, tensorflow will greedily take memory.

    :return: None
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def get_agent_action(agent, frame, pacman=True, agent_type="deepq", ablate_agent=False):
    """
    Gets the action that an agent would choose on the given single frame under a greedy policy. The given frame is
    copied 3 times for the input to get a 4-image input for atari agents.

    :param agent: The trained agent (Pytorch and Keras are supported).
    :param frame: The input frame to the agent.
    :param pacman: Whether the target environment is Pac-Man or Space Invaders.
    :param agent_type: The type of agent. "deepq" for keras deepq, "acer" for baselines acer, "torch" for a pytorch acer-critic
    :return: Integer that encodes the chosen action.
    """
    if not isinstance(frame, Image.Image):
        raise NotImplementedError("get_agent_action is only implemented for Image frames (not numpy arrays or other)")
    if frame.size == (176, 176):
        frame = frame.crop((8, 1, 168, 174))

    agent_prediction = get_agent_prediction(agent, frame, pacman=pacman, agent_type=agent_type,
                                            ablate_agent=ablate_agent)
    return int(np.argmax(np.squeeze(agent_prediction)))


def get_agent_prediction(agent, frame, pacman=True, agent_type="deepq", ablate_agent=False):
    """
    Gets the unprocessed agent output of the given agent on the given single frame. The given frame is copied 3
    times for the input to get a 4-image input for atari agents.

    :param agent: The trained agent (Pytorch and Keras are supported).
    :param frame: The input frame to the agent.
    :param pacman: Whether the target environment is Pac-Man or Space Invaders.
    :param agent_type: The type of agent. "deepq" for keras deepq, "acer" for baselines acer, "torch" for a pytorch acer-critic
    :return: An action distribution (A list of numeric output values for each action).
    """
    if pacman:
        if agent_type == "deepq":
            frame = AtariWrapper.preprocess_frame(np.array(frame))
        elif agent_type == "acer":
            frame = AtariWrapper.preprocess_frame_ACER(np.array(frame))
    else:
        frame = AtariWrapper.preprocess_space_invaders_frame(np.array(frame), ablate_agent=ablate_agent)
    frame = np.squeeze(frame)
    stacked_frames = np.stack([frame for _ in range(4)], axis=-1)
    if not pacman:
        stacked_frames = AtariWrapper.to_channels_first(stacked_frames)
    stacked_frames = np.expand_dims(stacked_frames, axis=0)
    if isinstance(agent, keras.Model):
        output = agent.predict(stacked_frames)
        if len(output) == 2:
            output = output[0]
    elif agent_type == "acer":
        sess = agent.step_model.sess
        feed_dict = {agent.step_model.X: adjust_shape(agent.step_model.X, stacked_frames)}
        output = sess.run(agent.step_model.pi, feed_dict)
    else:
        torch_state = torch.Tensor(stacked_frames)
        output = agent.pi(agent(torch_state)).detach().cpu().numpy()
    return output


def get_agent_prediction_from_stacked_frames(agent, stacked_frames):
    """
    Gets the unprocessed agent output of the given agent on the given stacked frames.

    :param agent: The trained agent (Pytorch and Keras are supported).
    :param stacked_frames: A list or array of 4 frames.
    :return: An action distribution (A list of numeric output values for each action).
    """
    stacked_frames = np.expand_dims(stacked_frames, axis=0)

    if isinstance(agent, keras.models.Model):
        # Keras
        output = agent.predict(stacked_frames)
        if len(output) == 2:
            # in case a dueling net is used
            output = output[0]
    else:
        # Pytorch
        torch_state = torch.Tensor(stacked_frames)
        output = agent.pi(agent(torch_state)).detach().numpy()

    return output


def denorm(x):
    """
    Converts the range of input x from [-1, 1] to [0, 1].

    :param x: Torch variable in range [-1, 1]
    :return: Torch variable in range [0, 1]
    """
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def array_to_pil_format(array):
    """
    Converts an input array to PIL's image format (range [0, 255] and 8-bit integer encoding).

    :param array: Array representing an image.
    :return: If the input is not already 8-bit encoded, returns the input array times 255 in 8-bit encoding.
    """
    if array.dtype != np.uint8:
        return (array * 255).astype(np.uint8)
    else:
        return array


def load_baselines_model(path, num_actions = False, num_env = 4):
    """
    Loads a model of the baselines repository.

    :param path: Path to the trained agent, that was trained by the openai baselines repository.
    :param num_actions: Used to define custom amount of actions. For example to remove ambiguous actions in Pacman.
        If False, the nomral amount is used.
    :param num_env: The number of envs to learn in parallel. Should be equivalent to batchsize during GAN training.
    """
    args = []
    args.append("--alg=acer")
    args.append("--env=MsPacmanNoFrameskip-v4")
    args.append("--num_timesteps=0")
    args.append("--load_path=" + path)
    args.append("--play")
    args.append(f"--num_env={num_env}")

    # configure logger, disable logging in child MPI processes (with rank > 0)
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        configure_logger(args.log_path)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger(args.log_path, format_strs=[])

    env_type, env_id = get_env_type(args)
    print('env_type: {}'.format(env_type))

    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)

    env = build_env(args)
    if num_actions:
        env.action_space = gym.spaces.Discrete(num_actions)

    if args.save_video_interval != 0:
        raise NotImplementedError("Video saving is not implemented here. Use the origninal Baselines repository.")

    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))

    model = learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        **alg_kwargs
    )

    return model


def generate_counterfactual(generator, image, target_domain, nb_domains, image_size=176):
    """
    Generates a counterfactual frame for the given image with StarGAN.

    :param generator: The StarGAN generator.
    :param image: The PIL image to generate a counterfactual for.
    :param target_domain: The integer encoded target action/domain.
    :param nb_domains: Number of possible actions/domains.
    :param image_size: The image size of the input image and the counterfactual (squared images are assumed).
    :return: (counterfactual, generation_time) - The counterfactual is a PIL image and the generation time is the pure
        time spent for the forward call (without pre- or postprocessing).
    """
    # define preprocessing
    # transform = get_star_gan_transform(image_size, image_size, len(image.getbands()))
    #
    # # load and preprocess example image
    # image = transform(image).cuda()
    # image = image.unsqueeze(0)

    # convert target class to onehot
    onehot_target_class = np.zeros(nb_domains, dtype=int)
    onehot_target_class[target_domain] = 1
    onehot_target_class = torch.tensor([onehot_target_class])

    # generate counterfactual
    start_time = time.time()
    counterfactual = generator(image, onehot_target_class)
    generation_time = time.time() - start_time

    # convert to PIL image
    # counterfactual = denorm(counterfactual)
    # counterfactual = counterfactual.detach().permute(0, 2, 3, 1).cpu().numpy()
    # counterfactual = np.squeeze(counterfactual, axis=0)
    # counterfactual = (counterfactual * 255).astype(np.uint8)
    # counterfactual = Image.fromarray(counterfactual)

    return counterfactual, generation_time


if __name__ == "__main__":
    pass
