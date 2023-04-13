
import gym
import numpy as np

class DiscreteWrapper:
    ''' simple implementation for discrete envs'''

    def __init__(self, env_name, env, power_pill_objective=False, deepq_preprocessing=True, ablate_agent=False):
        self.env_name = env_name
        self.env = env
        self.env.reset()

        self.space_invaders = False

    @staticmethod
    def preprocess_frame(frame):
        ''' preprocessing according to openai's atari_wrappers.WrapFrame
            also applys scaling between 0 and 1 which is done in tensorflow in baselines
        :param frame: the input frame
        :return: rescaled and greyscaled frame
        '''
        frame = np.array([0, 0, 0] + list(frame) + [0,0,0])
        return frame

    @staticmethod
    def preprocess_frame_ACER(frame):
        ''' preprocessing according to openai's atari_wrappers.WrapFrame
            Does NOT apply scaling between 0 and 1 since ACER does not use it
        :param frame: the input frame
        :return: rescaled and greyscaled frame
        '''
        return frame

    @staticmethod
    def preprocess_space_invaders_frame(frame, ablate_agent):
        return frame

    @staticmethod
    def preprocess_original_frame(frame):
        return frame

    def update_stacked_frame(self, new_frame):
        ''' adds the new_frame to the stack of 4 frames, while shifting each other frame one to the left
        :param new_frame:
        :return: the new stacked frame
        '''
        return new_frame

    def update_original_stacked_frame(self, new_frame):
        return new_frame

    @staticmethod
    def to_channels_first(stacked_frames):
        return stacked_frames

    def step(self, action, skip_frames=4):
        return self.env.step(action)

    def repeat_frames(self, action, skip_frames=4):
        ''' skip frames to be inline with baselines DQN. stops when the current game is done
        :param action: the choosen action which will be repeated
        :param skip_frames: the number of frames to skip
        :return max frame: the frame used by the agent
        :return stacked_observations: all skipped observations '''
        skip_frames = 0
        stacked_observations = []
        # TODO dirty numbers
        obs_buffer = np.zeros((2, self.original_size[0], self.original_size[1], 3), dtype='uint8')
        total_reward = 0
        for i in range(skip_frames):
            observation, reward, done, info = self.env.step(action)
            stacked_observations.append(observation)
            if i == skip_frames - 2: obs_buffer[0] = observation
            if i == skip_frames - 1: obs_buffer[1] = observation
            if done:
                break

            if self.power_pill_objective:
                if self.ate_power_pill(reward):
                    self.power_pills_left -= 1
                    reward = 50
                else:
                    reward = 0

            total_reward += reward

        max_frame = obs_buffer.max(axis=0)
        return max_frame, stacked_observations, total_reward, done, info

    def reset(self, noop_min=0, noop_max=30):
        """ Do no-op action for a number of steps in [1, noop_max], to achieve random game starts.
        We also do no-op for 250 steps because Pacman cant do anything at the beginning of the game (number found empirically)
        """
        return self.env.reset()

    def fixed_reset(self, step_number, action):
        '''
        Create a fixed starting position for the environment by doing *action* for *step_number* steps
        :param step_number: number of steps to be done at the beginning of the game
        :param action: action to be done at the start of the game
        :return: obs at the end of the starting sequence
        '''
        return self.env.reset()


