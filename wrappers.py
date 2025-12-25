import gym
from gym import spaces
import numpy as np
import cv2

class JoypadSpace(gym.Wrapper):
    """Restricts actions to the essential ones (Right, Jump, etc)."""
    def __init__(self, env, actions):
        super().__init__(env)
        self.actions = actions
        self.action_space = spaces.Discrete(len(actions))

    def step(self, action):
        return self.env.step(self.actions[action])

class SkipFrame(gym.Wrapper):
    """Skips frames. The agent acts every 4th frame to speed up training."""
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            done = terminated or truncated
            if done:
                break
        return obs, total_reward, done, truncated, info

class GrayScaleObservation(gym.ObservationWrapper):
    """Converts game to Black & White to save VRAM."""
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        return observation

class ResizeObservation(gym.ObservationWrapper):
    """Resizes screen to 84x84."""
    def __init__(self, env, shape=84):
        super().__init__(env)
        self.shape = (shape, shape)
        self.observation_space = spaces.Box(low=0, high=255, shape=self.shape, dtype=np.uint8)

    def observation(self, observation):
        observation = cv2.resize(observation, self.shape, interpolation=cv2.INTER_AREA)
        return observation