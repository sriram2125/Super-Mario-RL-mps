import torch
from pathlib import Path
import datetime, os
from gym.wrappers import FrameStack
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY 
from nes_py.wrappers import JoypadSpace

from wrappers import SkipFrame, GrayScaleObservation, ResizeObservation
from agent import Mario

# 1. Setup Environment
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', apply_api_compatibility=True, render_mode="rgb_array")
# Note: render_mode="human" is faster for training than "human"

env = JoypadSpace(env, RIGHT_ONLY) # Train him to just go Right first (easier)
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)

# 2. Setup Agent
save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)

# 3. Training Loop
episodes = 40000 
for e in range(episodes):
    state, info = env.reset()
    
    while True:
        # Run agent on the state
        action = mario.act(state)

        # Agent performs action
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Remember
        mario.cache(state, next_state, action, reward, done)

        # Learn
        q, loss = mario.learn()

        # Update state
        state = next_state

        if done or info['flag_get']:
            break

    print(f"Episode {e} finished. Epsilon: {mario.exploration_rate:.4f}")

    # Optional: Save every 20 episodes
    if e % 20 == 0:
         # You can add code here to save model weights if you want
         pass