import torch
import random
import numpy as np
from collections import deque
from model import MarioNet

class Mario:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Agent running on: {self.device}")

        self.net = MarioNet(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)

        self.exploration_rate = 1.0
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.memory = deque(maxlen=40000) 
        self.batch_size = 32

        self.gamma = 0.9 
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()
        
        self.burnin = 1e4
        self.learn_every = 3
        self.sync_every = 1e4

    def act(self, state):
        """Given a state, choose an epsilon-greedy action."""
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        else:
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            # FIXED: Added .float() to convert integer pixels to expected float format
            state = torch.tensor(state, device=self.device).unsqueeze(0).float()
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """Store the experience in memory."""
        state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
        next_state = next_state[0].__array__() if isinstance(next_state, tuple) else next_state.__array__()

        # We store as integers (default) to save memory, convert to float only when using
        state = torch.tensor(state, device=self.device)
        next_state = torch.tensor(next_state, device=self.device)
        action = torch.tensor([action], device=self.device)
        reward = torch.tensor([reward], device=self.device)
        done = torch.tensor([done], device=self.device)

        self.memory.append((state, next_state, action, reward, done))

    def recall(self):
        """Retrieve a batch of experiences from memory."""
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        
        # FIXED: Convert the batch to float() and normalize pixel values (0-255 -> 0-1)
        # Normalizing helps the neural network learn much faster!
        return state.float() / 255.0, next_state.float() / 255.0, action.squeeze(), reward.squeeze(), done.squeeze()

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        state, next_state, action, reward, done = self.recall()

        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)

        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)

    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")
        return current_Q[np.arange(0, self.batch_size), action]

    def td_target(self, reward, next_state, done):
        future_Q = self.net(next_state, model="target")
        best_future_Q = torch.argmax(self.net(next_state, model="online"), axis=1)
        future_val = future_Q[np.arange(0, self.batch_size), best_future_Q]
        return (reward + (1 - done.float()) * self.gamma * future_val).to(torch.float32)

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())