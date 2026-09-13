"""Prioritized replay and n-step returns from the agent's own trajectories."""

from collections import deque

import numpy as np

from .env import SHAPE


class Replay:
    def __init__(self, capacity=200_000, alpha=0.6):
        self.capacity, self.alpha = capacity, alpha
        self.pos, self.size, self.maximum = 0, 0, 1.0
        self.obs = np.empty((capacity, *SHAPE), np.uint8)
        self.next_obs = np.empty_like(self.obs)
        self.actions = np.empty(capacity, np.int32)
        self.returns = np.empty(capacity, np.float32)
        self.discounts = np.empty(capacity, np.float32)
        self.leaves = 1 << (capacity-1).bit_length()
        self.tree = np.zeros(self.leaves*2, np.float64)

    def add(self, obs, action, reward, next_obs, discount):
        i = self.pos
        self.obs[i], self.next_obs[i] = obs, next_obs
        self.actions[i], self.returns[i], self.discounts[i] = action, reward, discount
        node = self.leaves+i
        change = self.maximum-self.tree[node]
        while node:
            self.tree[node] += change
            node //= 2
        self.pos = (i+1) % self.capacity
        self.size = min(self.size+1, self.capacity)

    def sample(self, batch_size, rng, beta):
        total = self.tree[1]
        masses = (np.arange(batch_size)+rng.random(batch_size))*total/batch_size
        nodes = np.ones(batch_size, np.int64)
        while nodes[0] < self.leaves:
            left = nodes*2
            go_right = masses >= self.tree[left]
            masses -= self.tree[left]*go_right
            nodes = left+go_right
        indices = nodes-self.leaves
        weights = (self.size*self.tree[nodes]/total)**(-beta)
        weights /= weights.max()
        return indices, (self.obs[indices], self.actions[indices], self.returns[indices],
                         self.next_obs[indices], self.discounts[indices], weights.astype(np.float32))

    def priorities(self, indices, errors):
        values = (np.asarray(errors, np.float64)+0.01)**self.alpha
        self.maximum = max(self.maximum, float(values.max()))
        nodes, unique = np.unique(indices+self.leaves, return_index=True)
        self.tree[nodes] = values[unique]
        while nodes[0] > 1:
            nodes = np.unique(nodes//2)
            self.tree[nodes] = self.tree[nodes*2]+self.tree[nodes*2+1]


class NStep:
    def __init__(self, replay, n=5, gamma=0.995):
        self.replay, self.n, self.gamma = replay, n, gamma
        self.queue = deque()

    def append(self, obs, action, reward, next_obs, terminated, truncated):
        self.queue.append((obs, action, reward, next_obs, terminated))
        if len(self.queue) >= self.n:
            self._emit()
        if terminated or truncated:
            while self.queue:
                self._emit()

    def _emit(self):
        reward, discount = 0.0, 1.0
        for item in self.queue:
            reward += discount*item[2]
            discount *= self.gamma
            if item[4]:
                discount = 0.0
                break
        first = self.queue.popleft()
        self.replay.add(first[0], first[1], reward, item[3], discount)
