"""Optional self-imitation from the learner's own completed training suffixes.

No replay-file loading or inference-time action override. Monte Carlo targets
include only score earned after each observation, ending at a real learning
terminal. A bounded suffix drops old observations, never invents a terminal.
"""

from collections import deque

import numpy as np

from .env import SHAPE, screen_info


class SILReplay:
    def __init__(self, capacity=32768, alpha=.6):
        if capacity < 1 or not 0 <= alpha <= 1:
            raise ValueError("invalid SIL replay capacity or priority exponent")
        self.capacity, self.alpha = capacity, alpha
        self.size, self.pos, self.maximum = 0, 0, 1.0
        self.obs = np.empty((capacity, *SHAPE), np.uint8)
        self.actions = np.empty(capacity, np.int32)
        self.returns = np.empty(capacity, np.float32)
        self.from_boot = np.empty(capacity, bool)
        self.leaves = 1 << (capacity-1).bit_length()
        self.tree = np.zeros(self.leaves*2, np.float64)

    def add(self, obs, action, target, from_boot):
        i = self.pos
        self.obs[i], self.actions[i], self.returns[i], self.from_boot[i] = obs, action, target, from_boot
        node = self.leaves+i
        change = self.maximum-self.tree[node]
        while node:
            self.tree[node] += change
            node //= 2
        self.pos = (i+1) % self.capacity
        self.size = min(self.size+1, self.capacity)

    def sample(self, batch_size, rng, beta=.1):
        if not self.size or batch_size < 1 or not 0 <= beta <= 1:
            raise ValueError("invalid SIL sample request")
        total = self.tree[1]
        masses = (np.arange(batch_size)+rng.random(batch_size))*total/batch_size
        nodes = np.ones(batch_size, np.int64)
        while nodes[0] < self.leaves:
            left = nodes*2
            right = masses >= self.tree[left]
            masses -= self.tree[left]*right
            nodes = left+right
        indices = nodes-self.leaves
        weights = (self.size*self.tree[nodes]/total)**(-beta)
        weights /= weights.max()
        return indices, (self.obs[indices], self.actions[indices], self.returns[indices],
                         weights.astype(np.float32))

    def priorities(self, indices, advantages):
        advantages = np.asarray(advantages, np.float64)
        if not np.isfinite(advantages).all() or (advantages < 0).any():
            raise ValueError("SIL priorities must be finite positive advantages")
        values = (advantages+.01)**self.alpha
        self.maximum = max(self.maximum, float(values.max()))
        nodes, unique = np.unique(np.asarray(indices)+self.leaves, return_index=True)
        self.tree[nodes] = values[unique]
        while nodes[0] > 1:
            nodes = np.unique(nodes//2)
            self.tree[nodes] = self.tree[nodes*2]+self.tree[nodes*2+1]


class TrainingSuffixes:
    def __init__(self, replay, workers, gamma=.995, suffix_steps=2048):
        if workers < 1 or suffix_steps < 1 or not 0 <= gamma <= 1:
            raise ValueError("invalid SIL trajectory settings")
        self.replay, self.gamma, self.limit = replay, gamma, suffix_steps
        self.pending = [deque(maxlen=suffix_steps) for _ in range(workers)]
        self.trimmed = [0]*workers
        self.segments, self.truncations, self.dropped_prefix = 0, 0, 0
        self.committed = dict(from_boot=0, restored=0)

    def append(self, worker, obs, action, reward, terminal, truncated, from_boot):
        if (not 0 <= worker < len(self.pending) or obs.shape != SHAPE
                or obs.dtype != np.uint8 or not isinstance(action,(int,np.integer))
                or not 0 <= action < 6 or not np.isfinite(reward)):
            raise ValueError("invalid own-training transition")
        queue = self.pending[worker]
        if queue and queue[-1][3] != from_boot:
            raise ValueError("SIL trajectory origin changed without a boundary")
        if len(queue) == self.limit:
            self.trimmed[worker] += 1
            self.dropped_prefix += 1
        queue.append((obs.copy(), int(action), float(reward), bool(from_boot)))
        if not (terminal or truncated):
            return None
        event = dict(worker=worker,retained_suffix_steps=len(queue),
                     dropped_prefix_steps=self.trimmed[worker],full_game=bool(from_boot),
                     initial_score=screen_info(queue[0][0][-1])["score"],
                     score_reward_sum=sum(x[2] for x in queue),truncated=bool(truncated),
                     committed=not truncated)
        if truncated:
            self.truncations += 1
        else:
            target, transitions = 0.0, []
            for screen, chosen, reward, origin in reversed(queue):
                target = reward+self.gamma*target
                transitions.append((screen, chosen, target, origin))
            for transition in reversed(transitions):
                self.replay.add(*transition)
            self.segments += 1
            self.committed["from_boot" if from_boot else "restored"] += len(queue)
            event.update(first_return=target,max_return=max(x[2] for x in transitions))
        queue.clear()
        self.trimmed[worker] = 0
        return event

    def metrics(self):
        n = sum(len(q) for q in self.pending)
        return dict(replay_size=self.replay.size,pending_transitions=n,
                    pending_screen_bytes=n*int(np.prod(SHAPE)),
                    replay_array_bytes=sum(x.nbytes for x in
                        (self.replay.obs,self.replay.actions,self.replay.returns,
                         self.replay.from_boot,self.replay.tree)),
                    completed_learning_segments=self.segments,
                    discarded_truncated_segments=self.truncations,
                    dropped_prefix_transitions=self.dropped_prefix,
                    committed_transitions=dict(self.committed))


def sil_terms(logits, values, actions, returns, weights):
    import mlx.core as mx

    logp = logits-mx.logsumexp(logits,axis=-1,keepdims=True)
    chosen = mx.take_along_axis(logp,actions[:,None],axis=-1)[:,0]
    positive = mx.maximum(returns-values,0)
    actor = -mx.mean(weights*chosen*mx.stop_gradient(positive))
    critic = .5*mx.mean(weights*mx.square(positive))
    return actor, critic, positive


class SelfImitation:
    def __init__(self, agent, loss_weight=.1, value_weight=.01):
        if (not np.isfinite(loss_weight) or not np.isfinite(value_weight)
                or loss_weight <= 0 or value_weight < 0):
            raise ValueError("invalid SIL loss weights")
        self.agent, self.loss_weight, self.value_weight = agent, loss_weight, value_weight
        # This auxiliary path deliberately uses ordinary lazy MLX operations.
        # nn.value_and_grad temporarily updates model parameters; splitting it
        # across separately compiled functions can leave uncaptured state.
        # The existing PPO update/inference paths remain compiled and unchanged.
        self.updates = 0

    def _loss(self, model, obs, actions, returns, weights):
        logits, values = model.policy_value(obs)
        actor, critic, positive = sil_terms(logits,values,actions,returns,weights)
        return self.loss_weight*(actor+self.value_weight*critic), (actor,critic,positive)

    def _gradients(self, obs, actions, returns, weights):
        import mlx.nn as nn

        return nn.value_and_grad(self.agent.model,self._loss)(
            self.agent.model,obs,actions,returns,weights)

    def _apply(self, grads):
        import mlx.optimizers as optim

        grads, norm = optim.clip_grad_norm(grads,.5)
        self.agent.optimizer.update(self.agent.model,grads)
        return norm

    def train(self, batch):
        import mlx.core as mx

        (loss, (actor,critic,positive)), grads = self._gradients(*(mx.array(x) for x in batch))
        mx.eval(loss,actor,critic,positive,grads)
        if not np.isfinite(float(loss.item())):
            raise RuntimeError("Non-finite SIL loss")
        advantages = np.array(positive)
        applied = bool(np.any(advantages>0))
        if applied:
            norm = self._apply(grads)
            mx.eval(norm,self.agent.state)
            self.updates += 1
        # Skipping all-zero batches also avoids updates from Adam momentum.
        return advantages, dict(loss=float(loss.item()),actor_loss=float(actor.item()),
                                value_loss=float(critic.item()),
                                positive_fraction=float(np.mean(advantages>0)),applied=applied)
