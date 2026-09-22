"""Multi-option returns from actual own experience at the base-action discount.

The acting policy and hold executor are unchanged. Only completed option
transitions are queued; boundaries flush every remaining start without crossing
lives or episodes. No shorter-duration counterfactuals are synthesized.
"""

from collections import deque
import operator

from .defense_repeat import OptionReturns


class MultiOptionReturns:
    def __init__(self, replay, n=5, gamma=.997, action_count=20, durations=(1, 4, 16, 64)):
        if isinstance(n, bool):
            raise ValueError('option horizon must be an integer in 1..32')
        try:
            n = operator.index(n)
        except TypeError as error:
            raise ValueError('option horizon must be an integer in 1..32') from error
        if not 1 <= n <= 32:
            raise ValueError('option horizon must be an integer in 1..32')
        self.replay, self.n = replay, n
        self.queue = deque()
        self.emitted = 0
        self.option = OptionReturns(self, gamma, action_count, durations)

    @property
    def completed(self):
        return self.option.completed

    @property
    def interrupted(self):
        return self.option.interrupted

    def begin(self, observation, option):
        self.option.begin(observation, option)

    def append(self, reward, following, terminated, truncated):
        finished = self.option.append(reward, following, terminated, truncated)
        if terminated or truncated:
            while self.queue:
                self._emit()
        return finished

    def add(self, first, option, reward, following, discount):
        # OptionReturns owns first; following must survive the next vector step.
        self.queue.append((first, option, reward, following.copy(), discount))
        if len(self.queue) >= self.n:
            self._emit()

    def _emit(self):
        reward, discount = 0., 1.
        for item in self.queue:
            reward += discount * item[2]
            discount *= item[4]
            if discount == 0.:
                break
        first = self.queue.popleft()
        self.replay.add(first[0], first[1], reward, item[3], discount)
        self.emitted += 1
