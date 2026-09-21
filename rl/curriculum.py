"""Training-only restarts from level entries reached by the agent itself.

A Go-Explore-inspired start-state curriculum, not a reproduction of the full
algorithm. The current policy still chooses every action and learns only from
new score differences. Evaluation always uses the ordinary BreakdownEnv.
"""

import operator

import numpy as np

from .env import BreakdownEnv, validate_observation_stride
from .snapshot import Snapshot, capture, restore


class CurriculumEnv(BreakdownEnv):
    def __init__(self, seed=0, curriculum_probability=.5, curriculum_min_level=2,
                 curriculum_per_level=8, curriculum_share=False, worker_id=None,
                 curriculum_reset=True, curriculum_score_interval=0, **config):
        if isinstance(curriculum_score_interval, (bool, np.bool_)):
            raise ValueError("Score interval must be a nonnegative integer")
        curriculum_score_interval = operator.index(curriculum_score_interval)
        if (not 0 <= curriculum_probability <= 1 or curriculum_min_level < 2
                or curriculum_per_level < 1 or curriculum_score_interval < 0):
            raise ValueError("Invalid curriculum probability, minimum level, or archive capacity")
        if curriculum_share and (not curriculum_probability or worker_id is None or worker_id < 0):
            raise ValueError("Shared curriculum requires a positive reset probability and worker ID")
        super().__init__(seed=seed, **config)
        self.curriculum_probability = curriculum_probability
        self.curriculum_min_level = curriculum_min_level
        self.curriculum_per_level = curriculum_per_level
        self.curriculum_share, self.worker_id = curriculum_share, worker_id
        self.curriculum_reset = curriculum_reset
        self.curriculum_score_interval = curriculum_score_interval
        self.archive, self.encounters = {}, {}
        self.curriculum_rng = np.random.default_rng(np.random.SeedSequence([seed, 719]))
        self.total_actions = 0

    def reset(self, seed=None):
        # Explicit seeded resets always boot; only the training worker's ordinary
        # automatic reset may choose an archived start. Boot RNG is not rewound.
        if (seed is None and self.curriculum_reset and self.archive
                and self.curriculum_rng.random() < self.curriculum_probability):
            level = int(self.curriculum_rng.choice(sorted(self.archive)))
            bank = self.archive[level]
            saved = bank[int(self.curriculum_rng.integers(len(bank)))]
            obs = restore(self, saved)
            self.steps, self.episode_reward = 0, 0.0
            self.full_game = False
            self.segment_source_worker = saved.source_worker
            self.segment_source_action = saved.source_action
        else:
            obs = super().reset(seed)
            self.full_game = True
            self.segment_source_worker = self.segment_source_action = None
        self.segment_start_score, self.segment_start_level = self.score, self.level
        self.last_archive_score = self.score
        return obs

    def _reserve_slot(self, level):
        count = self.encounters.get(level, 0) + 1
        self.encounters[level] = count
        bank = self.archive.setdefault(level, [])
        slot = len(bank) if len(bank) < self.curriculum_per_level else int(
            self.curriculum_rng.integers(count))
        return slot if slot < self.curriculum_per_level else None

    def _install(self, saved, slot):
        bank = self.archive[saved.level]
        if slot == len(bank):
            bank.append(saved)
        else:
            bank[slot] = saved

    def receive_archive(self, entries):
        """Receive same-run self-play entries routed by the parent, never files.

        Installing an entry changes only the archive, not the active emulator,
        observation, policy RNG, reward, or current episode.
        """
        if not self.curriculum_share:
            raise ValueError("Archive sharing is disabled")
        entries = tuple(entries)
        for saved in entries:
            if not isinstance(saved, Snapshot):
                raise ValueError("Invalid peer archive entry")
            stride = validate_observation_stride(saved.observation_stride)
            if (saved.tstates != self.tstates or stride != self.observation_stride
                    or saved.level < self.curriculum_min_level
                    or saved.source_worker is None or saved.source_worker == self.worker_id
                    or saved.source_action is None or saved.source_action < 1
                    or saved.frames.shape != (3*stride+1, 16, 64) or saved.frames.dtype != np.uint8):
                raise ValueError("Invalid peer archive entry")
        for saved in entries:
            slot = self._reserve_slot(saved.level)
            if slot is not None:
                saved.frames.flags.writeable = False
                self._install(saved, slot)

    def step(self, action):
        previous_level = self.level
        obs, reward, terminal, truncated, info = super().step(action)
        self.total_actions += 1
        info.update(full_game=self.full_game, segment_start_score=self.segment_start_score,
                    segment_start_level=self.segment_start_level,
                    segment_source_worker=self.segment_source_worker,
                    segment_source_action=self.segment_source_action)
        level_entry = self.level > previous_level
        progress = self.score-self.last_archive_score
        score_entry = (self.curriculum_score_interval > 0 and not level_entry
                       and progress >= self.curriculum_score_interval)
        if level_entry or score_entry:
            self.last_archive_score = self.score
        if (self.curriculum_probability and (level_entry or score_entry)
                and self.level >= self.curriculum_min_level and not self.done):
            # Reservoir sampling keeps a bounded set of self-reached entries at
            # each visible level. It never creates or edits a game state.
            slot = self._reserve_slot(self.level)
            if slot is not None or self.curriculum_share:
                saved = capture(self)
                if slot is not None:
                    self._install(saved, slot)
                info["curriculum_archive_add"] = dict(
                    level=self.level, score=self.score, slot=slot, source_action=self.total_actions,
                    source_episode_steps=self.steps, source_full_game=self.full_game,
                    source_start_level=self.segment_start_level, start_tstates=self.start_tstates,
                    retained=slot is not None, shared=self.curriculum_share)
                if self.curriculum_score_interval:
                    info["curriculum_archive_add"].update(
                        entry_kind="level_entry" if level_entry else "score_progress",
                        progress_since_previous_archive=progress,
                        source_start_score=self.segment_start_score,
                        source_parent_worker=self.segment_source_worker,
                        source_parent_action=self.segment_source_action)
                if self.curriculum_share:
                    # VectorEnv removes this private payload before returning
                    # anything to the learner or logger. It is never model input.
                    info["_curriculum_snapshot"] = saved
        if terminal or truncated:
            info["curriculum_archive_counts"] = {str(k): len(v) for k, v in self.archive.items()}
        return obs, reward, terminal, truncated, info
