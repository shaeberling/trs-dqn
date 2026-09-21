"""Training-only resets to unmodified states reached by this learner itself.

Archive selection uses visible stage and score earned since that stage/life
began. No action script, hidden-state labels, demonstration files or bonuses.
Complete-game evaluation always uses the ordinary DefenseEnv instead.
"""

import numpy as np

from .defense import DefenseEnv, ENVIRONMENT_VERSION, GAME_SHA256, positive_integer
from .defense_snapshot import DefenseSnapshot, capture, restore


class DefenseCurriculumEnv(DefenseEnv):
    def __init__(self, seed=0, curriculum_probability=.5, curriculum_score_interval=20,
                 curriculum_per_bin=4, curriculum_bins=16, curriculum_share=False,
                 worker_id=None, curriculum_reset=True, **config):
        if not np.isfinite(curriculum_probability) or not 0 <= curriculum_probability <= 1:
            raise ValueError("invalid curriculum probability")
        self.interval = positive_integer(curriculum_score_interval, "score interval", allow_zero=True)
        self.per_bin = positive_integer(curriculum_per_bin, "entries per bin")
        self.bins = positive_integer(curriculum_bins, "bins per stage")
        if worker_id is not None:
            worker_id = positive_integer(worker_id, "worker ID", allow_zero=True)
        if curriculum_share and (not curriculum_probability or worker_id is None):
            raise ValueError("shared curriculum requires a worker ID and positive probability")
        super().__init__(seed=seed, **config)
        self.curriculum_probability = curriculum_probability
        self.curriculum_share, self.worker_id = curriculum_share, worker_id
        self.curriculum_reset = curriculum_reset
        self.curriculum_rng = np.random.default_rng(np.random.SeedSequence([seed, 719]))
        self.archive, self.encounters = {}, {}
        self.total_actions = 0
        self.progress_start_score = 0

    def _key(self, stage, score, start_score):
        return stage, (score-start_score)//self.interval if self.interval else 0

    def _reserve_slot(self, key):
        if key not in self.archive:
            peers = sorted(k for k in self.archive if k[0] == key[0])
            if len(peers) >= self.bins:
                if key <= peers[0]:
                    return None
                del self.archive[peers[0]], self.encounters[peers[0]]
            self.archive[key], self.encounters[key] = [], 0
        self.encounters[key] += 1
        bank = self.archive[key]
        slot = len(bank) if len(bank) < self.per_bin else int(
            self.curriculum_rng.integers(self.encounters[key]))
        return slot if slot < self.per_bin else None

    def _install(self, key, saved, slot):
        saved.frames.flags.writeable = False
        bank = self.archive[key]
        if slot == len(bank):
            bank.append(saved)
        else:
            bank[slot] = saved

    def reset(self, seed=None):
        if (seed is None and self.curriculum_reset and self.archive
                and self.curriculum_rng.random() < self.curriculum_probability):
            stage = int(self.curriculum_rng.choice(sorted({k[0] for k in self.archive})))
            keys = sorted(k for k in self.archive if k[0] == stage)
            bank = self.archive[keys[int(self.curriculum_rng.integers(len(keys)))]]
            saved = bank[int(self.curriculum_rng.integers(len(bank)))]
            obs = restore(self, saved)
            self.steps, self.missions, self.highest_stage = 0, 0, self.stage
            self.full_game = False
            self.segment_source_worker, self.segment_source_action = saved.source_worker, saved.source_action
        else:
            obs = super().reset(seed)
            self.full_game = True
            self.progress_start_score = 0
            self.segment_source_worker = self.segment_source_action = None
        self.segment_start_score, self.segment_start_stage = self.score, self.stage
        self.last_archive_key = self._key(self.stage, self.score, self.progress_start_score)
        return obs

    def receive_archive(self, entries):
        if not self.curriculum_share:
            raise ValueError("archive sharing is disabled")
        entries = tuple(entries)
        for saved in entries:
            if (not isinstance(saved, DefenseSnapshot) or saved.game_sha256 != GAME_SHA256
                    or saved.environment_version != ENVIRONMENT_VERSION
                    or (saved.tstates, saved.max_steps, saved.action_count) !=
                       (self.tstates, self.max_steps, len(self.actions))
                    or saved.source_worker is None or saved.source_worker == self.worker_id
                    or saved.source_action < 1 or not 1 <= saved.lives <= 4
                    or not 1 <= saved.stage <= saved.highest_stage <= 3
                    or not 0 <= saved.progress_start_score <= saved.score
                    or saved.frames.shape != (4, 16, 64) or saved.frames.dtype != np.uint8):
                raise ValueError("invalid same-run Defense peer snapshot")
        for saved in entries:
            key = self._key(saved.stage, saved.score, saved.progress_start_score)
            slot = self._reserve_slot(key)
            if slot is not None:
                self._install(key, saved, slot)

    def step(self, action):
        previous_stage = self.stage
        obs, reward, terminal, truncated, info = super().step(action)
        self.total_actions += 1
        stage_entry = self.stage != previous_stage
        if stage_entry or info["life_lost"]:
            self.progress_start_score = self.score
            self.last_archive_key = self._key(self.stage, self.score, self.progress_start_score)
        key = self._key(self.stage, self.score, self.progress_start_score)
        progress_entry = bool(self.interval and key != self.last_archive_key)
        self.last_archive_key = key
        info.update(full_game=self.full_game, segment_start_score=self.segment_start_score,
                    segment_start_stage=self.segment_start_stage,
                    segment_source_worker=self.segment_source_worker,
                    segment_source_action=self.segment_source_action,
                    episode_reward=float(self.score-self.segment_start_score))
        if (self.curriculum_probability and not self.done and not info["life_lost"]
                and (stage_entry or progress_entry)):
            slot = self._reserve_slot(key)
            if slot is not None or self.curriculum_share:
                saved = capture(self)
                if slot is not None:
                    self._install(key, saved, slot)
                info["curriculum_archive_add"] = dict(
                    stage=self.stage, score=self.score, progress=self.score-self.progress_start_score,
                    progress_bin=key[1], entry_kind="stage_entry" if stage_entry else "score_progress",
                    source_action=self.total_actions, source_episode_steps=self.steps,
                    source_full_game=self.full_game, retained=slot is not None, shared=self.curriculum_share,
                    source_parent_worker=self.segment_source_worker,
                    source_parent_action=self.segment_source_action)
                if self.curriculum_share:
                    info["_curriculum_snapshot"] = saved
        if terminal or truncated:
            info["curriculum_archive_counts"] = {f"{stage}:{bucket}": len(bank)
                                                for (stage, bucket), bank in self.archive.items()}
        return obs, reward, terminal, truncated, info
