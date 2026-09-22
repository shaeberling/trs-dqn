"""Training-only resets to unmodified states reached by this learner itself.

Archive selection uses visible stage and score, coarse screens or an own-action
counter since the visible life/stage boundary. No hidden-state labels or bonuses.
Complete-game evaluation always uses the ordinary DefenseEnv instead.
"""

from collections import deque

import numpy as np

from .defense import DefenseEnv, ENVIRONMENT_VERSION, GAME_SHA256, positive_integer
from .defense_snapshot import DefenseSnapshot, capture, restore
from .defense_cells import screen_cell


class DefenseCurriculumEnv(DefenseEnv):
    def __init__(self, seed=0, curriculum_probability=.5, curriculum_score_interval=20,
                 curriculum_per_bin=4, curriculum_bins=16, curriculum_share=False,
                 worker_id=None, curriculum_reset=True, curriculum_lookback=0,
                 curriculum_cells="score", curriculum_screen_interval=32,
                 curriculum_age_interval=32, curriculum_trigger="progress",
                 curriculum_restored_life_only=False, **config):
        if not np.isfinite(curriculum_probability) or not 0 <= curriculum_probability <= 1:
            raise ValueError("invalid curriculum probability")
        self.interval = positive_integer(curriculum_score_interval, "score interval", allow_zero=True)
        self.per_bin = positive_integer(curriculum_per_bin, "entries per bin")
        self.bins = positive_integer(curriculum_bins, "bins per stage")
        self.lookback = positive_integer(curriculum_lookback, "curriculum lookback", allow_zero=True)
        if curriculum_trigger not in ("progress", "life-loss"):
            raise ValueError("invalid curriculum trigger")
        if curriculum_trigger == "life-loss" and not self.lookback:
            raise ValueError("life-loss archive requires a positive lookback")
        self.trigger = curriculum_trigger
        if not isinstance(curriculum_restored_life_only, (bool, np.bool_)):
            raise ValueError("restored-life-only must be boolean")
        self.restored_life_only = bool(curriculum_restored_life_only)
        if curriculum_cells not in ("score", "screen", "age"):
            raise ValueError("invalid curriculum cell representation")
        self.cells = curriculum_cells
        self.screen_interval = positive_integer(curriculum_screen_interval, "screen cell interval")
        self.age_interval = positive_integer(curriculum_age_interval, "life age interval")
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
        self.history = deque(maxlen=self.lookback+1)
        self.total_actions = 0
        self.progress_start_score = 0
        self.life_steps = 0

    def _key(self, stage, score, start_score, frames=None, life_steps=None):
        if self.cells == "screen":
            return stage, screen_cell(frames)
        if self.cells == "age":
            return stage, (self.life_steps if life_steps is None else life_steps)//self.age_interval
        return stage, (score-start_score)//self.interval if self.interval else 0

    def _reserve_slot(self, key):
        if key not in self.archive:
            peers = sorted(k for k in self.archive if k[0] == key[0])
            if len(peers) >= self.bins:
                if self.cells == "screen":
                    # Bottom-k hash priorities retain a bounded, score-independent
                    # sample of distinct cells. Frequent revisits cannot crowd
                    # out rare cells merely by being encountered more often.
                    victim = peers[-1]
                    if key >= victim:
                        return None
                else:
                    victim = peers[0]
                    if key <= victim:
                        return None
                del self.archive[victim], self.encounters[victim]
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
        self.history.clear()
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
            self.life_steps = 0
            self.segment_source_worker = self.segment_source_action = None
        self.segment_start_score, self.segment_start_stage = self.score, self.stage
        self.last_archive_key = self._key(self.stage, self.score, self.progress_start_score, obs)
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
                    or not isinstance(saved.life_steps, (int, np.integer))
                    or isinstance(saved.life_steps, (bool, np.bool_)) or saved.life_steps < 0
                    or not isinstance(saved.observation_stride, (int, np.integer))
                    or isinstance(saved.observation_stride, (bool, np.bool_))
                    or saved.observation_stride != self.observation_stride
                    or saved.frames.shape != (3*self.observation_stride+1, 16, 64)
                    or saved.frames.dtype != np.uint8):
                raise ValueError("invalid same-run Defense peer snapshot")
        for saved in entries:
            key = self._key(saved.stage, saved.score, saved.progress_start_score,
                            saved.frames[::saved.observation_stride], saved.life_steps)
            slot = self._reserve_slot(key)
            if slot is not None:
                self._install(key, saved, slot)

    def step(self, action):
        previous_stage = self.stage
        obs, reward, terminal, truncated, info = super().step(action)
        self.total_actions += 1
        self.life_steps += 1
        stage_entry = self.stage != previous_stage
        if (self.curriculum_probability and self.trigger == "life-loss"
                and info["life_lost"] and not stage_entry):
            # The history still belongs to the lost life. Select an actual
            # earlier state BEFORE clearing it or changing its score baseline.
            # A visible loss is not a known physical-collision timestamp.
            self._archive_before_loss(info)
        if stage_entry or info["life_lost"]:
            self.progress_start_score = self.score
            self.life_steps = 0
            self.last_archive_key = self._key(self.stage, self.score, self.progress_start_score, obs)
        if self.curriculum_probability and self.lookback:
            # Keep only actually visited states within this life and stage.
            # Neither the policy nor the reward receives this opaque history.
            if stage_entry or info["life_lost"] or self.done:
                self.history.clear()
            if not self.done and not info["life_lost"]:
                self.history.append(capture(self))
        key = self._key(self.stage, self.score, self.progress_start_score, obs)
        if self.cells == "screen":
            sample = self.total_actions % self.screen_interval == 0
            progress_entry = sample and key != self.last_archive_key
            if sample:
                self.last_archive_key = key
        else:
            progress_entry = bool((self.cells == "age" or self.interval) and key != self.last_archive_key)
            self.last_archive_key = key
        info.update(full_game=self.full_game, segment_start_score=self.segment_start_score,
                    segment_start_stage=self.segment_start_stage,
                    segment_source_worker=self.segment_source_worker,
                    segment_source_action=self.segment_source_action,
                    episode_reward=float(self.score-self.segment_start_score))
        if (self.curriculum_probability and not self.done and not info["life_lost"]
                and (stage_entry or (self.trigger == "progress" and progress_entry))):
            saved = None
            if self.lookback and not stage_entry:
                # A new stage is always captured immediately. Other archive
                # events can select an earlier own-play state, giving a future
                # reset more lead-in before the triggering event.
                if len(self.history) <= self.lookback:
                    return obs, reward, terminal, truncated, info
                saved = self.history[0]
                key = self._key(saved.stage, saved.score, saved.progress_start_score,
                                saved.frames[::saved.observation_stride], saved.life_steps)
            slot = self._reserve_slot(key)
            if slot is not None or self.curriculum_share:
                if saved is None:
                    saved = capture(self)
                if slot is not None:
                    self._install(key, saved, slot)
                info["curriculum_archive_add"] = dict(
                    stage=saved.stage, score=saved.score, progress=saved.score-saved.progress_start_score,
                    progress_bin=(saved.score-saved.progress_start_score)//self.interval if self.interval else 0,
                    entry_kind="stage_entry" if stage_entry else (
                        "screen_cell" if self.cells == "screen" else
                        "life_age" if self.cells == "age" else "score_progress"),
                    source_action=saved.source_action, source_episode_steps=saved.steps,
                    source_full_game=saved.source_full_game, retained=slot is not None, shared=self.curriculum_share,
                    source_parent_worker=self.segment_source_worker,
                    source_parent_action=self.segment_source_action)
                if self.cells == "screen":
                    info["curriculum_archive_add"]["screen_cell"] = key[1]
                if self.cells == "age":
                    info["curriculum_archive_add"].update(
                        life_steps=saved.life_steps, life_age_bin=key[1],
                        trigger_life_steps=self.life_steps)
                if self.lookback:
                    info["curriculum_archive_add"].update(
                        lookback_actions=self.total_actions-saved.source_action,
                        trigger_action=self.total_actions, trigger_score=self.score,
                        trigger_progress=self.score-self.progress_start_score)
                if self.curriculum_share:
                    info["_curriculum_snapshot"] = saved
        if (self.restored_life_only and not self.full_game and info["life_lost"]
                and not terminal and not truncated):
            # End only an already-restored training segment. Native lives and
            # score are not changed, and the actual loss screen is returned.
            # The vector wrapper then performs its ordinary own-state/boot
            # reset. Boot games, including reserved workers, remain complete.
            self.done = truncated = True
            info.update(truncated=True, curriculum_life_cut=True)
        if terminal or truncated:
            info["curriculum_archive_counts"] = {f"{stage}:{bucket}": len(bank)
                                                for (stage, bucket), bank in self.archive.items()}
        return obs, reward, terminal, truncated, info

    def _archive_before_loss(self, info):
        # Current (loss) state was not appended; -L is exactly L actions ago.
        # No borrowing from a previous life/reset when this history is short.
        if len(self.history) < self.lookback:
            return
        saved = self.history[-self.lookback]
        if self.total_actions-saved.source_action != self.lookback:
            raise RuntimeError("noncontiguous own-life snapshot history")
        key = self._key(saved.stage, saved.score, saved.progress_start_score,
                        saved.frames[::saved.observation_stride], saved.life_steps)
        slot = self._reserve_slot(key)
        if slot is None and not self.curriculum_share:
            return
        if slot is not None:
            self._install(key, saved, slot)
        event = dict(
            stage=saved.stage, score=saved.score, progress=saved.score-saved.progress_start_score,
            progress_bin=(saved.score-saved.progress_start_score)//self.interval if self.interval else 0,
            entry_kind="life_loss_lookback", source_action=saved.source_action,
            source_episode_steps=saved.steps, source_full_game=saved.source_full_game,
            source_lives=saved.lives, source_life_steps=saved.life_steps,
            retained=slot is not None, shared=self.curriculum_share,
            source_parent_worker=self.segment_source_worker,
            source_parent_action=self.segment_source_action,
            lookback_actions=self.total_actions-saved.source_action,
            trigger_action=self.total_actions, trigger_score=self.score,
            trigger_progress=self.score-self.progress_start_score,
            trigger_life_steps=self.life_steps, trigger_lives=self.lives,
            trigger_terminal=bool(info["terminated"]))
        if self.cells == "screen":
            event["screen_cell"] = key[1]
        if self.cells == "age":
            event.update(life_steps=saved.life_steps, life_age_bin=key[1])
        info["curriculum_archive_add"] = event
        if self.curriculum_share:
            info["_curriculum_snapshot"] = saved
