"""One existing native emulator per spawned process."""

import multiprocessing as mp
import operator
import os
import sys
import traceback

import numpy as np

from .env import BreakdownEnv


def worker(pipe, seed, config):
    env = None
    try:
        config = dict(config)
        curriculum = config.pop("curriculum", False)
        if curriculum:
            from .curriculum import CurriculumEnv
            env = CurriculumEnv(seed=seed, **config)
        else:
            env = BreakdownEnv(seed=seed, **config)
        pipe.send(("ok", env.reset()))
        while True:
            command, value = pipe.recv()
            if command == "close":
                break
            if command == "reset":
                pipe.send(("ok", env.reset(value)))
            elif command == "runtime":
                # Observational host diagnostics only; no emulation or RNG step.
                info = {"pid": os.getpid(), "mlx_loaded": "mlx.core" in sys.modules}
                if curriculum:
                    info["curriculum_reset_enabled"] = env.curriculum_reset
                pipe.send(("ok", info))
            elif command == "archive" and curriculum:
                env.receive_archive(value)
                pipe.send(("ok", None))
            elif command == "step":
                obs, reward, terminated, truncated, info = env.step(value)
                reset = env.reset() if terminated or truncated else None
                pipe.send(("ok", (obs, reward, terminated, truncated, info, reset)))
            else:
                raise ValueError(command)
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:
        pipe.send(("error", traceback.format_exc()))
    finally:
        if env is not None:
            env.close()
        pipe.close()


class VectorEnv:
    def __init__(self, count=8, seed=0, curriculum_boot_envs=0, **config):
        if count < 1:
            raise ValueError("count must be positive")
        curriculum_boot_envs = operator.index(curriculum_boot_envs)
        if not 0 <= curriculum_boot_envs < count:
            raise ValueError("curriculum_boot_envs must be between zero and count minus one")
        self.share_curriculum = bool(config.get("curriculum_share", False))
        if self.share_curriculum and not config.get("curriculum", False):
            raise ValueError("Shared archives require curriculum workers")
        if curriculum_boot_envs and not self.share_curriculum:
            raise ValueError("Reserved boot workers require shared curriculum")
        context = mp.get_context("spawn")
        self.pipes, self.processes = [], []
        for i in range(count):
            parent, child = context.Pipe()
            worker_config = {**config, "worker_id": i} if config.get("curriculum") else config
            if curriculum_boot_envs:
                # These workers still discover/share entries, but never restore
                # one. They protect a fixed fraction of from-boot action samples.
                worker_config["curriculum_reset"] = i >= curriculum_boot_envs
            process = context.Process(target=worker, args=(child, seed+i, worker_config), daemon=True)
            process.start()
            child.close()
            self.pipes.append(parent)
            self.processes.append(process)
        self.observations = np.stack([self._receive(p) for p in self.pipes])

    def _receive(self, pipe):
        if not pipe.poll(120):
            raise RuntimeError("Emulator worker did not respond within 120 seconds")
        kind, value = pipe.recv()
        if kind != "ok":
            raise RuntimeError(value)
        return value

    def _select(self, indices):
        indices = list(range(len(self.pipes))) if indices is None else [operator.index(i) for i in indices]
        if len(set(indices)) != len(indices) or any(i < 0 or i >= len(self.pipes) for i in indices):
            raise ValueError("worker indices must be unique and in range")
        return [self.pipes[i] for i in indices]

    def reset(self, seeds, *, indices=None):
        pipes, seeds = self._select(indices), list(seeds)
        if not pipes or len(pipes) != len(seeds):
            raise ValueError("provide one seed per selected worker")
        for p, seed in zip(pipes, seeds, strict=True):
            p.send(("reset", seed))
        return np.stack([self._receive(p) for p in pipes])

    def step(self, actions, *, indices=None):
        indices = list(range(len(self.pipes))) if indices is None else list(indices)
        pipes, actions = self._select(indices), [int(action) for action in actions]
        if len(pipes) != len(actions):
            raise ValueError("provide one action per selected worker")
        for p, action in zip(pipes, actions, strict=True):
            p.send(("step", int(action)))
        results = [self._receive(p) for p in pipes]
        entries = []
        for index, result in zip(indices, results, strict=True):
            saved = result[4].pop("_curriculum_snapshot", None)
            if saved is not None:
                if not self.share_curriculum or saved.source_worker != index:
                    raise RuntimeError("Unexpected curriculum snapshot provenance")
                entries.append((index, saved))
                result[4]["curriculum_archive_add"]["shared_with"] = len(self.pipes)-1
        if entries:
            pending = []
            for index, p in enumerate(self.pipes):
                peers = [saved for source, saved in entries if source != index]
                if peers:
                    p.send(("archive", peers))
                    pending.append(p)
            for p in pending:
                self._receive(p)
        return results

    def runtime(self):
        for p in self.pipes:
            p.send(("runtime", None))
        return [self._receive(p) for p in self.pipes]

    def close(self):
        for p in self.pipes:
            try:
                p.send(("close", None))
            except (OSError, BrokenPipeError):
                pass
        for p in self.processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
                p.join()
        for p in self.pipes:
            p.close()
