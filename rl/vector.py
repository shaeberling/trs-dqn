"""One existing native emulator per spawned process."""

import multiprocessing as mp
import traceback

import numpy as np

from .env import BreakdownEnv


def worker(pipe, seed, config):
    env = None
    try:
        env = BreakdownEnv(seed=seed, **config)
        pipe.send(("ok", env.reset()))
        while True:
            command, value = pipe.recv()
            if command == "close":
                break
            if command == "reset":
                pipe.send(("ok", env.reset(value)))
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
    def __init__(self, count=8, seed=0, **config):
        context = mp.get_context("spawn")
        self.pipes, self.processes = [], []
        for i in range(count):
            parent, child = context.Pipe()
            process = context.Process(target=worker, args=(child, seed+i, config), daemon=True)
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

    def step(self, actions):
        for p, action in zip(self.pipes, actions, strict=True):
            p.send(("step", int(action)))
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
