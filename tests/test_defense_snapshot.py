from dataclasses import replace
import json
import multiprocessing as mp
from pathlib import Path
import unittest

import numpy as np

from rl.defense import DefenseEnv
from rl.defense_snapshot import capture, restore


def peer_continuation(pipe, saved, actions):
    env = DefenseEnv(max_steps=0)
    try:
        env.reset(99)
        restore(env, saved)
        rows = []
        for action in actions:
            obs, reward, done, truncated, info = env.step(int(action))
            rows.append((obs.tobytes(), reward, done, truncated, info))
        pipe.send(rows)
    finally:
        env.close()
        pipe.close()


class DefenseSnapshotTests(unittest.TestCase):
    def test_exact_learned_trace_continuation_in_same_and_peer_process(self):
        # Existing learned actions are an emulator regression fixture only,
        # never training examples or a controller for a new policy.
        path = Path("results/defense/learned/versions/step-000001133312-fede3ea00c70-seed-10000/trace.npz")
        with np.load(path) as trace:
            metadata = json.loads(str(trace["metadata"]))
            actions, frames, rewards = trace["actions"], trace["frames"], trace["rewards"]
        env = DefenseEnv(max_steps=0)
        try:
            env.reset(metadata["result"]["seed"])
            for action in actions[:80]:
                env.step(int(action))
            saved = capture(env)
            self.assertGreater(saved.score, 0)
            self.assertFalse(saved.frames.flags.writeable)
            expected = []
            for index, action in enumerate(actions[80:], 80):
                obs, reward, done, truncated, info = env.step(int(action))
                np.testing.assert_array_equal(obs[-1], frames[index+1])
                self.assertEqual(reward, rewards[index])
                expected.append((obs.tobytes(), reward, done, truncated, info))
            self.assertTrue(expected[-1][2])
            with self.assertRaises(ValueError):
                capture(env)
            env.reset(99)
            rng_state = env.rng.bit_generator.state
            env.close()  # Deliberately disarm native game-over stop; restore must recover it.
            np.testing.assert_array_equal(restore(env, saved), saved.frames)
            self.assertEqual(env.rng.bit_generator.state, rng_state)
            for action, original in zip(actions[80:], expected, strict=True):
                obs, reward, done, truncated, info = env.step(int(action))
                self.assertEqual((obs.tobytes(), reward, done, truncated, info), original)
            self.assertEqual(env.score-saved.score, sum(row[1] for row in expected))
        finally:
            env.close()
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(target=peer_continuation, args=(child, saved, actions[80:]))
        process.start()
        child.close()
        try:
            self.assertTrue(parent.poll(30), "peer Defense continuation did not finish")
            self.assertEqual(parent.recv(), expected)
            process.join(timeout=10)
            self.assertEqual(process.exitcode, 0)
        finally:
            if process.is_alive():
                process.terminate()
                process.join()
            parent.close()

    def test_invalid_snapshot_does_not_mutate_state(self):
        env = DefenseEnv(max_steps=0)
        try:
            env.reset(12)
            saved = capture(env)
            for invalid in (replace(saved, native=saved.native[:-1]),
                            replace(saved, native=b"FAIL"+saved.native[4:]),
                            replace(saved, tstates=50000), replace(saved, max_steps=10),
                            replace(saved, action_count=21), replace(saved, game_sha256="wrong"),
                            replace(saved, lives=0), replace(saved, highest_stage=0),
                            replace(saved, frames=saved.frames[:3])):
                with self.assertRaises(ValueError):
                    restore(env, invalid)
                self.assertEqual(capture(env).native, saved.native)
                np.testing.assert_array_equal(np.stack(env.frames), saved.frames)
        finally:
            env.close()

    def test_optional_enter_profile_roundtrip(self):
        env = DefenseEnv(max_steps=0, allow_enter=True)
        try:
            env.reset(7)
            for _ in range(80):
                env.step(20)
            saved = capture(env)
            expected = env.step(20)
            restore(env, saved)
            actual = env.step(20)
            np.testing.assert_array_equal(actual[0], expected[0])
            self.assertEqual(actual[1:], expected[1:])
            self.assertEqual(env._mission_visible, saved.mission_visible)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
