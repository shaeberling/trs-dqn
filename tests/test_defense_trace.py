import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np

from rl.defense_trace import TraceLearner, trace_targets
from rl.defense_trace_replay import TraceNStep, TraceReplay
from rl.model import Learner
from rl.replay import NStep, Replay


def frame(value):
    return np.full((4, 16, 64), value, np.uint8)


class TraceTests(unittest.TestCase):
    def test_targets_greedy_prefix_deviation_terminal_truncation_and_ties(self):
        # Greedy online is action 1; target would prefer 0, which must not
        # choose either the trace boundary or the bootstrap action.
        online = mx.array(np.tile([1., 2.], (5, 3, 1)))
        target = mx.array(np.tile([100., 4.], (5, 3, 1)))
        actions = mx.array([[0, 1, 1], [0, 0, 1], [0, 1, 0],
                            [0, 1, -1], [0, 1, 1]])
        rewards = mx.array([[1., 2., 3.], [1., 200., 300.], [1., 2., 300.],
                            [1., 2., 0.], [1., 2., 3.]])
        discounts = mx.array([[.5, .5, .5]]*3+[[.5, .5, 0.], [0., .5, .5]])
        labels, lengths, cuts = trace_targets(online, target, actions, rewards, discounts)
        np.testing.assert_allclose(np.array(labels), [3.25, 3., 3., 3., 1.])
        np.testing.assert_array_equal(np.array(lengths), [3, 1, 2, 2, 1])
        np.testing.assert_array_equal(np.array(cuts), [False, True, True, False, False])
        # Root action never has to be greedy. Deterministic tie-breaking is
        # the same argmax used in evaluation: equal-valued other actions cut.
        labels, lengths, cuts = trace_targets(mx.ones((1, 2, 2)), mx.full((1, 2, 2), 4.),
            mx.array([[1, 1]]), mx.array([[1., 100.]]), mx.full((1, 2), .5))
        self.assertEqual(labels.item(), 3.)
        self.assertEqual(lengths.item(), 1)
        self.assertTrue(cuts.item())

    def test_targets_one_step_identity_and_stop_gradient(self):
        online = mx.array([[[1., 3.]], [[4., 2.]]])
        target = mx.array([[[100., 7.]], [[5., 100.]]])
        actions = mx.array([[0], [1]])
        rewards = mx.array([[2.], [4.]])
        discounts = mx.array([[.5], [0.]])
        labels, lengths, cuts = trace_targets(online, target, actions, rewards, discounts)
        np.testing.assert_allclose(np.array(labels), [5.5, 4.])
        np.testing.assert_array_equal(np.array(lengths), 1)
        self.assertFalse(np.array(cuts).any())
        gradient = mx.grad(lambda t: trace_targets(online, t, actions, rewards, discounts)[0].sum())(target)
        np.testing.assert_array_equal(np.array(gradient), 0.)

    def test_current_online_greediness_recomputed_for_the_same_recorded_path(self):
        actions = mx.array([[0, 1, 1]])
        rewards, discounts = mx.array([[1., 2., 3.]]), mx.full((1, 3), .5)
        target = mx.array([[[100., 4.]]*3])
        before = trace_targets(mx.array([[[1., 2.]]*3]), target, actions, rewards, discounts)
        after = trace_targets(mx.array([[[2., 1.]]*3]), target, actions, rewards, discounts)
        self.assertEqual(before[0].item(), 3.25)
        self.assertEqual(before[1].item(), 3)
        self.assertEqual(after[0].item(), 51.)
        self.assertEqual(after[1].item(), 1)
        np.testing.assert_array_equal(np.array(actions), [[0, 1, 1]])

    def test_replay_exact_paths_sampling_priorities_and_ring_references(self):
        replay = TraceReplay(3, 3)
        reference = Replay(3, compact=True)
        for j in range(20):
            size = j % 3+1
            sequence = [(frame(j+k), k, float(k+1), frame(j+k+1), False)
                        for k in range(size)]
            replay.add_sequence(sequence, .9)
            reference.add(sequence[0][0], 0, sum(.9**k*(k+1) for k in range(size)),
                          sequence[-1][3], .9**size)
            np.testing.assert_array_equal(replay.tree, reference.tree)
            rng1, rng2 = np.random.default_rng(j), np.random.default_rng(j)
            indices, batch = replay.sample(5, rng1, .7)
            expected_indices, expected = reference.sample(5, rng2, .7)
            np.testing.assert_array_equal(indices, expected_indices)
            np.testing.assert_array_equal(batch[0], expected[0])
            np.testing.assert_array_equal(batch[-1], expected[-1])
            self.assertEqual(rng1.bit_generator.state, rng2.bit_generator.state)
            slot = (replay.pos-1) % replay.capacity
            stored = replay.frame_storage.following[slot*3+np.arange(3)]
            np.testing.assert_array_equal(stored, [sequence[min(k, size-1)][3] for k in range(3)])
            np.testing.assert_array_equal(replay.sequence_actions[slot], list(range(size))+[-1]*(3-size))
            batch[0][:] = 255; batch[3][:] = 255
            np.testing.assert_array_equal(replay.obs[slot], sequence[0][0])
            indices = np.array([0, 0, min(1, replay.size-1)])
            errors = np.array([1., 2., 3.])
            replay.priorities(indices, errors); reference.priorities(indices, errors)
            np.testing.assert_array_equal(replay.tree, reference.tree)
            pool = replay.frame_storage.pool
            tables = [replay.obs, replay.next_obs, replay.frame_storage.following]
            counts = np.zeros(len(pool.frames), np.int64)
            for table in tables:
                ids = table.indices[table.indices >= 0]
                counts += np.bincount(ids, minlength=len(counts))
            np.testing.assert_array_equal(counts, pool.references)
            self.assertEqual(len(pool.lookup), np.count_nonzero(counts))
        self.assertEqual(replay.frame_storage.stats()['screen_index_bytes'], 3*(2+3)*4*4)

    def test_life_truncation_and_worker_queues_never_cross_into_reset(self):
        replay = TraceReplay(16, 5)
        buffers = [TraceNStep(replay, 5, .9) for _ in range(2)]
        buffers[0].append(frame(1), 0, 1., frame(2), False, False)
        buffers[1].append(frame(10), 2, 4., frame(11), False, False)
        buffers[0].append(frame(2), 1, 2., frame(3), True, False)
        self.assertFalse(buffers[0].queue)
        self.assertEqual(len(buffers[1].queue), 1)
        np.testing.assert_array_equal(replay.sequence_actions[:2], [[0, 1, -1, -1, -1], [1, -1, -1, -1, -1]])
        np.testing.assert_allclose(replay.sequence_discounts[:2], [[.9, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
        buffers[1].append(frame(11), 3, 5., frame(12), False, True)
        np.testing.assert_allclose(replay.sequence_discounts[2:4], [[.9, .9, 0, 0, 0], [.9, 0, 0, 0, 0]])
        buffers[0].append(frame(99), 4, 7., frame(100), False, True)
        np.testing.assert_array_equal(replay.obs[4], frame(99))
        np.testing.assert_array_equal(replay.next_obs[:4], [frame(3), frame(3), frame(12), frame(12)])

    def test_invalid_trace_inputs_and_cli(self):
        for args in [(0, 5), (8, 0), (8, 33), (True, 5), (8, True), (2**30, 5)]:
            with self.assertRaises(ValueError): TraceReplay(*args)
        replay = TraceReplay(8, 2)
        good = (frame(1), 0, 1., frame(2), False)
        for sequence, gamma in [([], .9), ([good]*3, .9), ([good], 1.),
                                ([(frame(1), 20, 1., frame(2), False)], .9),
                                ([(frame(1), 0, np.nan, frame(2), False)], .9),
                                ([(frame(1), 0, 1., frame(2), True), good], .9)]:
            with self.assertRaises(ValueError): replay.add_sequence(sequence, gamma)
            self.assertEqual(replay.size, 0)
        from rl import defense_dqn
        base = ['dqn', '--run', 'runs/invalid-trace-test', '--artifacts', 'runs/invalid-trace-artifacts',
                '--greedy-trace-cut']
        for extra in ([], ['--compact-replay', '--quantiles', '32'],
                      ['--compact-replay', '--bootstrap-heads', '5'], ['--compact-replay', '--n-step', '33']):
            with patch.object(sys, 'argv', base+extra), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as error:
                defense_dqn.main()
            self.assertEqual(error.exception.code, 2)

    def test_real_model_update_keeps_target_fixed_then_syncs(self):
        agent = TraceLearner(seed=12, horizon=3)
        replay = TraceReplay(8, 3)
        replay.add_sequence([(frame(i+128), i, float(i+1), frame(i+129), False) for i in range(3)], .997)
        _, batch = replay.sample(4, np.random.default_rng(8), .4)
        before = {k: np.array(v) for k, v in tree_flatten(agent.target.parameters())}
        loss, errors, q = agent.train(batch)
        self.assertTrue(np.isfinite([loss, q]).all())
        self.assertTrue(np.isfinite(errors).all())
        self.assertEqual(agent.trace_stats()['sampled_backups'], 4)
        for k, v in tree_flatten(agent.target.parameters()):
            np.testing.assert_array_equal(before[k], np.array(v))
        self.assertTrue(any(not np.array_equal(before[k], np.array(v))
                            for k, v in tree_flatten(agent.online.parameters())))
        agent.sync_target()
        for (_, online), (_, target) in zip(tree_flatten(agent.online.parameters()),
                                           tree_flatten(agent.target.parameters()), strict=True):
            np.testing.assert_array_equal(np.array(online), np.array(target))

    def test_native_default_identity_trace_training_and_exact_resume(self):
        from rl.defense_learning import load_policy, policy_description
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def run(name, extra):
                result = subprocess.run([sys.executable, '-m', 'rl.defense_dqn',
                    '--run', str(root/name), '--artifacts', str(root/(name+'-artifacts'))]+extra,
                    capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                return json.loads((root/name/'latest/state.json').read_text())
            common = ['--envs', '2', '--capacity', '32', '--compact-replay', '--warmup', '4',
                '--batch-size', '4', '--n-step', '2', '--train-every', '4', '--target-every', '2',
                '--steps', '64', '--eval-every', '1000', '--max-episode-steps', '3',
                '--mlx-cache-mb', '64', '--exploration-max-repeat', '64', '--epsilon-final', '.9',
                '--curriculum-probability', '.5', '--curriculum-share', '--curriculum-boot-envs', '1',
                '--curriculum-boot-epsilon', '.05', '--curriculum-lookback', '4']
            default = run('default', common)
            disabled = run('disabled', common+['--no-greedy-trace-cut'])
            for key in ('steps', 'updates', 'episodes', 'rng', 'persistent_exploration_rng'):
                self.assertEqual(default[key], disabled[key])
            enabled = run('enabled', common+['--greedy-trace-cut'])
            self.assertEqual(enabled['greedy_trace']['sampled_backups'], enabled['updates']*4)
            self.assertGreater(enabled['greedy_trace']['policy_cut_backups'], 0)
            self.assertGreater(enabled['greedy_trace']['backup_action_histogram'][1], 0)
            unchanged = run('unchanged', ['--resume', str(root/'enabled/latest'), '--steps', '64'])
            for key in ('steps', 'updates', 'episodes', 'rng', 'persistent_exploration_rng'):
                self.assertEqual(enabled[key], unchanged[key])
            self.assertEqual(unchanged['greedy_trace']['sampled_backups'], 0)
            for left, right in [('default', 'disabled'), ('enabled', 'unchanged')]:
                for file in ('model.safetensors', 'target.safetensors', 'optimizer.npz'):
                    a = mx.load(str(root/left/'latest'/file)); b = mx.load(str(root/right/'latest'/file))
                    self.assertEqual(set(a), set(b))
                    for key in a: np.testing.assert_array_equal(np.array(a[key]), np.array(b[key]))
            resumed = run('resumed', ['--resume', str(root/'enabled/latest'), '--steps', '96'])
            self.assertTrue(resumed['config']['greedy_trace_cut'])
            self.assertGreater(resumed['updates'], enabled['updates'])
            self.assertEqual(resumed['greedy_trace']['sampled_backups'],
                             (resumed['updates']-enabled['updates'])*4)
            policy, config = load_policy(root/'enabled/latest/model.safetensors')
            self.assertEqual(policy_description(config), 'learned Q-values, greedy')
            self.assertFalse(hasattr(policy, 'remaining'))
            rows = [json.loads(l) for l in (root/'enabled/metrics.jsonl').read_text().splitlines()]
            workers = next(r['workers'] for r in rows if r['event']=='workers_started')
            self.assertTrue(all(not w['mlx_loaded'] for w in workers))

    def test_native_trace_with_own_archives_and_reserved_boot_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = [sys.executable, '-m', 'rl.defense_dqn', '--run', str(root/'run'),
                '--artifacts', str(root/'artifacts'), '--envs', '2', '--capacity', '4096',
                '--compact-replay', '--warmup', '32', '--batch-size', '4', '--train-every', '256',
                '--steps', '4096', '--eval-every', '10000', '--max-episode-steps', '512',
                '--mlx-cache-mb', '64', '--n-step', '5', '--greedy-trace-cut',
                '--exploration-max-repeat', '64', '--epsilon-final', '1',
                '--curriculum-probability', '1', '--curriculum-share', '--curriculum-boot-envs', '1',
                '--curriculum-boot-epsilon', '.05', '--curriculum-lookback', '4']
            result = subprocess.run(command, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            rows = [json.loads(l) for l in (root/'run/metrics.jsonl').read_text().splitlines()]
            archives = [r for r in rows if r['event']=='curriculum_archive']
            episodes = [r for r in rows if r['event']=='episode']
            self.assertTrue(archives)
            self.assertTrue(all(r['trigger_action']-r['source_action']==4 for r in archives))
            self.assertTrue(all(r['full_game'] for r in episodes if r['worker']==0))
            self.assertTrue(any(not r['full_game'] for r in episodes if r['worker']==1))
            state = json.loads((root/'run/latest/state.json').read_text())
            self.assertEqual(state['greedy_trace']['sampled_backups'], state['updates']*4)
            self.assertGreater(state['greedy_trace']['policy_cut_backups'], 0)
            self.assertGreater(state['greedy_trace']['backup_action_histogram'][-1], 0)
            self.assertGreater(state['restored_segments'], 0)


if __name__ == '__main__':
    unittest.main()
