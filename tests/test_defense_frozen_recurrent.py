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

from rl.defense import ENVIRONMENT_VERSION, GAME_SHA256, action_names
from rl.defense_initialization import initialize_policy
from rl.defense_learning import load_policy, record_game, verify_policy_trace
from rl.defense_recurrent import RecurrentPPO
from rl.ppo import PPO
from rl.recurrent_policy import RECURRENT_ARCHITECTURE


def base_config():
    return dict(game='defense', algorithm='ppo', game_sha256=GAME_SHA256,
                environment_version=ENVIRONMENT_VERSION, action_names=list(action_names()),
                tstates=100000, observation_stride=1, eval_max_steps=0)


def arrays(tree):
    return {name: np.array(value) for name, value in tree_flatten(tree)}


class FrozenRecurrentTests(unittest.TestCase):
    def test_default_and_explicit_unfrozen_are_identical(self):
        default = RecurrentPPO(seed=7, hidden_size=8)
        explicit = RecurrentPPO(seed=7, hidden_size=8, freeze_base=False)
        for a, b in [(default.model.parameters(), explicit.model.parameters()),
                     (default.optimizer.state, explicit.optimizer.state)]:
            aa, bb = arrays(a), arrays(b)
            self.assertEqual(aa.keys(), bb.keys())
            for name in aa:
                np.testing.assert_array_equal(aa[name], bb[name])
        with self.assertRaises(ValueError):
            RecurrentPPO(freeze_base=1)

    def test_only_memory_parameters_update_and_complete_policy_reloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = PPO(seed=4, action_count=20)
            parent.save(root/'parent', dict(config=base_config(), steps=100))
            agent = RecurrentPPO(seed=7, hidden_size=8, freeze_base=True)
            provenance = initialize_policy(agent.model, root/'parent')
            self.assertTrue(provenance['frozen_parameters'])
            agent.compile()
            before = arrays(agent.model.parameters())
            trainable = arrays(agent.model.trainable_parameters())
            self.assertTrue(trainable)
            self.assertTrue(all(name.startswith('memory') for name in trainable))
            self.assertTrue(all('base' not in name.split('.') for name in arrays(agent.optimizer.state)))
            rng = np.random.default_rng(9)
            obs = mx.array(rng.integers(128, 192, (2, 4, 4, 16, 64), dtype=np.uint8))
            hidden, starts = mx.zeros((2, 8)), mx.zeros((2, 4), dtype=mx.bool_)
            logits, values, _ = agent.model.sequence(obs, hidden, starts)
            logp = logits-mx.logsumexp(logits, axis=-1, keepdims=True)
            data = (obs, mx.zeros((2, 4), dtype=mx.int32), logp[:, :, 0],
                    mx.ones((2, 4)), values+1, hidden, starts)
            mx.eval(data)
            for _ in range(3):
                loss, aux = agent.update(*data)
                mx.eval(loss, aux, agent.state)
                self.assertTrue(np.isfinite(loss.item()))
            after = arrays(agent.model.parameters())
            for name in before:
                if name.startswith('base.'):
                    np.testing.assert_array_equal(before[name], after[name])
            for name in ['memory.Wx', 'memory_actor.weight', 'memory_value.weight']:
                self.assertFalse(np.array_equal(before[name], after[name]), name)
            self.assertTrue(all('base' not in name.split('.') for name in arrays(agent.optimizer.state)))
            cfg = dict(base_config(), architecture=RECURRENT_ARCHITECTURE,
                       recurrent_hidden=8, memory_scale=1., freeze_recurrent_base=True)
            agent.save(root/'frozen', dict(config=cfg, steps=3))
            saved = mx.load(str(root/'frozen/model.safetensors'))
            self.assertEqual(set(saved), set(after))
            loaded, _ = load_policy(root/'frozen/model.safetensors')
            expected = agent.policy(seed=5)
            loaded.reset_seed(5)
            screens = np.array(obs[:, 0])
            for _ in range(4):
                np.testing.assert_array_equal(loaded(screens), expected(screens))
            trace = record_game(loaded, 10000, tstates=100000, max_steps=0)
            verification = verify_policy_trace(root/'frozen/model.safetensors', *trace[:4])
            self.assertTrue(verification['verified'])
            self.assertEqual(verification['verified_actions'], len(trace[1]))

    def test_cli_frozen_initialization_resume_and_mode_guards(self):
        from rl import defense_train
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for extra in [['--freeze-recurrent-base'],
                          ['--freeze-recurrent-base', '--recurrent-hidden=8'],
                          ['--freeze-recurrent-base', '--recurrent-hidden=8', '--memory-scale=0',
                           '--initialize-policy=missing']]:
                with patch.object(sys, 'argv', ['train', '--run', str(root/'invalid'), *extra]), \
                        contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    defense_train.main()
                self.assertEqual(error.exception.code, 2)
            parent = PPO(seed=4, action_count=20)
            parent.save(root/'parent', dict(config=base_config(), steps=100))
            command = [sys.executable, '-m', 'rl.defense_train', '--envs', '2', '--rollout', '8',
                       '--batch-size', '16', '--epochs', '1', '--recurrent-hidden', '8',
                       '--sequence-length', '4', '--mlx-cache-mb', '128', '--max-episode-steps', '3']
            def run(name, extra, succeeds=True):
                result = subprocess.run(command+['--run', str(root/name), '--artifacts', str(root/(name+'-artifacts'))]+extra,
                                        capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0 if succeeds else 2, result.stdout+result.stderr)
                if succeeds:
                    return json.loads((root/name/'latest/state.json').read_text())
            first = run('first', ['--initialize-policy', str(root/'parent'), '--freeze-recurrent-base', '--steps', '32'])
            later = run('later', ['--resume', str(root/'first/latest'), '--steps', '64'])
            self.assertTrue(later['config']['freeze_recurrent_base'])
            self.assertEqual(later['config']['initialization'], first['config']['initialization'])
            self.assertEqual(later['steps'], 64)
            expected = mx.load(str(root/'parent/model.safetensors'))
            for name in ['first', 'later']:
                weights = mx.load(str(root/name/'latest/model.safetensors'))
                for key in expected:
                    np.testing.assert_array_equal(np.array(expected[key]), np.array(weights['base.'+key]))
                optimizer = mx.load(str(root/name/'latest/optimizer.npz'))
                self.assertTrue(all('base' not in key.split('.') for key in optimizer))
            run('invalid-unfreeze', ['--resume', str(root/'first/latest'), '--no-freeze-recurrent-base'], succeeds=False)
            # A legacy, unfrozen optimizer cannot silently turn into a frozen one.
            state = json.loads((root/'first/latest/state.json').read_text())
            state['config']['freeze_recurrent_base'] = False
            (root/'first/latest/state.json').write_text(json.dumps(state))
            run('invalid-freeze', ['--resume', str(root/'first/latest'), '--freeze-recurrent-base'], succeeds=False)


if __name__ == '__main__':
    unittest.main()
