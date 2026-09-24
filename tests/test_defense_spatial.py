"""Screen-only spatial duration policy and full-own-checkpoint transfer."""

import json
from pathlib import Path
import tempfile
import unittest

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from rl.defense import DefenseEnv, ENVIRONMENT_VERSION, GAME_SHA256, action_names
from rl.defense_duration_ppo import DurationPPO, duration_action_names
from rl.defense_learning import load_policy, policy_description
from rl.defense_spatial import (SpatialDurationNetwork, SpatialDurationPPO,
                                initialize_duration_checkpoint)
from rl.defense_spatial_spec import SPATIAL_ARCHITECTURE
from rl.model import QNetwork


DURATIONS = (1, 4, 16, 64)


def own_duration_checkpoint(folder):
    folder = Path(folder)
    QNetwork(80).save_weights(str(folder/"model.safetensors"))
    config = dict(game="defense", game_sha256=GAME_SHA256,
                  environment_version=ENVIRONMENT_VERSION, algorithm="ppo",
                  action_names=list(action_names(False)),
                  policy_action_names=list(duration_action_names(DURATIONS)),
                  learned_durations=list(DURATIONS), life_terminal=True,
                  tstates=100_000, observation_stride=1)
    (folder/"state.json").write_text(json.dumps(dict(config=config, steps=123)))
    return config


class DefenseSpatialTests(unittest.TestCase):
    def test_longest_duration_extension_copies_all_old_rows_without_key_bias(self):
        with tempfile.TemporaryDirectory() as tmp:
            own_duration_checkpoint(tmp)
            target = QNetwork(100)
            transfer = initialize_duration_checkpoint(
                target, tmp, (*DURATIONS, 128), tstates=100_000,
                observation_stride=1, spatial=False, extend_longest=True)
            source = mx.load(str(Path(tmp)/"model.safetensors"))
            copied = dict(tree_flatten(target.parameters()))
            for name, value in source.items():
                got = np.array(copied[name])
                old = np.array(value)
                np.testing.assert_array_equal(got[:len(old)] if name.startswith("advantage.") else got,
                                              old)
            np.testing.assert_array_equal(
                np.array(copied["advantage.weight"])[80:100],
                np.array(source["advantage.weight"])[60:80])
            np.testing.assert_array_equal(
                np.array(copied["advantage.bias"])[80:100],
                np.array(source["advantage.bias"])[60:80]-2.)
            self.assertEqual(transfer["source_durations"], list(DURATIONS))
            self.assertEqual(transfer["target_durations"], [1, 4, 16, 64, 128])
            self.assertEqual(transfer["appended_logit_offset"], -2.)
            self.assertFalse(transfer["trajectories_loaded"])
            initialize_duration_checkpoint(
                target, tmp, (*DURATIONS, 128), tstates=100_000,
                observation_stride=1, spatial=False, extend_longest=True,
                appended_logit_offset=0.)
            np.testing.assert_array_equal(
                np.array(target.advantage.bias)[80:100],
                np.array(source["advantage.bias"])[60:80])
            with self.assertRaises(ValueError):
                initialize_duration_checkpoint(QNetwork(100), tmp, (1, 4, 16, 64, 128),
                                               tstates=100_000, observation_stride=1,
                                               spatial=False)
            with self.assertRaises(ValueError):
                initialize_duration_checkpoint(QNetwork(100), tmp, (1, 4, 16, 64, 128),
                                               tstates=100_000, observation_stride=1,
                                               spatial=True, extend_longest=True)
            with self.assertRaises(ValueError):
                initialize_duration_checkpoint(QNetwork(100), tmp, (1, 4, 16, 64, 128),
                                               tstates=100_000, observation_stride=1,
                                               spatial=False, extend_longest=True,
                                               appended_logit_offset=True)

    def test_zero_gate_preserves_full_own_duration_policy_on_real_screens(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = own_duration_checkpoint(tmp)
            ordinary, spatial = QNetwork(80), SpatialDurationNetwork(80)
            control = initialize_duration_checkpoint(
                ordinary, tmp, DURATIONS, tstates=100_000, observation_stride=1,
                spatial=False)
            transfer = initialize_duration_checkpoint(
                spatial, tmp, DURATIONS, tstates=100_000, observation_stride=1,
                spatial=True)
            self.assertEqual(control["source_model_sha256"], transfer["source_model_sha256"])
            self.assertEqual(float(spatial.spatial_scale.item()), 0.)
            self.assertIn("spatial_scale", dict(tree_flatten(spatial.trainable_parameters())))
            env = DefenseEnv(seed=310)
            try:
                screens = [env.reset()]
                for action in (0, 4, 9, 1):
                    screens.append(env.step(action)[0])
            finally:
                env.close()
            obs = mx.array(np.stack(screens))
            before = ordinary.policy_value(obs)
            after = spatial.policy_value(obs)
            for left, right in zip(before, after, strict=True):
                np.testing.assert_array_equal(np.array(left), np.array(right))
            spatial.save_weights(str(Path(tmp)/"spatial.safetensors"))
            config["architecture"] = SPATIAL_ARCHITECTURE
            (Path(tmp)/"state.json").write_text(json.dumps(dict(config=config, steps=123)))
            policy, loaded = load_policy(Path(tmp)/"spatial.safetensors")
            self.assertIn("spatial-residual", policy_description(loaded))
            self.assertEqual(np.asarray(policy(screens[:2])).shape, (2,))

    def test_gate_receives_gradient_and_optimizer_moves_it(self):
        agent = SpatialDurationPPO(seed=332, action_count=80, entropy=0)
        rng = np.random.default_rng(332)
        obs = mx.array(rng.integers(256, size=(4, 4, 16, 64), dtype=np.uint8))
        logits, values = agent.predict(obs)
        old_logp = logits-mx.logsumexp(logits, axis=-1, keepdims=True)
        actions = mx.array([0, 1, 2, 3], mx.int32)
        old_logp = mx.take_along_axis(old_logp, actions[:, None], axis=-1)[:, 0]
        args = (obs, actions, old_logp, mx.array([1., -.5, .5, -1.]),
                values+mx.array([1., 2., -2., -1.]), mx.ones(4))
        loss, _ = agent.update(*args)
        mx.eval(loss, agent.state)
        self.assertTrue(np.isfinite(float(loss.item())))
        self.assertNotEqual(float(agent.model.spatial_scale.item()), 0.)

    def test_invalid_spatial_profile_and_incompatible_transfer_rejected(self):
        with self.assertRaises(ValueError):
            policy_description(dict(algorithm="ppo", architecture=SPATIAL_ARCHITECTURE))
        with tempfile.TemporaryDirectory() as tmp:
            config = own_duration_checkpoint(tmp)
            model = SpatialDurationNetwork(80)
            with self.assertRaises(ValueError):
                initialize_duration_checkpoint(model, tmp, DURATIONS, tstates=50_000,
                                               observation_stride=1, spatial=True)
            with self.assertRaises(ValueError):
                initialize_duration_checkpoint(QNetwork(80), tmp, DURATIONS, tstates=100_000,
                                               observation_stride=1, spatial=True)
            config["architecture"] = SPATIAL_ARCHITECTURE
            (Path(tmp)/"state.json").write_text(json.dumps(dict(config=config, steps=123)))
            with self.assertRaises(ValueError):
                initialize_duration_checkpoint(model, tmp, DURATIONS, tstates=100_000,
                                               observation_stride=1, spatial=True)


if __name__ == "__main__":
    unittest.main()
