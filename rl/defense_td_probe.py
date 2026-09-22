"""Frozen own-replay TD scale audit; never training or a search for actions.

Reconstruct actual one/multi-option returns and compare them with the matching
saved target network. Unit-weight loss summaries are NOT measurements of the
prioritized training distribution, gradients, or causal explanations of failure.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .defense_alias_probe import replay_observation
from .defense_learning import REPEAT_ALGORITHM, load_policy, sha256
from .defense_loss_probe import analyze
from .defense_option_returns import MultiOptionReturns
from .defense_repeat import OptionReturns


def td_summary(predicted, labels, returns):
    predicted, labels, returns = map(np.asarray, (predicted, labels, returns))
    if (predicted.ndim != 1 or not len(predicted) or labels.shape != predicted.shape
            or returns.shape != predicted.shape or not all(np.isfinite(a).all() for a in (predicted, labels, returns))):
        raise ValueError('matching nonempty finite value vectors required')
    absolute = np.abs(predicted - labels)
    huber = np.where(absolute < 1., .5*absolute**2, absolute-.5)
    groups = []
    for name, selected in [('zero_observed_return', returns == 0), ('positive_observed_return', returns > 0)]:
        n = int(selected.sum())
        groups.append(dict(group=name, transitions=n,
                           huber_loss_share=float(huber[selected].sum()/huber.sum()) if huber.sum() else 0.,
                           mean_absolute_td=float(absolute[selected].mean()) if n else None,
                           linear_huber_fraction=float((absolute[selected] >= 1).mean()) if n else None))
    return dict(transitions=len(predicted), mean_absolute_td=float(absolute.mean()),
                p90_absolute_td=float(np.quantile(absolute, .9)), maximum_absolute_td=float(absolute.max()),
                linear_huber_fraction=float((absolute >= 1).mean()), mean_unit_weight_huber=float(huber.mean()),
                mean_predicted_value=float(predicted.mean()), mean_label=float(labels.mean()),
                mean_observed_return=float(returns.mean()), maximum_observed_return=float(returns.max()),
                groups=groups)


class FrozenRows:
    def __init__(self):
        self.rows = []

    def add(self, obs, action, reward, following, discount):
        self.rows.append((obs.copy(), action, reward, following.copy(), discount))


def probe(bundle, checkpoint):
    bundle, checkpoint = Path(bundle).resolve(strict=True), Path(checkpoint).resolve(strict=True)
    report, frames, actions = analyze(bundle)
    state = json.loads((bundle/'state.json').read_text())
    saved = json.loads((checkpoint/'state.json').read_text())
    config = state['config']
    if (config.get('algorithm') != REPEAT_ALGORITHM or config.get('allow_enter', False)
            or not config.get('life_terminal') or config.get('n_step') != 1
            or config != saved['config'] or state['steps'] != saved['steps']):
        raise ValueError('matching standard learned-duration replay and full checkpoint required')
    checkpoint_hashes = {name: sha256(checkpoint/name) for name in ('model.safetensors', 'target.safetensors', 'state.json')}
    if checkpoint_hashes['model.safetensors'] != report['source_hashes']['model.safetensors']:
        raise ValueError('online checkpoint is not the recorded replay policy')
    if config['repeat_source_sha256'] != sha256(Path(__file__).with_name('defense_repeat.py')):
        raise ValueError('recorded executor source differs')
    horizon = config.get('repeat_n_step', 1)
    if horizon > 1 and config['option_returns_source_sha256'] != sha256(Path(__file__).with_name('defense_option_returns.py')):
        raise ValueError('recorded return source differs')
    policy, loaded = load_policy(bundle/'model.safetensors')
    if loaded != config:
        raise ValueError('loaded policy configuration differs')
    policy.reset_seed(report['result']['seed']+1_000_000)
    with np.load(bundle/'trace.npz', allow_pickle=False) as trace:
        rewards = trace['rewards']
    sink = FrozenRows()
    returns = (MultiOptionReturns(sink, horizon, config['gamma'], 20, config['learned_repeats']) if horizon > 1
               else OptionReturns(sink, config['gamma'], 20, config['learned_repeats']))
    ends = {life['visible_loss_frame'] for life in report['lives']}
    for i, expected in enumerate(actions):
        obs = replay_observation(frames, i, config.get('observation_stride', 1))
        idle = policy.serial_keys is None or policy.memories.get(policy.serial_keys[0], (0, 0))[1] == 0
        if int(policy(obs[None])[0]) != int(expected):
            raise ValueError(f'frozen command differs at {i}')
        if idle:
            returns.begin(obs, policy.memories[policy.serial_keys[0]][0])
        following = replay_observation(frames, i+1, config.get('observation_stride', 1))
        terminal = i+1 in ends
        returns.append(float(rewards[i])*config['reward_scale'], following, terminal, False)
        policy.observe_boundaries(np.asarray([terminal], dtype=bool))
    if len(sink.rows) != returns.completed:
        raise ValueError('complete trace did not emit every option start')

    import mlx.core as mx
    from .model import QNetwork
    target = QNetwork(20*len(config['learned_repeats']))
    target.load_weights(str(checkpoint/'target.safetensors'))
    mx.eval(target.state)
    target_predict = mx.compile(target, inputs=target.state)
    predictions, labels, observed, discounts = [], [], [], []
    for start in range(0, len(sink.rows), 64):
        rows = sink.rows[start:start+64]
        obs = np.stack([r[0] for r in rows]); following = np.stack([r[3] for r in rows])
        selected = np.asarray([r[1] for r in rows])
        # float32 matches the actual replay batch consumed by the MLX learner.
        reward = np.asarray([r[2] for r in rows], np.float32)
        discount = np.asarray([r[4] for r in rows], np.float32)
        current_q, next_q = policy.infer(obs), policy.infer(following)
        target_q = np.asarray(target_predict(mx.array(following)))
        index = np.arange(len(rows))
        predictions.extend(current_q[index, selected])
        labels.extend(reward + discount*target_q[index, next_q.argmax(axis=1)])
        observed.extend(reward); discounts.extend(discount)
    for name, digest in report['source_hashes'].items():
        if sha256(bundle/name) != digest:
            raise RuntimeError('replay changed during analysis')
    for name, digest in checkpoint_hashes.items():
        if sha256(checkpoint/name) != digest:
            raise RuntimeError('checkpoint changed during analysis')
    return dict(bundle=str(bundle), checkpoint=str(checkpoint), source_hashes=report['source_hashes'],
                checkpoint_hashes=checkpoint_hashes, probe_source_sha256=sha256(Path(__file__)),
                result=report['result'], repeat_n_step=horizon, durations=config['learned_repeats'],
                reward_scale=config['reward_scale'], gamma=config['gamma'],
                commands_reproduced=len(actions), completed_options=returns.completed,
                interrupted_options=returns.interrupted, terminal_returns=int(np.sum(np.asarray(discounts) == 0)),
                td=td_summary(predictions, labels, observed),
                raw_reward_counts={str(float(v)): int(n) for v,n in zip(*np.unique(rewards, return_counts=True))},
                original_native_verification=report['original_native_verification'], native_reexecution=False,
                diagnostic_only=True, training_data_written=False, parameter_updates=0, ranking_eligible=False,
                limitations=[
                    'One selected frozen boot replay, not prioritized or exploratory training samples.',
                    'Unit-weight Huber summaries are not actual PER loss weighting or gradient contributions.',
                    'Frozen checkpoint Double-Q labels, not Monte Carlo truth or optimal counterfactual returns.',
                    'Observed return groups classify reward sums, not obstacle positions or recoverability.',
                    'Magnitude or saturation does not establish a causal learning defect.',
                    'Original native verification reused; only recorded visible screens and own choices reconstructed.',
                    'No rewards, policy choices, parameters or training data are changed.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite evidence')
    result = probe(args.bundle, args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['td'], indent=2))


if __name__ == '__main__':
    main()
