"""Frozen inverse-classifier controls on its own verified stage-one replay.

No parameter updates, emulator actions, rewards or training data are produced.
Repeated/permuted following screens are inference-only diagnostic ablations.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .defense_alias_probe import replay_observation
from .defense_learning import DQN_ALGORITHM, sha256
from .defense_loss_probe import analyze


def classification_metrics(logits, actions):
    values, labels = np.asarray(logits, dtype=np.float64), np.asarray(actions)
    if (values.ndim != 2 or not len(values) or values.shape[1] != 20
            or labels.shape != (len(values),) or labels.dtype.kind not in 'iu'
            or np.any(labels < 0) or np.any(labels >= 20) or not np.isfinite(values).all()):
        raise ValueError('requires finite 20-class logits and matching action labels')
    shifted = values-values.max(axis=1, keepdims=True)
    cross_entropy = np.log(np.exp(shifted).sum(axis=1))-shifted[np.arange(len(values)), labels]
    predicted = values.argmax(axis=1)
    confusion = np.zeros((20,20), np.int64)
    np.add.at(confusion, (labels, predicted), 1)
    return dict(examples=len(values), cross_entropy=float(cross_entropy.mean()),
                accuracy=float(np.mean(predicted == labels)),
                action_counts=np.bincount(labels, minlength=20).tolist(),
                prediction_counts=np.bincount(predicted, minlength=20).tolist(),
                confusion_true_rows_predicted_columns=confusion.tolist())


def summarize(actions, paired, repeated, permuted):
    labels = np.asarray(actions)
    reports = {name: classification_metrics(value, labels)
               for name, value in [('paired', paired), ('repeated_current', repeated), ('permuted_next', permuted)]}
    if len(labels) < 2:
        raise ValueError('requires at least two recorded actions')
    split = len(labels)//2
    # Fixed smoothing gives absent first-half classes a nonzero probability.
    counts = np.bincount(labels[:split], minlength=20)
    probabilities = (counts+1)/(counts.sum()+20)
    constant = np.broadcast_to(np.log(probabilities), (len(labels)-split,20))
    heldout = {name: classification_metrics(value[split:], labels[split:])
               for name,value in [('paired',paired),('repeated_current',repeated),('permuted_next',permuted)]}
    heldout['first_half_frequency_prior'] = classification_metrics(constant, labels[split:])
    return dict(all_examples=reports, second_half=heldout,
                frequency_prior_first_half_examples=split, frequency_prior_pseudocount=1,
                paired_vs_repeated_disagreement=float(np.mean(np.argmax(paired,axis=1)!=np.argmax(repeated,axis=1))),
                paired_vs_permuted_disagreement=float(np.mean(np.argmax(paired,axis=1)!=np.argmax(permuted,axis=1))))


def probe(checkpoint, bundle):
    import mlx.core as mx
    from .defense_inverse import ARCHITECTURE, AUX_FILES, InverseHead
    from .model import QNetwork
    checkpoint = Path(checkpoint).resolve(strict=True)
    report, frames, actions = analyze(bundle)
    names = ('state.json', 'model.safetensors', *AUX_FILES)
    before = {name: sha256(checkpoint/name) for name in names}
    saved = json.loads((checkpoint/'state.json').read_text())
    config, verification = saved['config'], report['original_native_verification']
    if (config.get('algorithm') != DQN_ALGORITHM or config.get('inverse_weight',0) <= 0
            or config.get('inverse_architecture') != ARCHITECTURE or config.get('allow_enter',False)
            or verification.get('temperature',1) != 1 or verification.get('quantile_power',0) != 0
            or before['model.safetensors'] != report['source_hashes']['model.safetensors']
            or before['state.json'] != report['source_hashes']['state.json']
            or saved.get('inverse_auxiliary_hashes') != {n: before[n] for n in AUX_FILES}):
        raise ValueError('requires matching full inverse checkpoint and ordinary verified replay')
    mx.set_cache_limit(128*1024*1024)
    network, auxiliary = QNetwork(20), InverseHead(20)
    network.load_weights(str(checkpoint/'model.safetensors'))
    auxiliary.load_weights(str(checkpoint/AUX_FILES[0]))
    mx.eval(network.state, auxiliary.state)
    stride = config.get('observation_stride',1)
    features = []
    for first in range(0,len(frames),32):
        obs = mx.array(np.stack([replay_observation(frames,i,stride)
                               for i in range(first,min(first+32,len(frames)))]))
        values = np.array(network(obs))
        count = min(32,max(0,len(actions)-first))
        if (not np.isfinite(values).all()
                or not np.array_equal(values[:count].argmax(axis=1),actions[first:first+count])):
            raise ValueError('recorded greedy actions differ from checkpoint')
        features.append(np.array(network.features(obs)))
    features = np.concatenate(features)
    roots, following = features[:-1], features[1:]
    # A half-game cyclic permutation is deterministic, bijective and has no
    # fixed indices. It is not an alternative playable trajectory.
    permuted = following[(np.arange(len(actions))+len(actions)//2) % len(actions)]
    logits = []
    for after in (following,roots,permuted):
        batches=[]
        for first in range(0,len(actions),64):
            batches.append(np.array(auxiliary(mx.array(roots[first:first+64]),mx.array(after[first:first+64]))))
        logits.append(np.concatenate(batches))
    metrics = summarize(actions,*logits)
    if before != {name:sha256(checkpoint/name) for name in names}:
        raise RuntimeError('checkpoint changed during diagnostic')
    final,_,_ = analyze(bundle)
    if final['source_hashes'] != report['source_hashes']:
        raise RuntimeError('replay changed during diagnostic')
    return dict(checkpoint=str(checkpoint),bundle=report['bundle'],checkpoint_hashes=before,
                replay_hashes=report['source_hashes'],probe_sha256=sha256(Path(__file__)),
                diagnostic_only=True,parameter_updates=0,training_data_written=False,
                native_reexecution=False,promotion_eligible=False,replay_actions_reproduced=len(actions),
                result=report['result'],metrics=metrics,limitations=[
                    'One selected greedy replay, not the exploratory training distribution or fresh performance test.',
                    'Frequency prior fits the first half; temporal halves are not independent observations.',
                    'Repeated and permuted next screens are classifier ablations, not playable counterfactuals.',
                    'Aliases, persistent actions and start-screen shortcuts can make action accuracy misleading.',
                    'An ablation gap does not establish useful causal dynamics or navigation benefit.',
                    'Every real pair is one recorded transition; no diagnostic data enters learning.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('bundle',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('refusing to overwrite evidence')
    result=probe(args.checkpoint,args.bundle)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:{n:v for n,v in x.items() if n in ('examples','accuracy','cross_entropy')}
                      for k,x in result['metrics']['second_half'].items()},indent=2))


if __name__ == '__main__':
    main()
