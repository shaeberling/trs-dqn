"""Train a categorical readout of a frozen own world: diagnostic, never acting."""

import argparse
import json
from pathlib import Path
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np

from .defense_learning import sha256, write_json
from .defense_world_byte_audit import audit_bytes
from .defense_world_bytes import ByteWorldLearner
from .defense_world_data import Sequences, disk_guard


class FrozenByteProbe:
    def __init__(self, learner):
        self.learner, self.updates = learner, 0
        self.state = [learner.model.state, learner.decoder.state, learner.byte_optimizer.state, learner.random]
        mx.eval(self.state)
        self.update = mx.compile(self._update, inputs=self.state, outputs=self.state)

    def _update(self, frames, actions):
        learner = self.learner
        key, following = mx.random.split(learner.random['key'])
        states,_ = learner.model.observe(frames,actions,key)
        features = mx.stop_gradient(learner.model.features(states)[:,learner.burn+1:])
        targets = frames[:,learner.burn+1:].astype(mx.int32)
        def objective(decoder):
            return nn.losses.cross_entropy(decoder(features),targets,reduction='mean')
        loss,grads = nn.value_and_grad(learner.decoder,objective)(learner.decoder)
        grads,norm = optim.clip_grad_norm(grads,max_norm=100.)
        learner.byte_optimizer.update(learner.decoder,grads)
        learner.random['key'] = following
        return loss,norm

    def train(self,batch):
        result = self.update(mx.array(batch[0]),mx.array(batch[1]))
        mx.eval(result,self.state)
        values = [float(v) for v in result]
        if not np.isfinite(values).all(): raise FloatingPointError('nonfinite readout update')
        self.updates += 1
        return dict(byte_cross_entropy=values[0],gradient_norm=values[1])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('world',type=Path)
    parser.add_argument('data',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--updates',type=int,default=2000)
    parser.add_argument('--every',type=int,default=1000)
    args=parser.parse_args()
    if args.output.exists() or min(args.updates,args.every)<1:
        parser.error('new output and positive counts required')
    saved=json.loads((args.world/'state.json').read_text());metadata=saved['metadata']
    if (metadata['dataset_sha256']!=sha256(args.data/'manifest.json')
            or metadata['length']!=32 or metadata['burn']!=8 or metadata['batch']!=8
            or saved.get('byte_reconstruction') or saved.get('overshooting',{}).get('weight',0.)):
        parser.error('requires the matching plain 32-action/8-burn/8-batch world parent')
    mx.set_cache_limit(128*1024*1024)
    learner=ByteWorldLearner(seed=0,burn=8)
    rng=np.random.default_rng(0);learner.restore(args.world,rng,allow_objective_change=True)
    probe=FrozenByteProbe(learner)
    train,held=Sequences(args.data,'train',32),Sequences(args.data,'heldout',32)
    frozen=held.sample(64,np.random.default_rng(82731))
    disk_guard(args.output.parent);args.output.mkdir(parents=True,exist_ok=False)
    config=dict(world=str(args.world.resolve()),world_sha256=sha256(args.world/'world.safetensors'),
        dataset_sha256=metadata['dataset_sha256'],batch=8,burn=8,length=32,
        diagnostic_only=True,policy_training=False,promotion_eligible=False,
        source_sha256=sha256(Path(__file__)),byte_source_sha256=sha256(Path(__file__).with_name('defense_world_bytes.py')),
        audit_source_sha256=sha256(Path(__file__).with_name('defense_world_byte_audit.py')),
        description='Same newly initialized byte decoder, own-data sampler RNG and stochastic-filter RNG as joint arm; only decoder Adam updates. World optimizer/weights frozen.')
    write_json(args.output/'config.json',config)
    started=time.monotonic()
    with (args.output/'metrics.jsonl').open('x') as log:
        def record(event):
            event=dict(event,elapsed_seconds=time.monotonic()-started)
            log.write(json.dumps(event)+'\n');log.flush();print(json.dumps(event),flush=True)
        def preserve():
            disk_guard(args.output)
            output=args.output/f'probe-{probe.updates:06d}';output.mkdir(exist_ok=False)
            learner.decoder.save_weights(str(output/'byte-decoder.safetensors'))
            mx.savez(str(output/'byte-optimizer.npz'),**dict(tree_flatten(learner.byte_optimizer.state)))
            mx.savez(str(output/'random.npz'),**learner.random)
            write_json(output/'state.json',dict(probe_updates=probe.updates,config=config,
                sampling_rng=rng.bit_generator.state,hashes={p.name:sha256(p) for p in output.iterdir() if p.is_file()}))
            report=audit_bytes(learner.model,learner.decoder,frozen,8)
            write_json(output/'byte-audit.json',report)
            record(dict(event='audit',probe_updates=probe.updates,observed=report['measurements']['observed']['groups']))
        preserve()
        while probe.updates<args.updates:
            stats=probe.train(train.sample(8,rng))
            if probe.updates%20==0:
                disk_guard(args.output);record(dict(event='update',probe_updates=probe.updates,**stats))
            if probe.updates%args.every==0 or probe.updates==args.updates:preserve()
        learner.model.save_weights(str(args.output/'frozen-world-check.safetensors'))
        mx.savez(str(args.output/'frozen-world-optimizer.npz'),**dict(tree_flatten(learner.optimizer.state)))
        if sha256(args.output/'frozen-world-check.safetensors')!=config['world_sha256']:
            raise RuntimeError('frozen world changed')
        with np.load(args.world/'optimizer.npz') as a,np.load(args.output/'frozen-world-optimizer.npz') as b:
            if set(a.files)!=set(b.files):raise RuntimeError('frozen optimizer keys changed')
            for key in a.files:np.testing.assert_array_equal(a[key],b[key])
        record(dict(event='finished',probe_updates=probe.updates,frozen_world_and_optimizer_verified=True))


if __name__=='__main__':main()
