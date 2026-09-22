"""Isolated input-responsiveness diagnostics, never training or policy search.

Reproduce an already verified own-policy trace exactly, then use opaque copies
at fixed pre-visible-loss offsets to test all ordinary keys for a short horizon.
Report only screen diversity, never preferred actions, routes or improved games.
No branch is fed into a learner, a reset archive, evaluation or replay ranking.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .defense import DefenseEnv
from .defense_learning import sha256
from .defense_loss_probe import analyze
from .defense_snapshot import capture, restore


OFFSETS = (128, 64, 32, 8)
HORIZON = 4


def screen_diversity(branches):
    """Compare only the common prefix, never crossing a visible life loss."""
    if (len(branches) != 20 or any(not b for b in branches)
            or any(f.shape != (16, 64) or f.dtype != np.uint8 for b in branches for f in b)):
        raise ValueError('requires twenty nonempty raw-screen branches')
    common = min(map(len, branches))
    graphics = lambda f: np.where((f[1:] >= 128) & (f[1:] <= 191), f[1:], 128).astype(np.uint8).tobytes()
    return dict(common_observed_decisions=common,
        distinct_screen_sequences=len({b''.join(f.tobytes() for f in b[:common]) for b in branches}),
        distinct_gameplay_graphics_sequences=len({b''.join(graphics(f) for f in b[:common]) for b in branches}),
        any_gameplay_graphics_response=len({b''.join(graphics(f) for f in b[:common]) for b in branches}) > 1)


def probe(bundle):
    bundle = Path(bundle).resolve(strict=True)
    reviewed, frames, actions = analyze(bundle)
    config = json.loads((bundle/'state.json').read_text())['config']
    with np.load(bundle/'trace.npz', allow_pickle=False) as trace:
        rewards = trace['rewards']
        metadata = json.loads(str(trace['metadata']))
    if config.get('allow_enter', False) or config.get('eval_max_steps', 0):
        raise ValueError('requires the ordinary 20-action uncapped complete-game profile')
    native = Path(__file__).resolve().parents[1]/'libtrs.so'
    if sha256(native) != config['native_sha256']:
        raise ValueError('native build differs from the original replay')
    anchors = []
    for life in reviewed['lives']:
        for distance in OFFSETS:
            index = life['visible_loss_frame']-distance
            if index >= life['previous_visible_loss_frame']:
                anchors.append(dict(life=life['life'],decisions_before_visible_loss=distance,decision=index))
    wanted = {a['decision'] for a in anchors}
    snapshots = {}
    env = DefenseEnv(tstates=config['tstates'], observation_stride=config.get('observation_stride',1))
    try:
        observation = env.reset(metadata['result']['seed'])
        np.testing.assert_array_equal(observation[-1],frames[0])
        for index, action in enumerate(actions):
            if index in wanted:snapshots[index]=capture(env)
            following,reward,terminal,truncated,info = env.step(int(action))
            np.testing.assert_array_equal(following[-1],frames[index+1])
            if reward != rewards[index] or truncated or terminal != (index==len(actions)-1):
                raise ValueError('recorded trajectory failed exact native reproduction')
        if set(snapshots)!=wanted or not info['game_over']:
            raise ValueError('missing own replay anchors or native game ending')
        results=[]
        for anchor in anchors:
            saved=snapshots[anchor['decision']]
            before=saved.frames.copy()
            branches=[];ending=[]
            for action in range(20):
                initial=restore(env,saved)
                np.testing.assert_array_equal(initial, saved.frames[::saved.observation_stride])
                branch=[]
                for _ in range(HORIZON):
                    obs,_,terminal,truncated,info=env.step(action)
                    branch.append(obs[-1].copy())
                    if terminal or truncated or info['life_lost']:break
                branches.append(branch);ending.append(bool(terminal or truncated or info['life_lost']))
            np.testing.assert_array_equal(saved.frames,before)
            results.append(dict(**anchor,**screen_diversity(branches),branches_ending_at_visible_boundary=sum(ending)))
    finally:env.close()
    for name,digest in reviewed['source_hashes'].items():
        if sha256(bundle/name)!=digest:raise RuntimeError('source changed during diagnostic')
    return dict(bundle=str(bundle),source_hashes=reviewed['source_hashes'],
        probe_source_sha256=sha256(Path(__file__)),native_sha256=config['native_sha256'],
        original_native_trajectory_reproduced=len(actions),
        original_neural_verification_reused=True,diagnostic_only=True,
        branch_horizon=HORIZON,branches_per_anchor=20,anchors=results,
        parameter_updates=0,training_data_written=False,ranking_eligible=False,
        snapshots_decoded=False,chosen_action_or_route_output=False,
        limitations=[
            'Selected verified replay anchors, not all live training-reset states.',
            'Nonidentical graphics prove short-horizon input responsiveness, not movement, survival, recoverability or obstacle passage.',
            'Identical graphics do not prove death: action effects can be delayed or visually hidden.',
            'Offsets refer to visible ship loss, never an exact physical collision.',
            'Branch diagnostics are not complete-game evaluation and never enter training, reset archives or replay ranking.',
            'Only opaque own-state copying and visible observations are used; no native payload is decoded.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('refusing to overwrite existing evidence')
    result=probe(args.bundle)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(reproduced=result['original_native_trajectory_reproduced'],anchors=result['anchors'])),flush=True)


if __name__=='__main__':main()
