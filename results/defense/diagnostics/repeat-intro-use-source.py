"""Read-only recognized-intro-text breakdown; never used by a policy or trainer.

Run from the repository root. The complement of recognized introduction text
is deliberately NOT labelled gameplay: it can include other waits/animations.
"""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()))
import numpy as np
from rl.defense import screen_info
from rl.defense_alias_probe import replay_observation
from rl.defense_learning import REPEAT_ALGORITHM, load_policy, sha256
from rl.defense_loss_probe import analyze


output=Path(sys.argv[1])
assert not output.exists()
reports=[]
for name in sys.argv[2:]:
    bundle=Path(name)
    reviewed,frames,actions=analyze(bundle)
    policy,config=load_policy(bundle/'model.safetensors')
    assert config['algorithm']==REPEAT_ALGORITHM
    assert config['repeat_source_sha256']==sha256(Path('rl/defense_repeat.py'))
    policy.reset_seed(reviewed['result']['seed']+1_000_000)
    durations=config['learned_repeats']
    counts=np.zeros((2,len(durations)),np.int64)
    starts=np.zeros_like(counts)
    ends={life['visible_loss_frame'] for life in reviewed['lives']}
    for i,action in enumerate(actions):
        deciding=policy.serial_keys is None or policy.memories.get(policy.serial_keys[0],(0,0))[1]==0
        obs=replay_observation(frames,i,config.get('observation_stride',1))[None]
        assert int(policy(obs)[0])==int(action),(name,i)
        option,_=policy.memories[policy.serial_keys[0]]
        group=int(screen_info(frames[i])['stage'] is not None)
        counts[group,option//20]+=1
        starts[group,option//20]+=int(deciding)
        policy.observe_boundaries(np.asarray([i+1 in ends],dtype=bool))
    assert counts.sum()==len(actions)
    assert all(sha256(bundle/name)==digest for name,digest in reviewed['source_hashes'].items())
    reports.append(dict(bundle=str(bundle),source_hashes=reviewed['source_hashes'],
        reproduced_commands=len(actions),durations=durations,
        intro_text_present=dict(base_commands_by_duration=counts[1].tolist(),decisions_by_duration=starts[1].tolist()),
        intro_text_not_recognized=dict(base_commands_by_duration=counts[0].tolist(),decisions_by_duration=starts[0].tolist())))
result=dict(analysis_source_sha256=sha256(Path(__file__)),reports=reports,
    classifier_source_sha256=sha256(Path('rl/defense.py')),
    classifier='Existing visible stage-introduction phrases in the PRE-action frame only.',
    diagnostic_only=True,native_reexecution=False,parameter_updates=0,training_data_written=False,
    limitations=['Recorded complete games, not new native evaluations or representative samples.',
                 'No recognized introduction text does NOT establish active gameplay or absence of animation.',
                 'Intro text is not a collision, movement, obstacle or recoverability label.',
                 'No routes, preferred keys, rewards or training examples are generated.'])
with output.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
