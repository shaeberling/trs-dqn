import json
from pathlib import Path
import mlx.core as mx
import numpy as np
from rl.defense_learning import sha256
from rl.defense_world_data import Sequences
from rl.defense_world_model import WorldModel

checkpoint = Path('runs/defense-world-fit-08-overshoot-control/update-018000')
data = Path('runs/defense-world-data-feedback-02')
saved = json.loads((checkpoint/'state.json').read_text())
for name,digest in saved['hashes'].items(): assert sha256(checkpoint/name)==digest
assert sha256(data/'manifest.json')==saved['metadata']['dataset_sha256']
held = Sequences(data,'heldout',32)
frames,actions,_,_ = held.sample(64,np.random.default_rng(82731))
model=WorldModel(); model.load_weights(str(checkpoint/'world.safetensors'))
mx.set_cache_limit(128*1024*1024)
groups={name:dict(cells=0,correct=0,within_2=0,within_5=0,squared_error=0.,persistence_correct=0) for name in ('nonblank_ascii','changed_ascii','digits','stars','changed_hud_cells')}
star_space=dict(stars=0,stars_above_midpoint=0,spaces=0,spaces_below_midpoint=0)
for start in range(0,len(frames),8):
    batch=frames[start:start+8]; commands=actions[start:start+8]
    states,_=model.observe(mx.array(batch),mx.array(commands),mx.random.key(0),sample=False)
    prediction=np.array(model.decode(model.features(states)[:,9:]))+.5
    values=prediction.reshape(len(batch),24,16,3,64,2,2)[...,1].mean(axis=(3,5))
    predicted=np.rint(values*127).clip(0,255).astype(np.uint8)
    target,previous=batch[:,9:],batch[:,8:-1]
    text=(target>32)&(target<127)
    exact=np.array(model.pixels(mx.array(target)))+.5
    exact_codes=np.rint(exact.reshape(len(batch),24,16,3,64,2,2)[...,1].mean(axis=(3,5))*127).astype(np.uint8)
    np.testing.assert_array_equal(exact_codes[text],target[text])
    hud=np.zeros_like(text);hud[:,:,0,:16]=True
    for kind,code,selected in [('stars',42,values*127>37),('spaces',32,values*127<=37)]:
        mask=hud&(target==code)
        star_space[kind]+=int(mask.sum())
        star_space[kind+('_above_midpoint' if kind=='stars' else '_below_midpoint')]+=int((mask&selected).sum())
    masks=dict(nonblank_ascii=text,changed_ascii=text&(target!=previous),
        digits=(target>=48)&(target<=57),stars=target==42,
        changed_hud_cells=hud&(target!=previous)&(target>=32)&(target<127))
    for name,mask in masks.items():
        row=groups[name];row['cells']+=int(mask.sum())
        row['correct']+=int(((predicted==target)&mask).sum())
        error=np.abs(predicted.astype(np.int32)-target.astype(np.int32))
        row['within_2']+=int(((error<=2)&mask).sum())
        row['within_5']+=int(((error<=5)&mask).sum())
        row['persistence_correct']+=int(((previous==target)&mask).sum())
        row['squared_error']+=float(np.sum(((values-target/127.)**2)*mask))
for row in groups.values():
    row['exact_accuracy']=row['correct']/row['cells'] if row['cells'] else None
    row['text_channel_mse']=row['squared_error']/row['cells'] if row['cells'] else None
report=dict(checkpoint_sha256=sha256(checkpoint/'world.safetensors'),dataset_sha256=sha256(data/'manifest.json'),
    source_sha256=sha256(Path(__file__)),groups=groups,star_space_midpoint=star_space,
    encoding_inverse_checked=True, windows=64,post_burn_arrival_frames=1536,
    training_updates=0, description='Observed-posterior reconstruction, NOT future prediction. Nearest-code decoding is diagnostic only; no parser or decoded cells enter a policy.',
    limitations='Reused uniformly sampled held-out windows. Repeated cells are not independent. HUD selection is read-only. Not collision ground truth or a gameplay result.')
root=Path('results/defense/diagnostics/world-text-reconstruction-01')
out=root/'report-with-tolerance.json';assert not out.exists()
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(groups=groups,star_space=star_space)))
