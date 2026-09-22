import importlib.util
import json
from pathlib import Path
import numpy as np
from mlx.utils import tree_flatten
from rl.defense_learning import sha256
from rl.defense_world_model import WorldLearner
from rl.defense_world_data import Sequences

source = Path('/private/tmp/defense_world_model_before_overshoot.py')
spec = importlib.util.spec_from_file_location('rl._world_legacy', source)
legacy = importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy)
checkpoint = Path('runs/defense-world-fit-07-actor-feedback/update-016000')
old, new = legacy.WorldLearner(), WorldLearner()
rng, other_rng = np.random.default_rng(0), np.random.default_rng(1)
old.restore(checkpoint,rng); new.restore(checkpoint,other_rng)
data = Sequences('runs/defense-world-data-feedback-02','train',32)
for update in range(4):
    first, second = data.sample(8,rng), data.sample(8,other_rng)
    for a,b in zip(first,second): np.testing.assert_array_equal(a,b)
    assert old.train(first) == new.train(second)
    for left,right in ((old.model.parameters(),new.model.parameters()),
                       (old.optimizer.state,new.optimizer.state),(old.random,new.random)):
        a,b = dict(tree_flatten(left)), dict(tree_flatten(right))
        assert a.keys() == b.keys()
        for k in a: np.testing.assert_array_equal(np.array(a[k]), np.array(b[k]))
root = Path('results/defense/training/world-model-overshoot-01')
root.mkdir(exist_ok=True)
report = dict(default_parity=True, updates=4, actual_own_training_windows=32,
    old_source_sha256=sha256(source), new_source_sha256=sha256(Path('rl/defense_world_model.py')),
    checkpoint_sha256=sha256(checkpoint/'world.safetensors'), model_optimizer_rng_exact=True,
    no_diagnostic_state_used_for_training=True)
(root/'default-parity.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
