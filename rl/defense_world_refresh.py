"""Explicit warm continuation of behavior after its own world-model update."""

from pathlib import Path

from .defense_learning import sha256


def refresh_world(learner, previous, current, world_checkpoint, world_metadata):
    """Retain actor/critic/Adam/RNG; replace only a proven descendant world.

    Caller has restored the complete actor state and verified the new world's
    saved file hashes. This is not an optimizer reset or a fresh actor.
    """
    for key in ('tstates', 'observation_stride'):
        if previous[key] != current[key]:
            raise ValueError('incompatible refreshed behavior: '+key)
    parent = world_metadata.get('args', {}).get('resume')
    if not parent or sha256(Path(parent)/'world.safetensors') != previous['world_sha256']:
        raise ValueError('new world must descend from this actor\'s previous frozen world')
    old_data = previous['dataset_sha256']
    if current['dataset_sha256'] != old_data and world_metadata.get('previous_dataset_sha256') != old_data:
        raise ValueError('new world does not declare this actor\'s prior dataset')
    path = Path(world_checkpoint)/'world.safetensors'
    if sha256(path) != current['world_sha256']:
        raise ValueError('refreshed world checksum mismatch')
    learner.model.world.load_weights(str(path))
    learner.rebind()
    return dict(world_refreshed=True, previous_world_sha256=previous['world_sha256'],
                previous_dataset_sha256=old_data, retained_actor_updates=learner.updates)
