"""Preserve verified models and portable replays outside disposable run folders."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from .evaluate import level_rank, summary
from .outcome import verified_win


FORMAT = {'format': 'breakdown-verified-replay-archive', 'version': 1}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def effort_rank(game):
    return (int(verified_win(game)), game['level'], game['score'], -game['steps'])


def record_rank(record, kind):
    if kind == 'single_game_best_effort':
        return effort_rank(record['replay_game'])
    if kind != 'validation_selected':
        raise ValueError('unknown artifact kind')
    primary, secondary = record['primary'], record['secondary']
    if ([g['seed'] for g in primary['games']] != list(range(10000, 10020))
            or [g['seed'] for g in secondary['games']] != list(range(10100, 10150))):
        raise ValueError('selected artifact requires declared20/50 validation seeds')
    combined = summary(primary['games']+secondary['games'])
    rank = level_rank(combined, 10, game_win=True)
    if combined != record['combined'] or rank is None:
        raise ValueError('selected artifact needs70 complete, correctly summarized games')
    return rank


def publish_record(record, root, kind='validation_selected'):
    # Local import avoids an import cycle with the supervisor's publication hook.
    from .supervise import verify_replay
    root = Path(root)
    checkpoint = Path(record['checkpoint'])
    rank = record_rank(record, kind)
    verify_replay(checkpoint, record['replay'], record['inspection'], record['replay_game'])
    if digest(checkpoint/'model.safetensors') != record['checkpoint_sha256']:
        raise ValueError('artifact checkpoint identity differs')
    if not root.exists():
        root.mkdir(parents=True)
        (root/'archive-format.json').write_text(json.dumps(FORMAT)+'\n')
    if (not (root/'archive-format.json').exists()
            or json.loads((root/'archive-format.json').read_text()) != FORMAT):
        raise ValueError('refusing to overwrite an unmanaged artifact directory')
    current = root/'current.json'
    if current.exists():
        previous = json.loads(current.read_text())
        if previous['kind'] != kind:
            raise ValueError('artifact kinds must have separate directories')
        if record_rank(previous['record'], kind) >= rank:
            # Do not replace a stronger bundle with a later but weaker model.
            return previous
    sources = {name: checkpoint/name for name in
               ('model.safetensors', 'optimizer.npz', 'state.json', 'evaluation.json')}
    sources.update({'replay.html': Path(record['replay']),
                    'inspection.json': Path(record['inspection'])})
    expected = {name: digest(path) for name, path in sources.items()}
    versions = root/'versions'
    versions.mkdir(exist_ok=True)
    version = expected['model.safetensors']+'-'+expected['replay.html'][:16]
    target = versions/version
    manifest = dict(format=FORMAT, kind=kind, version=version, record=record, files=expected,
                    note='Validation is reused. Single-game records are not model selection or fresh testing.')
    if not target.exists():
        stage = Path(tempfile.mkdtemp(prefix='.staging-', dir=versions))
        for name, source in sources.items():
            shutil.copy2(source, stage/name)
        (stage/'selection.json').write_text(json.dumps(record, indent=2)+'\n')
        (stage/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        if any(digest(stage/name) != value for name, value in expected.items()):
            raise ValueError('artifact copy verification failed; previous best retained')
        stage.rename(target)
    elif any(digest(target/name) != value for name, value in expected.items()):
        raise ValueError('existing immutable artifact was modified; previous best retained')
    # The version contains all model, optimizer, configuration, validation and
    # replay data. The stable HTML is a standalone copy, never a network redirect.
    temporary = root/'replay.tmp.html'
    shutil.copy2(target/'replay.html', temporary)
    if digest(temporary) != expected['replay.html']:
        raise ValueError('stable replay copy verification failed')
    temporary.replace(root/'replay.html')
    temporary_json = root/'current.tmp.json'
    temporary_json.write_text(json.dumps(manifest, indent=2)+'\n')
    temporary_json.replace(current)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('selection', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kind', choices=['validation_selected', 'single_game_best_effort'],
                        default='validation_selected')
    args = parser.parse_args()
    result = publish_record(json.loads(args.selection.read_text()), args.output, args.kind)
    print(json.dumps({k: result[k] for k in ('kind', 'version', 'files')}, indent=2))


if __name__ == '__main__':
    main()
