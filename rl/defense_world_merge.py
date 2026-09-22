"""Hash-checked union of own collections, preserving every held-out assignment."""

import argparse
import json
from pathlib import Path
import subprocess

from .defense_learning import sha256, write_json
from .defense_world_data import Sequences, disk_guard


def merge(sources, output):
    sources = [Path(p).resolve(strict=True) for p in sources]
    output = Path(output)
    if len(sources) < 2 or len(set(sources)) != len(sources) or output.exists():
        raise ValueError('requires distinct sources and a new output')
    manifests = [json.loads((p/'manifest.json').read_text()) for p in sources]
    hashes = [sha256(p/'manifest.json') for p in sources]
    if len(set(hashes)) != len(hashes):
        raise ValueError('duplicate source collection')
    common = ('schema', 'purpose', 'game_sha256', 'environment_version', 'tstates',
              'observation_stride', 'parent_hashes', 'existing_evaluation_data')
    seen_seeds = set()
    for source, manifest in zip(sources, manifests, strict=True):
        if any(manifest[k] != manifests[0][k] for k in common):
            raise ValueError('incompatible own collections')
        # Decode/check all files before creating any output.
        for split in ('train', 'heldout'):
            Sequences(source, split, 1)
        for row in manifest['episodes']:
            if row['split'] not in ('train', 'heldout') or row['seed'] in seen_seeds:
                raise ValueError('invalid split or overlapping collection seeds')
            seen_seeds.add(row['seed'])
    disk_guard(output.parent)
    output.mkdir(parents=True, exist_ok=False)
    episodes = []
    for number, (source, manifest) in enumerate(zip(sources, manifests, strict=True)):
        for row in manifest['episodes']:
            original = source/row['file']
            destination = output/f"source-{number:02d}-{row['file']}"
            disk_guard(output)
            subprocess.run(['cp', '-c', str(original), str(destination)], check=True)
            if sha256(original) != row['sha256'] or sha256(destination) != row['sha256']:
                raise RuntimeError('source changed or union copy mismatch')
            episodes.append(dict(row, file=destination.name, source_index=number, source_file=row['file']))
        if sha256(source/'manifest.json') != hashes[number]:
            raise RuntimeError('source manifest changed')
    combined = dict(manifests[0], episodes=episodes, games=len(episodes),
        heldout_games=sum(e['split']=='heldout' for e in episodes), epsilon=None, first_seed=None,
        assignment='unchanged original per-source whole-game assignments',
        dataset_operation='immutable byte-identical union; no new trajectories or split changes',
        merge_source_sha256=sha256(Path(__file__)),
        source_manifests=[dict(path=str(p), sha256=digest, epsilon=m['epsilon'])
                          for p,digest,m in zip(sources, hashes, manifests, strict=True)])
    write_json(output/'manifest.json', combined)
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sources', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = merge(args.sources, args.output)
    print(json.dumps(dict(games=result['games'], heldout_games=result['heldout_games'],
        train_actions=sum(e['steps'] for e in result['episodes'] if e['split']=='train'),
        heldout_actions=sum(e['steps'] for e in result['episodes'] if e['split']=='heldout'))))


if __name__ == '__main__':
    main()
