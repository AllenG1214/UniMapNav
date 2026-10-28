"""Sample a reproducible manifest and run simulator exports in an existing Linux environment."""
import argparse
import gzip
import json
import random
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=10)
    args = parser.parse_args()
    dataset = args.repo / 'resources/procthor/procthor-10k/train.jsonl.gz'
    with gzip.open(dataset, 'rt', encoding='utf-8') as stream:
        population = sum(1 for _ in stream)
    if population != 10000:
        raise ValueError(f'Expected 10000 training maps, found {population}')
    args.output.mkdir(parents=True, exist_ok=False)
    seed = random.SystemRandom().randrange(2**32)
    indices = random.Random(seed).sample(range(population), args.count)
    manifest = {'dataset': 'procthor-10k', 'split': 'train', 'population': population,
                'seed': seed, 'selected_indices_zero_based': indices,
                'selection': 'random sample without replacement', 'scenes': []}
    manifest_path = args.output / 'selection_and_results.json'
    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    save()
    print('SELECTED', indices, flush=True)
    for number, index in enumerate(indices, 1):
        name = f'train_{index:05d}'
        log_path = args.output / f'{name}.log'
        command = [sys.executable, str(Path(__file__).with_name('generate_map_full.py')),
                   '--repo', str(args.repo), '--split', 'train', '--index', str(index),
                   '--output-root', str(args.output / 'maps'), '--output-name', name]
        print(f'EXPORT {number}/{args.count}: {name}', flush=True)
        with log_path.open('w', encoding='utf-8') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=900)
        record = {'index': index, 'folder': name, 'exit_code': result.returncode,
                  'log_file': log_path.name, 'status': 'failed'}
        if result.returncode == 0:
            path = args.output / 'maps' / name / f'{name}.json'
            data = json.loads(path.read_text())
            objects = data['object_instances']
            assert len({o['id'] for o in objects}) == len(objects)
            assert all(str(o['id']) in data['object_footprints'] for o in objects)
            assert data['metadata']['export_mode'] == 'simulator_backed'
            record.update(status='verified', objects=len(objects), rooms=len(data['room_instances']),
                          map_file=str(path.relative_to(args.output)))
        manifest['scenes'].append(record)
        save()
        print(f'FINISHED {number}/{args.count}: {record}', flush=True)
        if result.returncode:
            raise RuntimeError(f'Export failed; inspect {log_path}')
    print('EXPORT_BATCH_COMPLETE', flush=True)

if __name__ == '__main__':
    main()
