"""Run the original simulator-backed exporter on a downloaded ProcTHOR scene.
Run under Linux; never silently fall back to the floor-only exporter.
"""
import argparse
import gzip
import json
import sys
import hashlib
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo', type=Path, required=True)
parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
parser.add_argument('--index', type=int, default=0)
parser.add_argument('--resolution', type=float, default=0.05)
parser.add_argument('--platform', choices=['linux', 'cloud'], default='linux')
parser.add_argument('--runtime-zip', type=Path)
parser.add_argument('--output-name')
parser.add_argument('--output-root', type=Path)
args = parser.parse_args()
if sys.platform != 'linux':
    parser.error('Run this script inside Linux/WSL, not Windows Python.')
if args.index < 0:
    parser.error('--index must be nonnegative')
repo = args.repo.resolve()
sys.path.insert(0, str(repo))
from scripts.make_maps.procthor.transform_procthor_to_map import transform_procthor_to_map
from scripts.make_maps.procthor.regenerate_procthor_maps import migrate_map_payload
from ai2thor.platform import Linux64, CloudRendering
if args.runtime_zip:
    import ai2thor.downloader
    original_download = ai2thor.downloader.download
    def local_download(url, sha256_digest, include_private_scenes=False):
        if 'thor-Linux64-f0825767cd50d69f666c7f282e54abfe58f1e917.zip' not in url:
            return original_download(url, sha256_digest, include_private_scenes)
        contents = args.runtime_zip.read_bytes()
        if hashlib.sha256(contents).hexdigest() != sha256_digest:
            raise ValueError('Downloaded runtime archive checksum mismatch')
        print('Official runtime archive checksum verified', flush=True)
        return contents
    ai2thor.downloader.download = local_download

house = None
with gzip.open(repo / 'resources/procthor/procthor-10k' / (args.split + '.jsonl.gz'), 'rt', encoding='utf-8') as stream:
    for index, line in enumerate(stream):
        if index == args.index:
            house = json.loads(line)
            break
if house is None:
    raise IndexError('Scene index outside downloaded dataset')
name = args.output_name or f'simulator_{args.split}_{args.index:05d}'
if Path(name).name != name or name in ('.', '..'):
    parser.error('--output-name must be a folder name')
folder = (args.output_root or repo / 'generated_maps') / name
if folder.exists():
    raise FileExistsError(f'Refusing to overwrite existing export: {folder}')
folder.mkdir(parents=True)
prefix = folder / name
scene, arrays_path, metadata_path, preview_path = transform_procthor_to_map(
    args.split, args.index, dataset={args.split: [None] * args.index + [house]},
    resolution=args.resolution, output_prefix=prefix, use_simulator=True,
    controller_kwargs={'platform': CloudRendering if args.platform == 'cloud' else Linux64},
    ai2thor_base_dir=Path.home() / '.cache/map2route-ai2thor',
)
map_path = prefix.with_suffix('.json')
payload = json.loads(map_path.read_text())
metadata = json.loads(metadata_path.read_text())
thinggraph = json.loads(Path(str(prefix) + '_thinggraph.json').read_text())
payload = migrate_map_payload(payload, metadata, thinggraph)
payload['metadata']['export_mode'] = 'simulator_backed'
payload['metadata']['export_folder'] = name
map_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
report = {'status': 'simulator_export_completed', 'source_split': args.split,
          'source_index': args.index, 'platform': args.platform, 'map_file': str(map_path),
          'preview_file': str(preview_path), 'objects': len(payload['object_instances']),
          'rooms': len(payload['room_instances']), 'footprints': len(payload['object_footprints'])}
(folder / 'export_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
