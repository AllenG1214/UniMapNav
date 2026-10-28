"""Export arbitrary map batches as hierarchical grounded navigation annotations.

The original resources and map geometry are read-only inputs. Generated routes
remain explicitly distinguished from human expert demonstrations.
"""
import argparse
import copy
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from annotate_navigation import annotate, descriptions, write
from description_variants import DescriptionAllocator

from hierarchical_descriptions import catalog, scene, resolve, depth, difficulty


def instruction(task):
    """Render ordered visits and explicitly scoped preferences together."""
    clauses = []
    for index, obj in enumerate(task['objects']):
        order = obj['order']
        if index == 0:
            verb = ['First, go near ', 'Begin by approaching ', 'Start by making your way to '][task['id'] % 3]
        elif index == len(task['objects']) - 1:
            verb = ['Finally, stop near ', 'Finish by approaching ', 'End your route near '][task['id'] % 3]
        else:
            verb = ['Next, go near ', 'Then continue to ', 'After that, approach '][task['id'] % 3]
        clause = verb + obj['description'] + '.'
        for soft in task['soft_constraints']:
            assert soft['scope'] == {'type': 'between_hard_constraints', 'from_order': order - 1, 'to_order': order} or soft['scope']['to_order'] != order
            if soft['scope']['to_order'] != order:
                continue
            preference = 'stay close to ' if soft['preference_type'] == 'near_preference' else 'keep your distance from '
            clause += ' On this leg of the route, try to ' + preference + soft['reference_description'] + '.'
        clauses.append(clause)
    return ' '.join(clauses)


def main():
    parser = argparse.ArgumentParser(description='Generate reusable hierarchical navigation annotations')
    parser.add_argument('--maps', type=Path, required=True, help='Map JSON or directory to search recursively')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-depth', type=int, default=4)
    parser.add_argument('--objects-per-task', type=int, default=3)
    parser.add_argument('--tasks-per-map', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--enable-counting', action='store_true', help='Generate exact object counts in room selectors')
    parser.add_argument('--enable-absence', action='store_true', help='Generate room selectors using absent object categories')
    args = parser.parse_args()
    if args.max_depth < 1 or args.objects_per_task < 1 or args.tasks_per_map < 0:
        parser.error('Invalid depth or task counts')
    OUTPUT = args.output.resolve()
    candidates = [args.maps] if args.maps.is_file() else sorted(args.maps.rglob('*.json'))
    sources = []
    for path in candidates:
        if OUTPUT == path.resolve() or OUTPUT in path.resolve().parents:
            continue
        if path.is_file() and (args.maps.is_file() or path.stem == path.parent.name):
            value = json.loads(path.read_text(encoding='utf-8-sig'))
            if 'object_instances' in value and 'layers' in value:
                sources.append(path)
    if not sources:
        parser.error('No maps found')
    if len({p.stem for p in sources}) != len(sources):
        parser.error('Map filenames collide; use separate output batches')
    soft_keys = ['reference_trajectory', 'reference_valid_path_length']
    timestamp = datetime.now(timezone.utc).isoformat()
    summary = []
    previews = []
    for source in sources:
        report_path = OUTPUT / source.stem / 'annotation_report.json'
        if report_path.exists():
            raise ValueError(f'Output already contains a run: {report_path}; choose a fresh output directory')
        initial_data = json.loads(source.read_bytes())
        inventory = catalog(initial_data, max_depth=args.max_depth, enable_counting=args.enable_counting, enable_absence=args.enable_absence)
        references = {}
        for entries in inventory['layers'].values():
            for candidate in entries:
                references.setdefault(candidate['object_id'], (candidate['expressions'][0], candidate['rule']))
        scene_objects, scene_rooms = scene(initial_data)
        items = []
        for obj in scene_objects:
            reference = references.get(obj['id'])
            found = resolve(scene_objects, scene_rooms, reference[1]) if reference else []
            items.append({**obj, 'description': reference[0] if reference else None,
                          'description_constraints': reference[1] if reference else None,
                          'description_depth': depth(reference[1]) if reference else None,
                          'uniqueness_verified': len(found) == 1 and found[0]['id'] == obj['id'],
                          'matched_object_ids': [o['id'] for o in found],
                          'description_status': 'verified' if reference else 'excluded_structural' if obj['category'] in {'wall', 'window', 'doorway'} else 'unresolved'})
        write(OUTPUT / source.stem / 'item_annotations.json', {
            'source_map': str(source.resolve()),
            'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'objects': items, 'status_counts': dict(Counter(o['description_status'] for o in items))})
        annotate(source, OUTPUT / source.stem, seed=args.seed, per_task=args.objects_per_task, limit=args.tasks_per_map, max_depth=args.max_depth, references=references)
        report = json.loads(report_path.read_text())
        source = Path(report['source_map'])
        raw = source.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == report['source_sha256']
        data = json.loads(raw)
        write(OUTPUT / source.stem / 'description_catalog.json', inventory)
        pools = {}
        for level, entries in inventory['layers'].items():
            write(OUTPUT / source.stem / f'descriptions_level_{level}.json', entries)
            for candidate in entries:
                pools.setdefault(candidate['object_id'], []).extend((text, candidate['rule']) for text in candidate['expressions'])
        allocator = DescriptionAllocator(data, candidates=pools)
        tasks = [json.loads(Path(entry['file']).read_text()) for entry in report['instructions']]
        counts = Counter(o['object_id'] for task in tasks for o in task['objects'])
        counts.update(s['reference_region']['object_id'] for task in tasks for s in task['soft_constraints'])
        allocator.require_capacity(counts)
        objects, rooms = scene(data)
        free = np.array(data['layers']['occupancy']) == data['layer_legends']['occupancy']['free']['value']
        exported = copy.deepcopy(report)
        exported['instructions'] = []
        for entry, task in zip(report['instructions'], tasks):
            assert task['map_id'] == data['metadata']['map_id']
            for obj in task['objects']:
                obj['description'], obj['description_constraints'] = allocator.take(obj['object_id'])
            for soft in task['soft_constraints']:
                oid = soft['reference_region']['object_id']
                soft['reference_description'], soft['reference_description_constraints'] = allocator.take(oid)
                for key in soft_keys:
                    if key not in soft:
                        soft[key] = [] if key == 'reference_trajectory' else 0.0 if key == 'reference_valid_path_length' else None
            label, reasons = difficulty(task)
            task['difficulty_metadata'] = reasons
            for obj in task['objects']:
                obj['description_depth'] = depth(obj['description_constraints'])
            for soft in task['soft_constraints']:
                soft['reference_description_depth'] = depth(soft['reference_description_constraints'])
            task.update(difficulty_level=label,
                        created_at=task.get('created_at', timestamp), updated_at=timestamp)
            task['instruction'] = instruction(task)
            task['notes'] = 'Automatically generated navigation annotation. The route is in generated_reference_trajectory; no human expert demonstration was collected.'
            assert not task['human_expert_trajectory']
            path = [tuple(p) for p in task['generated_reference_trajectory']]
            assert path[0] == (task['start_pose']['row'], task['start_pose']['col'])
            assert all(free[p] for p in path)
            assert all(abs(a[0]-b[0])+abs(a[1]-b[1]) == 1 for a, b in zip(path, path[1:]))
            cursor = 0
            assert task['object_count'] == len(task['objects']) == len(task['hard_constraints'])
            for obj, hard in zip(task['objects'], task['hard_constraints']):
                rule = obj['description_constraints']
                found = resolve(objects, rooms, rule)
                assert [o['id'] for o in found] == [obj['object_id']]
                assert obj['description'] in task['instruction']
                assert hard['object_id'] == obj['object_id'] and hard['order'] == obj['order']
                goals = {tuple(p) for p in hard['cells']}
                assert all(free[p] for p in goals)
                cursor = next(i for i in range(cursor, len(path)) if path[i] in goals)
            for soft in task['soft_constraints']:
                found = resolve(objects, rooms, soft['reference_description_constraints'])
                assert [o['id'] for o in found] == [soft['reference_region']['object_id']]
                assert soft['reference_description'] in task['instruction']
                assert soft['reference_region']['cells'] == data['object_footprints'][str(found[0]['id'])]['cells']
            destination = OUTPUT / source.stem / 'instruction_files' / f"instruction_{task['id']:06d}_{label}.json"
            write(destination, task)
            old_path = Path(entry['file'])
            if old_path != destination and old_path.resolve().is_relative_to(OUTPUT):
                old_path.unlink()
            exported['instructions'].append({**entry, 'file': str(destination.resolve())})
            if task['id'] == 1:
                previews.append(source.stem + '\n' + task['instruction'])
        assert hashlib.sha256(source.read_bytes()).hexdigest() == report['source_sha256']
        exported['description_diversity'] = {
            'policy': 'No repeated normalized description per scene/object, including target and soft-reference occurrences',
            'occurrences': sum(counts.values()), 'duplicate_descriptions': 0,
            'objects': {str(oid): {'occurrences': n, 'unique_descriptions': len(allocator.used[oid]),
                                  'unique_rules': len(allocator.rule_counts[oid])} for oid, n in counts.items()}}
        assert all(len(allocator.used[oid]) == n for oid, n in counts.items())
        write(OUTPUT / source.stem / 'annotation_report.json', exported)
        summary.append({'scene': source.stem, 'count': len(exported['instructions']), 'source_map': str(source.resolve()),
                        'source_sha256': report['source_sha256'], 'validation': 'passed'})
        print(source.stem, len(exported['instructions']), 'validated', flush=True)
    write(OUTPUT / 'manifest.json', summary)
    (OUTPUT / 'preview.txt').write_text('\n\n'.join(previews), encoding='utf-8')
    (OUTPUT / 'README.md').write_text('Generated navigation annotations. See hierarchical_navigation.md for depth, difficulty, and validation semantics. Manifest links unchanged source maps. Trajectories are algorithmic references, not human demonstrations.', encoding='utf-8')
    print('Total:', sum(s['count'] for s in summary))


if __name__ == '__main__':
    main()
