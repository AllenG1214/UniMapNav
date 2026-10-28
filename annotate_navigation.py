"""Create benchmark-shaped navigation annotations on existing annotated maps.

Run: .venv/Scripts/python.exe annotate_navigation.py
Object descriptions must already be grounded and uniqueness_verified.
Generated routes are algorithmic references, never human expert trajectories.
"""
import argparse
import hashlib
import heapq
import json
import random
from pathlib import Path

import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree

LANDMARKS = {'bed', 'sofa', 'arm_chair', 'desk', 'dining_table', 'counter_top',
             'dresser', 'sink', 'toilet', 'television', 'fridge', 'house_plant',
             'shelf', 'washing_machine', 'clothes_dryer'}

PREFERENCE_CLEARANCE_M = 1.5


def select_preference_anchor(anchors, footprints, baseline, goals, target_id,
                             resolution, near):
    """Reject far references within 1.5 m of either endpoint or the target.

    The entire arrival region and target footprint are checked, not just their
    centers. This filters reference selection; it is not a hard route constraint.
    """
    clearance = PREFERENCE_CLEARANCE_M / resolution
    protected = [baseline[0], *goals, *footprints[str(target_id)]['cells']]
    nearby = []
    for candidate in anchors:
        if candidate['id'] == target_id or candidate['category'] not in LANDMARKS:
            continue
        points = footprints[str(candidate['id'])]['cells']
        if not points:
            continue
        tree = cKDTree(points)
        if not near and float(tree.query(protected)[0].min()) <= clearance:
            continue
        proximity = float(tree.query(baseline)[0].min())
        if proximity <= clearance:
            nearby.append((proximity + len(candidate['description']) / 10, candidate))
    return min(nearby, key=lambda entry: entry[0])[1] if nearby else None


def english(text):
    for old, new in [('counter top', 'countertop'), ('house plant', 'houseplant'),
                     ('basket ball', 'basketball'), ('soap bar', 'bar of soap'), ('the cd', 'the CD'),
                     ('arm chair', 'armchair')]:
        text = text.replace(old, new)
    return text


def descriptions(data, all_candidates=False, max_depth=4):
    """Use verified bottom-up expressions, including unique category names."""
    from hierarchical_descriptions import catalog
    result = {}
    for entries in catalog(data, max_depth=max_depth)['layers'].values():
        for entry in entries:
            result.setdefault(entry['object_id'], []).extend(
                (english(text), entry['rule']) for text in entry['expressions'])
    return result if all_candidates else {oid: candidates[0] for oid, candidates in result.items()}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def route(free, start, goals, penalty):
    """Dijkstra on four-connected free cells; positive costs encode soft preferences."""
    queue = [(0.0, start)]
    costs, parents = {start: 0.0}, {}
    while queue:
        cost, cell = heapq.heappop(queue)
        if cost != costs[cell]:
            continue
        if cell in goals:
            path = [cell]
            while cell != start:
                cell = parents[cell]
                path.append(cell)
            return path[::-1]
        r, c = cell
        for nxt in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            y, x = nxt
            if not (0 <= y < free.shape[0] and 0 <= x < free.shape[1] and free[nxt]):
                continue
            value = cost + 1.0 + float(penalty[nxt])
            if value < costs.get(nxt, float('inf')):
                costs[nxt], parents[nxt] = value, cell
                heapq.heappush(queue, (value, nxt))
    raise ValueError('Target is unreachable')


def annotate(source, output, seed, per_task, limit, max_depth=4, references=None):
    raw = source.read_bytes()
    data = json.loads(raw)
    refs = references if references is not None else descriptions(data, max_depth=max_depth)
    free = np.array(data['layers']['occupancy']) == data['layer_legends']['occupancy']['free']['value']
    components, count = label(free)
    if not count:
        raise ValueError(f'No free space: {source}')
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    free &= components == sizes.argmax()
    cells = np.argwhere(free)
    resolution = data['metadata']['map_info']['resolution']
    max_approach = 1.0 / resolution
    candidates, excluded = [], []
    footprints = data['object_footprints']
    for obj in data['object_instances']:
        reason = None
        if obj['category'] in ('wall', 'window', 'doorway'):
            reason = 'Structural element; not a visit target'
        elif obj['id'] not in refs:
            reason = 'Description does not uniquely identify this object'
        points = footprints[str(obj['id'])].get('cells', [])
        if not points:
            reason = reason or 'No object footprint'
        if reason:
            excluded.append({'object_id': obj['id'], 'reason': reason})
            continue
        distances = cKDTree(points).query(cells)[0]
        nearest = float(distances.min())
        if nearest > max_approach:
            excluded.append({'object_id': obj['id'], 'reason': 'No approach within one metre in the largest connected free region'})
            continue
        # A small free-cell approach region, tied to the physical object footprint.
        goals = [tuple(map(int, p)) for p in cells[distances <= nearest + 2]]
        obj = dict(obj, description=refs[obj['id']][0],
                   description_constraints=refs[obj['id']][1])
        candidates.append((obj, goals))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    # Lead the preview with common navigation landmarks; retain all other targets.
    candidates.sort(key=lambda pair: (pair[0]['category'] not in LANDMARKS,
                                     len(pair[0]['description'])))
    map_id = data['metadata']['map_id']
    report = {'map_id': map_id, 'source_map': str(source.resolve()),
              'source_sha256': hashlib.sha256(raw).hexdigest(), 'excluded_objects': excluded,
              'eligible_objects': len(candidates), 'instructions': []}
    groups = [candidates[i:i+per_task] for i in range(0, len(candidates), per_task)]
    if limit:
        groups = groups[:limit]
    for number, group in enumerate(groups, 1):
        start = tuple(map(int, cells[rng.randrange(len(cells))]))
        current, trajectory, hard, soft, phrases, objects = start, [list(start)], [], [], [], []
        target_ids = {o['id'] for o, _ in group}
        anchors = [o for o, _ in candidates if o['id'] not in target_ids]
        for order, (obj, goals) in enumerate(group, 1):
            objects.append({'order': order, 'object_id': obj['id'], 'name': obj['name'],
                            'category': obj['category'], 'center': obj.get('grounding_center_grid', obj['center_grid']),
                            'description': obj['description'], 'description_constraints': obj['description_constraints']})
            hard.append({'constraint_id': f'hard_{number}_{order}', 'kind': 'must_pass',
                         'shape': 'freeform', 'center': np.mean(goals, axis=0).tolist(),
                         'radius': 6.0, 'width': 12.0, 'height': 12.0, 'order': order,
                         'source': 'template_object', 'object_id': obj['id'],
                         'scope': {'type': 'global', 'from_order': None, 'to_order': None},
                         'label': 'Must Pass', 'cells': [list(p) for p in goals]})
            penalty = np.zeros(free.shape)
            phrase = ('Go to ' if order == 1 else 'Finally, head to ' if order == len(group) else 'Then make your way to ') + obj['description'] + '.'
            if anchors:
                baseline = route(free, current, set(goals), penalty)
                anchor = select_preference_anchor(
                    anchors, footprints, baseline, goals, obj['id'], resolution,
                    near=order % 2 == 1)
            else:
                anchor = None
            if anchor:
                near = order % 2 == 1
                distances = cKDTree(footprints[str(anchor['id'])]['cells']).query(cells)[0]
                scale = max(1.0, float(distances.max()))
                penalty[free] = 2.0 * (distances / scale if near else 1.0 - distances / scale)
                phrase += (' Along the way, try to stay close to ' if near else ' During this part of the route, try to keep your distance from ') + anchor['description'] + '.'
                soft.append({'constraint_id': f'soft_{number}_{order}',
                             'preference_type': 'near_preference' if near else 'far_preference',
                             'shape': 'freeform', 'center': anchor.get('grounding_center_grid', anchor['center_grid']),
                             'radius': 6.0, 'width': 12.0, 'height': 12.0,
                             'object_ids': [anchor['id']], 'reference_regions': [],
                             'reference_region': {'mode': 'object', 'object_id': anchor['id'],
                                                  'cells': footprints[str(anchor['id'])]['cells']},
                             'scope': {'type': 'between_hard_constraints', 'from_order': order-1, 'to_order': order},
                             'label': 'Soft Constraint', 'cells': [],
                             'reference_description': anchor['description'],
                             'reference_description_constraints': anchor['description_constraints'],
                             'D_ref': None, 'reference_waypoint_count': 0,
                             'selection_policy': 'endpoint_compatible_v1',
                             'far_endpoint_clearance_m': None if near else PREFERENCE_CLEARANCE_M,
                             'generation_cost': '2 * normalized footprint distance' if near else '2 * (1 - normalized footprint distance)'})
            segment = route(free, current, set(goals), penalty)
            assert all(free[p] for p in segment)
            assert all(abs(a[0]-b[0])+abs(a[1]-b[1]) == 1 for a, b in zip(segment, segment[1:]))
            assert segment[-1] in goals
            if anchor:
                soft[-1]['D_ref'] = float(cKDTree(footprints[str(anchor['id'])]['cells']).query(segment)[0].mean())
                soft[-1]['reference_waypoint_count'] = len(segment)
                soft[-1]['reference_metric_status'] = 'computed'
            trajectory.extend([list(p) for p in segment[1:]])
            current = segment[-1]
            phrases.append(phrase)
        record = {'id': number, 'map_id': map_id, 'template_instruction_id': f'auto_{number:06d}',
                  'object_count': len(objects), 'objects': objects, 'instruction': '\n\n'.join(phrases),
                  'start_pose': {'row': start[0], 'col': start[1]}, 'feasible': True,
                  'name': ' -> '.join(o['name'] for o in objects), 'ordering': ' -> '.join(o['name'] for o in objects),
                  'hard_constraints': hard, 'soft_constraints': soft,
                  'human_expert_trajectory': [], 'generated_reference_trajectory': trajectory,
                  'created_by': 'automatic_navigation_annotator', 'notes': 'Automatically generated; not human-reviewed.',
                  'annotation_provenance': {'source_sha256': report['source_sha256'], 'seed': seed,
                                            'planner': 'four-connected weighted Dijkstra',
                                            'coordinate_order': '[row, col]', 'scope_order_zero': 'start_pose',
                                            'preference_selection_policy': 'endpoint_compatible_v1',
                                            'trajectory_verified': True}}
        destination = output / 'instruction_files' / f'instruction_{number:06d}.json'
        write(destination, record)
        report['instructions'].append({'file': str(destination.resolve()), 'target_ids': list(target_ids)})
    report['covered_target_ids'] = sorted({i for entry in report['instructions'] for i in entry['target_ids']})
    write(output / 'annotation_report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--objects-per-task', type=int, default=3)
    parser.add_argument('--tasks-per-map', type=int, default=0, help='0 covers all eligible targets')
    args = parser.parse_args()
    if args.objects_per_task < 1 or args.tasks_per_map < 0:
        parser.error('Invalid task counts')
    output = args.output or args.batch / 'train'
    results = []
    for source in sorted(args.batch.rglob('*.json')):
        if output.resolve() in source.resolve().parents or source.stem != source.parent.name:
            continue
        report = annotate(source, output / source.stem, args.seed, args.objects_per_task, args.tasks_per_map)
        print(f'{source.stem}: {len(report["instructions"])} tasks, {len(report["covered_target_ids"])} targets, {len(report["excluded_objects"])} excluded', flush=True)
        results.append(report)
    if not results:
        parser.error('No source maps found')
    write(output / 'summary.json', results)
    previews = []
    for result in results:
        if not result['instructions']:
            continue
        first = json.loads(Path(result['instructions'][0]['file']).read_text(encoding='utf-8'))
        previews.append({'scene': Path(result['source_map']).stem, 'file': result['instructions'][0]['file'],
                         'instruction': first['instruction']})
    write(output / 'instruction_previews.json', previews)
    (output / 'instruction_previews.md').write_text('\n\n'.join('## ' + r['scene'] + '\n\n' + r['instruction'] for r in previews), encoding='utf-8')


if __name__ == '__main__':
    main()
