"""Verify saved instructions against source maps, object grounding and route constraints."""
import argparse
import hashlib
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree
from hierarchical_descriptions import resolve as evaluate, scene, difficulty, depth
from description_variants import normalize




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    ROOT = args.root
    count = 0
    reports = list(ROOT.glob('*/annotation_report.json'))
    assert reports, 'No annotation reports found'
    for filename in reports:
        used_descriptions = defaultdict(set)
        report = json.loads(filename.read_text(encoding='utf-8'))
        raw = Path(report['source_map']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == report['source_sha256']
        data = json.loads(raw)
        free = np.array(data['layers']['occupancy']) == data['layer_legends']['occupancy']['free']['value']
        objects, rooms = scene(data)
        for entry in report['instructions']:
            task = json.loads(Path(entry['file']).read_text(encoding='utf-8'))
            assert task['map_id'] == data['metadata']['map_id']
            path = [tuple(p) for p in task['generated_reference_trajectory']]
            assert path[0] == (task['start_pose']['row'], task['start_pose']['col'])
            assert all(free[p] for p in path)
            assert all(abs(a[0]-b[0])+abs(a[1]-b[1]) == 1 for a, b in zip(path, path[1:]))
            assert not task['human_expert_trajectory']
            assert task['object_count'] == len(task['objects']) == len(task['hard_constraints'])
            expected, _ = difficulty(task)
            assert task['difficulty_level'] == expected
            assert Path(entry['file']).stem.endswith('_' + expected)
            assert [o['order'] for o in task['objects']] == list(range(1, len(task['objects']) + 1))
            cursor = 0
            segments = {}
            for obj, hard in zip(task['objects'], task['hard_constraints']):
                text = normalize(obj['description'])
                assert text not in used_descriptions[obj['object_id']], ('Repeated description', filename, obj['object_id'], text)
                used_descriptions[obj['object_id']].add(text)
                found = evaluate(objects, rooms, obj['description_constraints'])
                assert len(found) == 1 and found[0]['id'] == obj['object_id']
                assert obj['description'] in task['instruction']
                assert hard['object_id'] == obj['object_id'] and hard['order'] == obj['order']
                goals = {tuple(p) for p in hard['cells']}
                hits = [i for i in range(cursor, len(path)) if path[i] in goals]
                assert hits
                segments[obj['order']] = path[cursor:hits[0] + 1]
                cursor = hits[0]
            for soft in task['soft_constraints']:
                oid = soft['reference_region']['object_id']
                text = normalize(soft['reference_description'])
                assert text not in used_descriptions[oid], ('Repeated description', filename, oid, text)
                used_descriptions[oid].add(text)
                found = evaluate(objects, rooms, soft['reference_description_constraints'])
                assert len(found) == 1 and found[0]['id'] == soft['reference_region']['object_id']
                assert soft['reference_description'] in task['instruction']
                assert soft['scope']['type'] == 'between_hard_constraints'
                assert soft['scope']['from_order'] == soft['scope']['to_order'] - 1
                assert 1 <= soft['scope']['to_order'] <= len(task['objects'])
                assert soft['reference_region']['cells'] == data['object_footprints'][str(oid)]['cells']
                segment = segments[soft['scope']['to_order']]
                metric = float(cKDTree(soft['reference_region']['cells']).query(segment)[0].mean())
                assert abs(metric - soft['D_ref']) < 1e-6
                assert soft['reference_waypoint_count'] == len(segment)
            count += 1
    print(f'Passed: {len(reports)} source maps unchanged; {count} instructions grounded; routes free and ordered; no repeated object descriptions.')


if __name__ == '__main__':
    main()
