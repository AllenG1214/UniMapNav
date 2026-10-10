import unittest
from hierarchical_descriptions import catalog, resolve, depth, scene, difficulty
from description_variants import DescriptionAllocator
from annotate_navigation import route, select_preference_anchor
import json
from pathlib import Path
import numpy as np

class HierarchyTests(unittest.TestCase):
    def setUp(self):
        self.data = {'object_instances': [
            {'id': 1, 'category': 'sofa', 'room_id': 1, 'center_grid': [0, 0]},
            {'id': 2, 'category': 'chair', 'room_id': 1, 'center_grid': [0, 3]},
            {'id': 3, 'category': 'chair', 'room_id': 2, 'center_grid': [0, 9]},
            {'id': 4, 'category': 'lamp', 'room_id': 1, 'center_grid': [0, 4]},
            {'id': 5, 'category': 'desk', 'room_id': 2, 'center_grid': [0, 12]}],
            'room_instances': [{'id': 1, 'category': 'living_room', 'area_cells': 20},
                               {'id': 2, 'category': 'bedroom', 'area_cells': 10}]}

    def test_layers_roundtrip_and_no_cycles(self):
        result = catalog(self.data)
        objects, rooms = scene(self.data)
        for level, entries in result['layers'].items():
            self.assertTrue(entries)
            for entry in entries:
                self.assertEqual(depth(entry['rule']), int(level))
                self.assertEqual([o['id'] for o in resolve(objects, rooms, entry['rule'])], [entry['object_id']])
                self.assertEqual(len(entry['dependencies']), len(set(entry['dependencies'])))
        self.assertIn('the sofa', [t for e in result['layers']['1'] for t in e['expressions']])

    def test_second_farthest_and_tie(self):
        objects, rooms = scene(self.data)
        rule = {'category': 'chair', 'relation': 'farthest', 'rank': 2, 'anchor': {'category': 'sofa'}}
        self.assertEqual([o['id'] for o in resolve(objects, rooms, rule)], [2])
        objects[2]['center_grid'] = [0, 3]
        self.assertEqual(len(resolve(objects, rooms, rule)), 2)

    def test_ambiguous_room_and_size(self):
        objects, rooms = scene(self.data)
        rooms[2]['category'] = 'living_room'
        self.assertEqual(resolve(objects, rooms, {'category': 'sofa', 'room_category': 'living_room'}), [])
        rooms[2]['area_cells'] = 20
        self.assertEqual(resolve(objects, rooms, {'category': 'sofa', 'room_rank': 'largest'}), [])

    def test_allocator_exhaustion(self):
        allocator = DescriptionAllocator(self.data, candidates={1: [('the sofa', {'category': 'sofa'})]})
        self.assertEqual(allocator.take(1)[0], 'the sofa')
        with self.assertRaises(ValueError):
            allocator.take(1)

    def test_unreachable(self):
        with self.assertRaises(ValueError):
            route(np.array([[True, False, True]]), (0, 0), {(0, 2)}, np.zeros((1, 3)))

    def test_nonunique_anchor_is_rejected(self):
        objects, rooms = scene(self.data)
        rule = {'category': 'lamp', 'relation': 'nearest', 'anchor': {'category': 'chair'}}
        self.assertEqual(resolve(objects, rooms, rule), [])

    def test_colocated_objects_remain_unresolved(self):
        data = {'object_instances': [
            {'id': 1, 'category': 'cup', 'center_grid': [3, 3]},
            {'id': 2, 'category': 'cup', 'center_grid': [3, 3]},
            {'id': 3, 'category': 'table', 'center_grid': [8, 8]}]}
        result = catalog(data)
        self.assertEqual(set(result['unresolved_object_ids']), {1, 2})

    def test_difficulty(self):
        task = {'objects': [{'description_constraints': {'category': 'sofa'}}], 'soft_constraints': []}
        self.assertEqual(difficulty(task)[0], 'easy')
        task['objects'][0]['description_constraints'] = {'category': 'lamp', 'relation': 'nearest', 'anchor': {'category': 'chair', 'relation': 'nearest', 'anchor': {'category': 'sofa'}}}
        self.assertEqual(difficulty(task)[0], 'hard')

class PreferenceSelectionTests(unittest.TestCase):
    def choose(self, reference_cells, *, near=False, goals=None, target=None):
        anchor = {'id': 2, 'category': 'toilet', 'description': 'the toilet'}
        return select_preference_anchor(
            [anchor], {'1': {'cells': target or [[0, 20]]},
                       '2': {'cells': reference_cells}},
            [(0, c) for c in range(20)], goals or [(0, 19)], 1, 1.0, near)

    def test_far_rejects_target_neighbor_but_near_allows_it(self):
        self.assertIsNone(self.choose([[1, 19]]))
        self.assertEqual(self.choose([[1, 19]], near=True)['id'], 2)

    def test_far_checks_start_and_entire_goal_region(self):
        self.assertIsNone(self.choose([[1, 0]]))
        self.assertIsNone(self.choose([[1, 10]], goals=[(0, 19), (0, 10)]))

    def test_far_checks_target_footprint_not_only_arrival(self):
        self.assertIsNone(self.choose([[1, 10]], target=[[0, 20], [1, 11]]))

    def test_far_accepts_midroute_landmark_but_not_irrelevant_one(self):
        self.assertEqual(self.choose([[1, 10]])['id'], 2)
        self.assertIsNone(self.choose([[5, 10]]))

    def test_far_tries_another_anchor_and_uses_metric_clearance(self):
        anchors = [{'id': i, 'category': 'toilet', 'description': 'the toilet'} for i in [2, 3]]
        footprints = {'1': {'cells': [[0, 100]]}, '2': {'cells': [[1, 98]]},
                      '3': {'cells': [[1, 50]]}}
        result = select_preference_anchor(anchors, footprints, [(0, c) for c in range(100)],
                                          [(0, 99)], 1, .05, False)
        self.assertEqual(result['id'], 3)
        footprints['3']['cells'] = [[0, 69]]
        self.assertIsNone(select_preference_anchor(
            anchors, footprints, [(0, c) for c in range(100)], [(0, 99)], 1, .05, False))

    def test_saved_painting_toilet_conflict(self):
        base = Path(__file__).resolve().parent / 'generated_maps/trajectory_test'
        map_path = base / 'maps/train_02442/train_02442.json'
        if not map_path.exists():
            self.skipTest('Optional saved scene is unavailable')
        data = json.loads(map_path.read_text(encoding='utf-8'))
        task = json.loads((base / 'train/train_02442/instruction_files/instruction_000029_medium.json').read_text(encoding='utf-8'))
        path = task['generated_reference_trajectory']
        soft = task['soft_constraints'][1]
        anchor = next(dict(o, description=soft['reference_description'])
                      for o in data['object_instances'] if o['id'] == 128)
        start = task['soft_constraints'][0]['reference_waypoint_count'] - 1
        end = start + soft['reference_waypoint_count']
        self.assertIsNone(select_preference_anchor(
            [anchor], data['object_footprints'], path[start:end],
            task['hard_constraints'][1]['cells'], 85,
            data['metadata']['map_info']['resolution'], False))


if __name__ == '__main__':
    unittest.main()
