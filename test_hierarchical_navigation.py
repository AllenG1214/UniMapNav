import unittest
from hierarchical_descriptions import catalog, resolve, resolve_rooms, render, depth, scene, difficulty
from description_variants import DescriptionAllocator
from annotate_navigation import route, select_preference_anchor
from build_train_annotations import instruction
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

class RoomRelationTests(unittest.TestCase):
    def setUp(self):
        self.data = {
            'room_instances': [{'id': 1, 'category': 'bedroom'},
                               {'id': 2, 'category': 'living_room'},
                               {'id': 3, 'category': 'bedroom'}],
            'object_instances': [
                {'id': 1, 'category': 'bed', 'room_id': 1, 'center_grid': [0, 0]},
                {'id': 2, 'category': 'chair', 'room_id': 2, 'center_grid': [0, 3]},
                {'id': 3, 'category': 'chair', 'room_id': 3, 'center_grid': [0, 5]},
                {'id': 4, 'category': 'piano', 'room_id': 3, 'center_grid': [1, 5]},
                {'id': 5, 'category': 'cabinet', 'room_id': 2, 'center_grid': [1, 3]}],
            'layers': {'room': [[1, 1, 1, 2, 2, 3], [1, 1, 1, 2, 2, 3]],
                       'occupancy': [[0] * 6, [0] * 6]},
            'layer_legends': {'occupancy': {'free': {'value': 0}}}}

    def resolved(self, selector, category='chair'):
        objects, rooms = scene(self.data)
        return [o['id'] for o in resolve(objects, rooms, {'category': category, 'room': selector})]

    def test_connected_to_largest_room(self):
        rule = {'adjacent_to': {'rank': 'largest'}}
        self.assertEqual(self.resolved(rule), [2])
        self.assertEqual(depth({'category': 'chair', 'room': rule}), 3)
        self.assertIn('directly connected to the largest room', render({'category': 'chair', 'room': rule}))

    def test_room_containing_object_and_adjacent_room(self):
        self.assertEqual(self.resolved({'contains': 'bed'}, 'bed'), [1])
        self.assertEqual(self.resolved({'adjacent_to': {'contains': 'bed'}}), [2])
        self.assertEqual(self.resolved({'adjacent_to': {'contains': 'piano'}}, 'cabinet'), [5])
        self.assertIn('the room that contains a bed', render(
            {'category': 'chair', 'room': {'adjacent_to': {'contains': 'bed'}}}))

    def test_multiple_objects_in_one_room_are_existential(self):
        self.data['object_instances'].append({'id': 6, 'category': 'bed', 'room_id': 1})
        self.assertEqual(self.resolved({'adjacent_to': {'contains': 'bed'}}), [2])
        self.data['object_instances'].append({'id': 7, 'category': 'bed', 'room_id': 3})
        self.assertEqual(self.resolved({'adjacent_to': {'contains': 'bed'}}), [])

    def test_ambiguous_reference_room_is_rejected(self):
        self.assertEqual(self.resolved({'adjacent_to': {'category': 'bedroom'}}), [])
        objects, rooms = scene(self.data)
        rooms[3]['area_cells'] = rooms[1]['area_cells']
        self.assertEqual(resolve_rooms(objects, rooms, {'rank': 'largest'}), [])

    def test_ambiguous_destination_room_is_rejected_even_for_unique_object(self):
        # The living room connects to two bedrooms. A globally unique bed must
        # not disguise the ambiguous phrase 'the room connected to the living room'.
        selector = {'adjacent_to': {'category': 'living_room'}}
        self.assertEqual(self.resolved(selector, 'bed'), [])
        self.assertEqual(self.resolved(dict(selector, contains='bed'), 'bed'), [1])

    def test_duplicate_targets_are_rejected_by_catalog(self):
        self.data['object_instances'].append({'id': 6, 'category': 'chair', 'room_id': 2, 'center_grid': [0, 3]})
        selector = {'adjacent_to': {'rank': 'largest'}}
        self.assertEqual(self.resolved(selector), [2, 6])
        entries = [e for group in catalog(self.data)['layers'].values() for e in group]
        self.assertFalse(any(e['rule'] == {'category': 'chair', 'room': selector} for e in entries))

    def test_walls_and_missing_evidence_do_not_create_connections(self):
        for row in self.data['layers']['occupancy']:
            row[3] = 1
        self.assertEqual(self.resolved({'adjacent_to': {'rank': 'largest'}}), [])
        self.data['layers'].pop('occupancy')
        self.assertEqual(self.resolved({'adjacent_to': {'contains': 'bed'}}), [])

    def test_diagonal_contact_does_not_create_connection(self):
        self.data['layers'] = {'room': [[1, 0], [0, 2]], 'occupancy': [[0, 1], [1, 0]]}
        _, rooms = scene(self.data)
        self.assertEqual(rooms[1]['connected_room_ids'], [])

    def test_combined_room_predicates_disambiguate(self):
        selector = {'category': 'bedroom', 'contains': 'piano',
                    'adjacent_to': {'category': 'living_room'}}
        self.assertEqual(self.resolved(selector), [3])
        text = render({'category': 'chair', 'room': selector})
        self.assertIn('the bedroom that contains a piano and is directly connected', text)

    def test_nested_room_depth_and_catalog_roundtrip(self):
        selector = {'contains': 'bed', 'adjacent_to': {
            'adjacent_to': {'rank': 'smallest'}}}
        self.assertEqual(self.resolved(selector, 'bed'), [1])
        self.assertEqual(depth({'category': 'bed', 'room': selector}), 4)
        objects, rooms = scene(self.data)
        result = catalog(self.data)
        room_rules = []
        for level, entries in result['layers'].items():
            for entry in entries:
                self.assertEqual(depth(entry['rule']), int(level))
                self.assertEqual([o['id'] for o in resolve(objects, rooms, entry['rule'])], [entry['object_id']])
                if 'room' in entry['rule']:
                    room_rules.append(entry['rule'])
        self.assertIn({'category': 'chair', 'room': {'adjacent_to': {'rank': 'largest'}}}, room_rules)
        self.assertIn({'category': 'chair', 'room': {'adjacent_to': {'contains': 'bed'}}}, room_rules)
        self.assertFalse(any('room' in e['rule'] for group in catalog(self.data, max_depth=1)['layers'].values() for e in group))

    def test_room_reference_allocation_before_paraphrase(self):
        first = {'category': 'chair', 'room': {'adjacent_to': {'rank': 'largest'}}}
        second = {'category': 'chair', 'room': {'adjacent_to': {'contains': 'bed'}}}
        candidates = {2: [(render(first, 0), first), (render(first, 1), first), (render(second, 0), second)]}
        allocator = DescriptionAllocator(self.data, candidates=candidates)
        self.assertEqual(allocator.take(2)[1], first)
        self.assertEqual(allocator.take(2)[1], second)

    def test_room_rules_in_target_and_soft_reference_instruction(self):
        target = {'category': 'chair', 'room': {'adjacent_to': {'rank': 'largest'}}}
        reference = {'category': 'cabinet', 'room': {'adjacent_to': {'contains': 'bed'}}}
        allocator = DescriptionAllocator(self.data, candidates={
            2: [(render(target), target)], 5: [(render(reference), reference)]})
        target_text, target_rule = allocator.take(2)
        ref_text, ref_rule = allocator.take(5)
        task = {'id': 1, 'objects': [{'order': 1, 'description': target_text,
                                    'description_constraints': target_rule}],
                'soft_constraints': [{'scope': {'type': 'between_hard_constraints', 'from_order': 0, 'to_order': 1},
                                      'preference_type': 'near_preference', 'reference_description': ref_text,
                                      'reference_description_constraints': ref_rule}]}
        text = instruction(task)
        self.assertIn(target_text, text)
        self.assertIn(ref_text, text)
        objects, rooms = scene(self.data)
        self.assertEqual([o['id'] for o in resolve(objects, rooms, target_rule)], [2])
        self.assertEqual([o['id'] for o in resolve(objects, rooms, ref_rule)], [5])
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



class ExtendedDescriptionTests(unittest.TestCase):
    def setUp(self):
        self.data = {
            'object_instances': [
                {'id': 1, 'category': 'table', 'room_id': 1, 'center_grid': [3, 1]},
                {'id': 2, 'category': 'sofa', 'room_id': 3, 'center_grid': [3, 9]},
                {'id': 3, 'category': 'chair', 'room_id': 2, 'center_grid': [3, 4]},
                {'id': 4, 'category': 'chair', 'room_id': 3, 'center_grid': [5, 8]},
                {'id': 5, 'category': 'chair', 'room_id': 2, 'center_grid': [6, 4]},
                {'id': 6, 'category': 'lamp', 'room_id': 1, 'center_grid': [2, 1]},
                {'id': 7, 'category': 'clock', 'room_id': 2, 'center_grid': [2, 3]}],
            'room_instances': [{'id': 1, 'category': 'kitchen', 'area_cells': 12},
                               {'id': 2, 'category': 'hall', 'area_cells': 12},
                               {'id': 3, 'category': 'bedroom', 'area_cells': 12}],
            'layers': {'room': [[1, 1, 2, 2, 3, 3]] * 7,
                       'occupancy': [[0] * 6 for _ in range(7)]},
            'layer_legends': {'occupancy': {'free': {'value': 0}}}}

    def ids(self, rule):
        objects, rooms = scene(self.data)
        return [o['id'] for o in resolve(objects, rooms, rule)]

    def pair(self, relation):
        return {'category': 'chair', 'relation': relation,
                'anchor': {'category': 'table'}, 'other_anchor': {'category': 'sofa'}}

    def test_relative_distance_and_tie_margin(self):
        self.assertEqual(self.ids(self.pair('closer_than')), [3, 5])
        self.data['object_instances'][2]['center_grid'] = [3, 5]
        self.assertEqual(self.ids(self.pair('closer_than')), [5])

    def test_between_excludes_off_axis_endpoints_and_outside(self):
        self.assertEqual(self.ids(self.pair('between')), [3])
        self.data['object_instances'][2]['center_grid'] = [3, 1]
        self.assertEqual(self.ids(self.pair('between')), [])
        self.data['object_instances'][2]['center_grid'] = [3, 10]
        self.assertEqual(self.ids(self.pair('between')), [])

    def test_pair_rejects_ambiguous_duplicate_and_missing_anchors(self):
        rule = self.pair('between')
        rule['other_anchor'] = {'category': 'chair'}
        self.assertEqual(self.ids(rule), [])
        rule['other_anchor'] = {'category': 'table'}
        self.assertEqual(self.ids(rule), [])
        rule['other_anchor'] = {'category': 'missing'}
        self.assertEqual(self.ids(rule), [])

    def test_between_degenerate_and_tolerance(self):
        self.data['object_instances'][1]['center_grid'] = [3, 1]
        self.assertEqual(self.ids(self.pair('between')), [])
        self.data['object_instances'][1]['center_grid'] = [3, 9]
        rule = dict(self.pair('between'), between_tolerance_cells=3)
        self.assertEqual(self.ids(rule), [3, 4, 5])

    def test_room_scoped_ranking(self):
        rule = {'category': 'chair', 'room': {'category': 'hall'},
                'relation': 'farthest', 'anchor': {'category': 'table'}}
        self.assertEqual(self.ids(rule), [5])
        self.assertIn('in the hall', render(rule))

    def test_multi_room_and_exact_connectivity(self):
        selector = {'adjacent_to': [{'category': 'kitchen'}, {'category': 'bedroom'}],
                    'connection_count': 2}
        objects, rooms = scene(self.data)
        self.assertEqual([r['id'] for r in resolve_rooms(objects, rooms, selector)], [2])
        self.assertEqual(depth({'category': 'chair', 'room': selector}), 3)
        self.assertIn('exactly 2 other rooms', render({'category': 'chair', 'room': selector}))
        selector['adjacent_to'][1] = {'category': 'kitchen'}
        self.assertEqual(resolve_rooms(objects, rooms, selector), [])

    def test_no_evidence_is_not_zero_connections(self):
        self.data['layers'].pop('occupancy')
        objects, rooms = scene(self.data)
        self.assertEqual(resolve_rooms(objects, rooms, {'connection_count': 0}), [])

    def test_exact_counts_absence_and_ambiguity(self):
        objects, rooms = scene(self.data)
        self.assertEqual([r['id'] for r in resolve_rooms(objects, rooms, {'object_counts': {'chair': 2}})], [2])
        self.assertEqual([r['id'] for r in resolve_rooms(objects, rooms, {'absent_categories': ['chair']})], [1])
        self.assertEqual(resolve_rooms(objects, rooms, {'absent_categories': ['lamp']}), [])
        self.assertEqual(resolve_rooms(objects, rooms, {'object_counts': {'chair': -1}}), [])

    def test_catalog_generates_new_families_and_all_roundtrip(self):
        entries = [e for group in catalog(self.data)['layers'].values() for e in group]
        rules = [e['rule'] for e in entries]
        self.assertTrue(any(r.get('relation') == 'between' for r in rules))
        self.assertTrue(any(r.get('relation') == 'closer_than' for r in rules))
        self.assertTrue(any(r.get('anchor') and r.get('room') for r in rules))
        self.assertTrue(any(isinstance(r.get('room', {}).get('adjacent_to'), list) for r in rules))
        self.assertTrue(any('connection_count' in r.get('room', {}) for r in rules))
        for e in entries:
            self.assertEqual(self.ids(e['rule']), [e['object_id']])
            self.assertEqual(depth(e['rule']), e['depth'])
            self.assertEqual(len(e['dependencies']), len(set(e['dependencies'])))

    def test_extensions_work_in_target_and_soft_reference(self):
        target = self.pair('between')
        reference = dict(self.pair('closer_than'), category='lamp')
        allocator = DescriptionAllocator(self.data, candidates={
            3: [(render(target), target)], 6: [(render(reference), reference)]})
        text, rule = allocator.take(3)
        ref_text, ref_rule = allocator.take(6)
        task = {'id': 2, 'objects': [{'order': 1, 'description': text, 'description_constraints': rule}],
                'soft_constraints': [{'scope': {'type': 'between_hard_constraints', 'from_order': 0, 'to_order': 1},
                                      'preference_type': 'far_preference', 'reference_description': ref_text,
                                      'reference_description_constraints': ref_rule}]}
        rendered = instruction(task)
        self.assertIn('between', rendered)
        self.assertIn('closer to', rendered)
        self.assertIn('keep your distance from', rendered)
        self.assertEqual(self.ids(rule), [3])
        self.assertEqual(self.ids(ref_rule), [6])

    def test_count_absence_generation_opt_in(self):
        def selectors(result):
            return [e['rule'].get('room', {}) for g in result['layers'].values() for e in g]
        normal = catalog(self.data, max_depth=2)
        self.assertFalse(any('object_counts' in r or 'absent_categories' in r for r in selectors(normal)))
        enabled = catalog(self.data, max_depth=2, enable_counting=True, enable_absence=True)
        self.assertTrue(any('object_counts' in r for r in selectors(enabled)))
        self.assertTrue(any('absent_categories' in r for r in selectors(enabled)))
        self.assertTrue(enabled['enabled_features']['counting'])

    def test_pair_nested_depth_and_instruction_allocation(self):
        rule = self.pair('between')
        rule['anchor'] = {'category': 'table', 'room_category': 'kitchen'}
        self.assertEqual(depth(rule), 3)
        allocator = DescriptionAllocator(self.data, candidates={3: [(render(rule, v), rule) for v in range(4)]})
        text, allocated = allocator.take(3)
        task = {'id': 1, 'objects': [{'order': 1, 'description': text, 'description_constraints': allocated}],
                'soft_constraints': []}
        self.assertIn('between', instruction(task))
        self.assertEqual(difficulty(task)[0], 'hard')
        self.assertEqual(self.ids(allocated), [3])


if __name__ == '__main__':
    unittest.main()
