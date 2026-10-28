"""Bounded bottom-up referring expressions; semantic depth is independent of wording."""
import json
import math
from itertools import combinations, islice
from collections import Counter, defaultdict

def noun(category):
    return category.replace('_', ' ')


def position(obj):
    value = obj.get('center_grid') or obj.get('metadata_grid_cell')
    return tuple(map(float, value)) if value and len(value) == 2 else None


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])



def room_anchors(rule):
    value = rule.get('adjacent_to', [])
    return value if isinstance(value, list) else [value]


def room_depth(rule):
    return max([1] + [1 + room_depth(r) for r in room_anchors(rule)])


def depth(rule):
    levels = [1]
    for key in ('anchor', 'other_anchor'):
        if rule.get(key):
            levels.append(1 + depth(rule[key]))
    if rule.get('room'):
        levels.append(1 + room_depth(rule['room']))
    if rule.get('room_category') or rule.get('room_rank'):
        levels.append(2)
    return max(levels)


def scene(data):
    objects = [dict(o, center_grid=o.get('grounding_center_grid', o.get('center_grid'))) for o in data['object_instances']]
    rooms = {r['id']: dict(r) for r in data.get('room_instances', [])}
    areas = Counter(v for row in data.get('layers', {}).get('room', []) for v in row)
    for rid, room in rooms.items():
        room['area_cells'] = areas.get(rid, room.get('area_cells', 0))
        room['connected_room_ids'] = []
    # Connectivity is evidence from the supplied raster, never inferred from
    # room names, bounding boxes, shared walls, or doorway object names.
    layers = data.get('layers', {})
    grid, occupancy = layers.get('room', []), layers.get('occupancy', [])
    free_value = data.get('layer_legends', {}).get('occupancy', {}).get('free', {}).get('value')
    valid = (free_value is not None and bool(grid) and len(grid) == len(occupancy)
             and bool(grid[0]) and all(len(row) == len(grid[0]) for row in grid)
             and all(len(row) == len(grid[0]) for row in occupancy))
    edges = {rid: set() for rid in rooms}
    if valid:
        for row, cells in enumerate(grid):
            for col, rid in enumerate(cells):
                if rid not in rooms or occupancy[row][col] != free_value:
                    continue
                for y, x in ((row + 1, col), (row, col + 1)):
                    if y >= len(grid) or x >= len(cells):
                        continue
                    other = grid[y][x]
                    if other in rooms and other != rid and occupancy[y][x] == free_value:
                        edges[rid].add(other)
                        edges[other].add(rid)
    for rid, room in rooms.items():
        room['connected_room_ids'] = sorted(edges[rid])
        room['connectivity_known'] = valid and any(rid in row for row in grid)
    return objects, rooms


def resolve_rooms(objects, rooms, rule):
    """Every explicit room reference must denote exactly one room.

    `contains` is existential: multiple beds in one room are acceptable, but
    beds in multiple rooms do not identify 'the room that contains a bed'.
    """
    if not rule or set(rule) - {'category', 'rank', 'contains', 'adjacent_to', 'connection_count', 'object_counts', 'absent_categories'}:
        return []
    pool = list(rooms.values())
    if 'category' in rule:
        pool = [r for r in pool if r['category'] == rule['category']]
    if 'rank' in rule:
        sizes = [r.get('area_cells', 0) for r in rooms.values()]
        if rule['rank'] not in {'largest', 'smallest'} or not sizes or min(sizes) <= 0:
            return []
        best = max(sizes) if rule['rank'] == 'largest' else min(sizes)
        selected = [r['id'] for r in rooms.values() if r['area_cells'] == best]
        if len(selected) != 1:
            return []
        pool = [r for r in pool if r['id'] == selected[0]]
    if 'contains' in rule:
        containing = {o.get('room_id') for o in objects if o['category'] == rule['contains']}
        pool = [r for r in pool if r['id'] in containing]
    if 'connection_count' in rule:
        count = rule['connection_count']
        if type(count) is not int or count < 0:
            return []
        pool = [r for r in pool if r.get('connectivity_known')
                and len(r['connected_room_ids']) == count]
    for category, count in rule.get('object_counts', {}).items():
        if type(count) is not int or count < 0:
            return []
        pool = [r for r in pool if sum(o['category'] == category and o.get('room_id') == r['id']
                                     for o in objects) == count]
    for category in rule.get('absent_categories', []):
        pool = [r for r in pool if not any(o['category'] == category and o.get('room_id') == r['id']
                                         for o in objects)]
    anchor_ids = []
    for reference in room_anchors(rule):
        anchors = resolve_rooms(objects, rooms, reference)
        if len(anchors) != 1 or anchors[0]['id'] in anchor_ids:
            return []
        anchor_ids.append(anchors[0]['id'])
        pool = [r for r in pool if anchors[0]['id'] in r.get('connected_room_ids', [])]
    return pool if len(pool) == 1 else []


def resolve(objects, rooms, rule, cache=None):
    if cache is not None:
        key = json.dumps(rule, sort_keys=True)
        if key not in cache:
            cache[key] = _resolve(objects, rooms, rule, cache)
        return cache[key]
    return _resolve(objects, rooms, rule, cache)


def _resolve(objects, rooms, rule, cache):
    pool = [o for o in objects if o['category'] == rule['category']]
    if rule.get('room'):
        selected = resolve_rooms(objects, rooms, rule['room'])
        if len(selected) != 1:
            return []
        pool = [o for o in pool if o.get('room_id') == selected[0]['id']]
    if rule.get('room_category'):
        matching_rooms = [r for r in rooms.values() if r['category'] == rule['room_category']]
        if len(matching_rooms) != 1:
            return []
        pool = [o for o in pool if o.get('room_id') == matching_rooms[0]['id']]
    if rule.get('room_rank'):
        sizes = [r.get('area_cells', 0) for r in rooms.values()]
        if not sizes or min(sizes) <= 0:
            return []
        best = max(sizes) if rule['room_rank'] == 'largest' else min(sizes)
        selected = [r['id'] for r in rooms.values() if r['area_cells'] == best]
        if len(selected) != 1:
            return []
        pool = [o for o in pool if o.get('room_id') == selected[0]]
    if not rule.get('anchor'):
        return pool
    anchors = resolve(objects, rooms, rule['anchor'], cache)
    if len(anchors) != 1:
        return []
    anchor = anchors[0]
    pool = [o for o in pool if o['id'] != anchor['id']]
    if rule['relation'] == 'same_room':
        return [o for o in pool if anchor.get('room_id') in rooms and o.get('room_id') == anchor['room_id']]
    if position(anchor) is None:
        return []
    relation = rule['relation']
    if relation in {'closer_than', 'between'}:
        others = resolve(objects, rooms, rule.get('other_anchor', {}), cache) if rule.get('other_anchor') else []
        if len(others) != 1 or others[0]['id'] == anchor['id'] or position(others[0]) is None:
            return []
        other = others[0]
        pool = [o for o in pool if o['id'] != other['id'] and position(o) is not None]
        a, b = position(anchor), position(other)
        if relation == 'closer_than':
            margin = rule.get('tie_margin_cells', 1.0)
            return [o for o in pool if distance(position(o), b) - distance(position(o), a) > margin]
        delta = (b[0] - a[0], b[1] - a[1])
        length2 = sum(v*v for v in delta)
        if length2 == 0:
            return []
        tolerance = rule.get('between_tolerance_cells', 1.0)
        result = []
        for obj in pool:
            point = position(obj)
            t = sum((point[i] - a[i]) * delta[i] for i in (0, 1)) / length2
            projection = (a[0] + t*delta[0], a[1] + t*delta[1])
            if 0 < t < 1 and distance(point, projection) <= tolerance:
                result.append(obj)
        return result
    if relation not in {'nearest', 'farthest'}:
        return []
    scores = [(distance(position(o), position(anchor)), o) for o in pool if position(o) is not None]
    scores.sort(key=lambda x: x[0], reverse=rule['relation'] == 'farthest')
    rank = rule.get('rank', 1)
    if not 1 <= rank <= len(scores):
        return []
    value = scores[rank - 1][0]
    return [o for d, o in scores if abs(d - value) <= rule.get('tie_margin_cells', 1.0)]


def render_room(rule, variant=0):
    text = 'the ' + noun(rule['category']) if rule.get('category') else 'the room'
    predicates = []
    if rule.get('rank'):
        if rule.get('category'):
            predicates.append('is the ' + rule['rank'] + ' room')
        else:
            text = 'the ' + rule['rank'] + ' room'
    if rule.get('contains'):
        category = noun(rule['contains'])
        article = 'an' if category[0].lower() in 'aeiou' else 'a'
        predicates.append(['contains ', 'has ', 'contains ', 'has '][variant % 4]
                          + article + ' ' + category)
    references = room_anchors(rule)
    if references:
        words = ['is directly connected to ', 'has a direct passage to ',
                 'adjoins and is directly connected to ', 'is linked by a direct passage to ']
        predicates.append(words[variant % 4] + ' and '.join(render_room(r, variant) for r in references))
    if 'connection_count' in rule:
        count = rule['connection_count']
        predicates.append(f'is directly connected to exactly {count} other ' + ('room' if count == 1 else 'rooms'))
    for category, count in sorted(rule.get('object_counts', {}).items()):
        predicates.append(f'contains exactly {count} instances of the object category "{noun(category)}"')
    for category in sorted(rule.get('absent_categories', [])):
        predicates.append(f'contains no objects of the category "{noun(category)}"')
    return text + (' that ' + ' and '.join(predicates) if predicates else '')


def render(rule, variant=0):
    text = 'the ' + noun(rule['category'])
    if rule.get('room'):
        text += ' in ' + render_room(rule['room'], variant)
    if rule.get('room_category'):
        text += ' in the ' + noun(rule['room_category'])
    if rule.get('room_rank'):
        text += ' in the ' + rule['room_rank'] + ' room'
    if rule.get('anchor'):
        anchor = render(rule['anchor'], variant)
        relation = rule['relation']
        if relation in {'closer_than', 'between'}:
            other = render(rule['other_anchor'], variant)
            if relation == 'closer_than':
                words = [' closer to ', ' that is closer to ', ' nearer to ', ' that is nearer to '][variant % 4]
                return text + words + anchor + ' than to ' + other
            words = [' between ', ' that lies between ', ' located between ', ' positioned between '][variant % 4]
            return text + words + anchor + ' and ' + other
        if relation == 'same_room':
            words = [' in the same room as ', ' in the room containing ', ' that shares a room with ', ' located in the room with '][variant % 4]
        else:
            rank = rule.get('rank', 1)
            adjective = ('closest' if variant % 2 == 0 else 'nearest') if relation == 'nearest' else ('farthest' if variant % 2 == 0 else 'furthest')
            adjective = ('second ' if rank == 2 else '') + adjective
            words = [' ', ' ', ' that is ', ' located '][variant % 4] + adjective + (' to ' if relation == 'nearest' else ' from ')
        text += words + anchor
    return text


def catalog(data, max_depth=4, per_object_layer=24, anchor_limit=2,
            enable_counting=False, enable_absence=False, pair_limit=64):
    if max_depth < 1 or per_object_layer < 1 or anchor_limit < 1 or pair_limit < 1:
        raise ValueError('Depth and search bounds must be positive')
    objects, rooms = scene(data)
    layers = {level: defaultdict(list) for level in range(1, max_depth + 1)}
    seen = set()
    categories = sorted({o['category'] for o in objects if o['category'] not in {'wall', 'window', 'doorway'}})
    rejected = 0
    cache = {}

    # Build unique room expressions first, bounded separately at every depth.
    # Base selectors allow both room identity and existential object content.
    room_layers = {level: defaultdict(list) for level in range(1, max_depth)}
    room_seen = set()
    def add_room(rule):
        key = json.dumps(rule, sort_keys=True)
        if key in room_seen:
            return
        room_seen.add(key)
        selected = resolve_rooms(objects, rooms, rule)
        if len(selected) != 1:
            return
        level, rid = room_depth(rule), selected[0]['id']
        if level in room_layers:
            group = room_layers[level][rid]
            if len(group) >= per_object_layer:
                # Preserve one multi-room selector even in a saturated room layer.
                if not isinstance(rule.get('adjacent_to'), list) or any(isinstance(r.get('adjacent_to'), list) for r in group):
                    return
                group.pop()
            group.append(rule)

    for category in sorted({r['category'] for r in rooms.values()}):
        add_room({'category': category})
    for rank in ('largest', 'smallest'):
        add_room({'rank': rank})
    for category in categories:
        add_room({'contains': category})
        for room_category in sorted({r['category'] for r in rooms.values()}):
            add_room({'category': room_category, 'contains': category})
    room_categories = sorted({r['category'] for r in rooms.values()})
    for room in rooms.values():
        selectors = []
        if room.get('connectivity_known'):
            selectors.append({'connection_count': len(room['connected_room_ids'])})
        if enable_counting:
            counts = Counter(o['category'] for o in objects if o.get('room_id') == room['id'])
            selectors.extend({'object_counts': {c: counts[c]}} for c in categories if counts[c])
        if enable_absence:
            present = {o['category'] for o in objects if o.get('room_id') == room['id']}
            selectors.extend({'absent_categories': [c]} for c in categories if c not in present)
        for selector in selectors:
            add_room(selector)
            for category in room_categories:
                add_room(dict(selector, category=category))
    for level in range(2, max_depth):
        anchors = [r for group in room_layers[level - 1].values()
                   for r in (group if level == 2 else group[:anchor_limit])]
        for anchor in anchors:
            add_room({'adjacent_to': anchor})
            for category in sorted({r['category'] for r in rooms.values()}):
                add_room({'category': category, 'adjacent_to': anchor})
            for category in categories:
                add_room({'contains': category, 'adjacent_to': anchor})
        for first, second in islice(combinations(anchors, 2), pair_limit):
            add_room({'adjacent_to': [first, second]})
            for category in room_categories:
                add_room({'category': category, 'adjacent_to': [first, second]})

    def add(rule, dependencies=()):
        nonlocal rejected
        key = json.dumps(rule, sort_keys=True)
        if key in seen:
            return
        seen.add(key)
        found = resolve(objects, rooms, rule, cache)
        if len(found) != 1 or found[0]['id'] in dependencies:
            rejected += 1
            return
        oid = found[0]['id']
        level = depth(rule)
        if level > max_depth:
            return
        if len(layers[level][oid]) >= per_object_layer:
            def family(value):
                selector = value.get('room', {})
                room_kind = ('room_multi' if isinstance(selector.get('adjacent_to'), list) else
                             'room_count' if 'connection_count' in selector else
                             'room_object_count' if 'object_counts' in selector else
                             'room_absence' if 'absent_categories' in selector else 'room')
                return value.get('relation', room_kind if selector else 'base') + ('_scoped' if selector and value.get('anchor') else '')
            counts = Counter(family(entry['rule']) for entry in layers[level][oid])
            current = family(rule)
            largest = max(counts, key=counts.get)
            if counts[current] >= counts[largest]:
                return
            def protected(value):
                selector = value.get('room', {})
                reference = selector.get('adjacent_to')
                return (not value.get('anchor') and set(selector) == {'adjacent_to'}
                        and isinstance(reference, dict) and set(reference) in ({'rank'}, {'contains'}))
            victims = [i for i, entry in enumerate(layers[level][oid])
                       if family(entry['rule']) == largest and not protected(entry['rule'])]
            if not victims:
                return
            layers[level][oid].pop(victims[-1])
        layers[level][oid].append({'object_id': oid, 'depth': level, 'rule': rule,
                                  'dependencies': list(dict.fromkeys(dependencies)) + [oid],
                                  'expressions': list(dict.fromkeys(render(rule, v) for v in range(4)))})

    for category in categories:
        add({'category': category})
    for level in range(2, max_depth + 1):
        if level == 2:
            for category in categories:
                for room in sorted({r['category'] for r in rooms.values()}):
                    add({'category': category, 'room_category': room})
                for rank in ('largest', 'smallest'):
                    add({'category': category, 'room_rank': rank})
        for selectors in room_layers[level - 1].values():
            for selector in selectors:
                # Existing room_category/room_rank already cover these bases.
                if set(selector) in ({'category'}, {'rank'}):
                    continue
                for category in categories:
                    if selector.get('contains') != category:
                        add({'category': category, 'room': selector})
        anchors = [c for candidates in layers[level-1].values() for c in candidates[:anchor_limit]]
        for anchor in anchors:
            for category in categories:
                for relation, rank in [('same_room', 1), ('nearest', 1), ('farthest', 1), ('nearest', 2), ('farthest', 2)]:
                    rule = {'category': category, 'relation': relation, 'anchor': anchor['rule']}
                    if relation != 'same_room':
                        rule.update(rank=rank, tie_margin_cells=1.0)
                    add(rule, anchor['dependencies'])
                    if relation != 'same_room':
                        for selectors in room_layers.get(1, {}).values():
                            for selector in selectors[:anchor_limit]:
                                add(dict(rule, room=selector), anchor['dependencies'])
        # Both references are shallower than the enclosing rule. Pair search is bounded.
        pair_anchors = [c for previous in range(1, level) for group in layers[previous].values()
                        for c in group[:anchor_limit]]
        eligible_pairs = ((a, b) for a, b in combinations(pair_anchors, 2)
                          if a['object_id'] != b['object_id'] and max(a['depth'], b['depth']) == level-1)
        for first, second in islice(eligible_pairs, pair_limit):
            dependencies = list(dict.fromkeys(first['dependencies'] + second['dependencies']))
            for category in categories:
                for relation in ('between', 'closer_than'):
                    directions = [(first, second)] if relation == 'between' else [(first, second), (second, first)]
                    for a, b in directions:
                        rule = {'category': category, 'relation': relation,
                                'anchor': a['rule'], 'other_anchor': b['rule']}
                        if relation == 'between':
                            rule['between_tolerance_cells'] = 1.0
                        else:
                            rule['tie_margin_cells'] = 1.0
                        add(rule, dependencies)
    return {'schema_version': 3, 'max_depth': max_depth,
            'room_connectivity_policy': 'four-connected room-labelled free-cell transition; missing evidence yields no edges',
            'enabled_features': {'counting': enable_counting, 'absence': enable_absence},
            'search_bounds': {'per_object_layer': per_object_layer, 'anchors_per_object_layer': anchor_limit, 'pairs_per_level': pair_limit},
            'rejected_ambiguous_empty_or_cyclic': rejected,
            'layers': {str(k): [c for group in v.values() for c in group] for k, v in layers.items()},
            'unresolved_object_ids': [o['id'] for o in objects if o['category'] in categories and not any(o['id'] in v for v in layers.values())]}


def difficulty(task):
    rules = [o['description_constraints'] for o in task['objects']] + [s['reference_description_constraints'] for s in task['soft_constraints']]
    maximum = max(map(depth, rules), default=1)
    visits, soft = len(task['objects']), len(task['soft_constraints'])
    label = 'hard' if maximum >= 3 or visits >= 4 else 'medium' if maximum == 2 or visits >= 2 or soft else 'easy'
    return label, {'policy': 'v1: hard depth>=3 or visits>=4; medium depth=2 or visits>=2 or soft>0; otherwise easy',
                   'max_description_depth': maximum, 'ordered_visits': visits, 'soft_constraints': soft}
