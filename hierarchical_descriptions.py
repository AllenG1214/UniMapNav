"""Bounded bottom-up referring expressions; semantic depth is independent of wording."""
import json
import math
from collections import Counter, defaultdict

def noun(category):
    return category.replace('_', ' ')


def position(obj):
    value = obj.get('center_grid') or obj.get('metadata_grid_cell')
    return tuple(map(float, value)) if value and len(value) == 2 else None


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])



def room_depth(rule):
    return 1 + room_depth(rule['adjacent_to']) if rule.get('adjacent_to') else 1


def depth(rule):
    levels = [1]
    if rule.get('anchor'):
        levels.append(1 + depth(rule['anchor']))
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
    return objects, rooms


def resolve_rooms(objects, rooms, rule):
    """Every explicit room reference must denote exactly one room.

    `contains` is existential: multiple beds in one room are acceptable, but
    beds in multiple rooms do not identify 'the room that contains a bed'.
    """
    if not rule or set(rule) - {'category', 'rank', 'contains', 'adjacent_to'}:
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
    if 'adjacent_to' in rule:
        anchors = resolve_rooms(objects, rooms, rule['adjacent_to'])
        if len(anchors) != 1:
            return []
        anchor = anchors[0]
        pool = [r for r in pool if r['id'] in anchor.get('connected_room_ids', [])]
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
    if rule.get('adjacent_to'):
        words = ['is directly connected to ', 'has a direct passage to ',
                 'adjoins and is directly connected to ', 'is linked by a direct passage to ']
        predicates.append(words[variant % 4] + render_room(rule['adjacent_to'], variant))
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
        if relation == 'same_room':
            words = [' in the same room as ', ' in the room containing ', ' that shares a room with ', ' located in the room with '][variant % 4]
        else:
            rank = rule.get('rank', 1)
            adjective = ('closest' if variant % 2 == 0 else 'nearest') if relation == 'nearest' else ('farthest' if variant % 2 == 0 else 'furthest')
            adjective = ('second ' if rank == 2 else '') + adjective
            words = [' ', ' ', ' that is ', ' located '][variant % 4] + adjective + (' to ' if relation == 'nearest' else ' from ')
        text += words + anchor
    return text


def catalog(data, max_depth=4, per_object_layer=24, anchor_limit=2):
    if max_depth < 1 or per_object_layer < 1 or anchor_limit < 1:
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
        if level in room_layers and len(room_layers[level][rid]) < per_object_layer:
            room_layers[level][rid].append(rule)

    for category in sorted({r['category'] for r in rooms.values()}):
        add_room({'category': category})
    for rank in ('largest', 'smallest'):
        add_room({'rank': rank})
    for category in categories:
        add_room({'contains': category})
        for room_category in sorted({r['category'] for r in rooms.values()}):
            add_room({'category': room_category, 'contains': category})
    for level in range(2, max_depth):
        anchors = [r for group in room_layers[level - 1].values()
                   for r in (group if level == 2 else group[:anchor_limit])]
        for anchor in anchors:
            add_room({'adjacent_to': anchor})
            for category in sorted({r['category'] for r in rooms.values()}):
                add_room({'category': category, 'adjacent_to': anchor})
            for category in categories:
                add_room({'contains': category, 'adjacent_to': anchor})

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
            return
        layers[level][oid].append({'object_id': oid, 'depth': level, 'rule': rule,
                                  'dependencies': list(dependencies) + [oid],
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
    return {'schema_version': 2, 'max_depth': max_depth,
            'room_connectivity_policy': 'four-connected room-labelled free-cell transition; missing evidence yields no edges',
            'search_bounds': {'per_object_layer': per_object_layer, 'anchors_per_object_layer': anchor_limit},
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
