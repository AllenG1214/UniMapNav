"""Allocate grounded, non-repeating descriptions per scene/object occurrence."""
import copy
import json
from collections import Counter, defaultdict

from annotate_navigation import descriptions


def normalize(text):
    return ' '.join(text.casefold().split()).rstrip('.,;')


class DescriptionAllocator:
    def __init__(self, data, candidates=None):
        self.pools = {}
        self.used = defaultdict(set)
        self.rule_counts = defaultdict(Counter)
        self.anchor_counts = defaultdict(Counter)
        wording = {
            'same_room': (' in the same room as ', [
                ' in the same room as ', ' in the room containing ',
                ' that shares a room with ', ' located in the room with ']),
            'nearest': (' closest to ', [
                ' closest to ', ' nearest to ', ' that is closest to ',
                ' located nearest to ']),
            'farthest': (' farthest from ', [
                ' farthest from ', ' furthest from ', ' that is farthest from ',
                ' located furthest from ']),
            'left_of': (' to the left of ', [
                ' to the left of ', ' located to the left of ',
                ' that is to the left of ', ' positioned to the left of ']),
            'right_of': (' to the right of ', [
                ' to the right of ', ' located to the right of ',
                ' that is to the right of ', ' positioned to the right of ']),
            'distance_rank': (' that is ', [
                ' that is ', ' which is ', ' ranked ', ' that ranks ']),
        }
        for oid, candidates in (candidates if candidates is not None else descriptions(data, all_candidates=True)).items():
            pool = {}
            for rank, (phrase, rule) in enumerate(candidates):
                variants = [phrase]
                relation = rule.get('relation')
                if relation in wording:
                    token, replacements = wording[relation]
                    if token in phrase:
                        variants = [phrase.replace(token, replacement, 1) for replacement in replacements]
                for variant_index, text in enumerate(variants):
                    key = normalize(text)
                    if key not in pool:
                        pool[key] = (text, rule, rank, variant_index)
            self.pools[oid] = list(pool.values())

    def require_capacity(self, counts):
        missing = {oid: {'needed': n, 'available': len(self.pools.get(oid, []))}
                   for oid, n in counts.items() if len(self.pools.get(oid, [])) < n}
        if missing:
            raise ValueError(f'Insufficient unique grounded descriptions; no repeat fallback: {missing}')

    def take(self, oid):
        options = [v for v in self.pools[oid] if normalize(v[0]) not in self.used[oid]]
        if not options:
            raise ValueError(f'Description pool exhausted for object {oid}')

        def keys(rule):
            reference = {key: rule[key] for key in ('anchor', 'other_anchor', 'room', 'room_category', 'room_rank') if key in rule}
            return (json.dumps(rule, sort_keys=True), json.dumps(reference, sort_keys=True))

        def score(candidate):
            _, rule, rank, variant = candidate
            rule_key, anchor_key = keys(rule)
            # Change the reference first, then the relation, then paraphrase.
            return (self.anchor_counts[oid][anchor_key], self.rule_counts[oid][rule_key], variant, rank)

        text, rule, _, _ = min(options, key=score)
        rule_key, anchor_key = keys(rule)
        self.used[oid].add(normalize(text))
        self.rule_counts[oid][rule_key] += 1
        self.anchor_counts[oid][anchor_key] += 1
        return text, copy.deepcopy(rule)
