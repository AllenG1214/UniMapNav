"""Plot a saved reference trajectory in the map's native row/column frame."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', type=Path, required=True)
    parser.add_argument('--instruction', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label-items', action='store_true', help='Label every saved object instance with its exact name')
    parser.add_argument('--highlight-object', type=int, help='Highlight a saved object ID')
    parser.add_argument('--highlight-category', help='Highlight all instances of a category')
    args = parser.parse_args()
    data = json.loads(args.map.read_text(encoding='utf-8'))
    task = json.loads(args.instruction.read_text(encoding='utf-8'))
    path = np.asarray(task['generated_reference_trajectory'], dtype=int)
    free = np.asarray(data['layers']['occupancy']) == data['layer_legends']['occupancy']['free']['value']
    assert np.all(path >= 0) and np.all(path < np.asarray(free.shape))
    assert np.all(free[path[:, 0], path[:, 1]]), 'Trajectory intersects obstacles'
    assert np.all(np.abs(np.diff(path, axis=0)).sum(axis=1) == 1)
    assert list(path[0]) == [task['start_pose']['row'], task['start_pose']['col']]
    # Each generated leg stops at its first entry into its ordered goal region.
    boundaries = [0]
    for goal in sorted(task['hard_constraints'], key=lambda g: g['order']):
        cells = set(map(tuple, goal['cells']))
        hit = next(i for i in range(boundaries[-1], len(path)) if tuple(path[i]) in cells)
        boundaries.append(hit)
    assert boundaries[-1] == len(path) - 1
    for i, soft in enumerate(task['soft_constraints']):
        assert soft['reference_waypoint_count'] == boundaries[i + 1] - boundaries[i] + 1
    background = Image.open(args.map.with_suffix('.png'))
    assert background.size == (free.shape[1], free.shape[0])
    colors = ['#00bfff', '#ff38a5', '#ffad00']
    fig, ax = plt.subplots(figsize=(34, 24) if args.label_items else (12, 14), facecolor='white')
    ax.imshow(background, origin='upper', interpolation='nearest')
    # Distance preferences reference object footprints, not fixed near/far zones.
    references = {}
    for soft in task['soft_constraints']:
        ref = soft.get('reference_region', {})
        if ref.get('cells'):
            entry = references.setdefault(ref.get('object_id'), {'cells': ref['cells'], 'labels': []})
            scope = soft['scope']
            pref = 'near' if soft['preference_type'] == 'near_preference' else 'faraway'
            entry['labels'].append((f"{pref} {scope['from_order']+1}-{scope['to_order']+1}",
                                    colors[(scope['to_order']-1) % len(colors)]))
    for ref_index, ref in enumerate(references.values(), 1):
        region = np.asarray(ref['cells'])
        # An explicitly illustrative vicinity, not an algorithmic constraint boundary.
        padding = 1.0 / data['metadata']['map_info']['resolution']
        lo = np.maximum(region.min(axis=0) - padding, 0)
        hi = np.minimum(region.max(axis=0) + padding, np.asarray(free.shape)-1)
        ax.add_patch(Rectangle((lo[1], lo[0]), hi[1]-lo[1], hi[0]-lo[0],
                               facecolor='#38bfff', edgecolor='#38bfff', alpha=.22,
                               linewidth=2, zorder=2))
        ax.add_patch(Rectangle((lo[1], lo[0]), hi[1]-lo[1], hi[0]-lo[0],
                               fill=False, edgecolor=ref['labels'][0][1], linestyle='--',
                               linewidth=1.6, zorder=3))
        mask = np.zeros(free.shape)
        mask[region[:, 0], region[:, 1]] = 1
        overlay = np.zeros((*free.shape, 4))
        overlay[mask == 1] = [1, .72, .05, .35]
        ax.imshow(overlay, origin='upper', interpolation='nearest')
        ax.contour(mask, levels=[.5], colors=['#ffd54f'], linewidths=1.5)
        r, c = region.mean(axis=0)
        ax.text(c, r, f'R{ref_index}', ha='center', va='center', color='white', weight='bold', fontsize=10,
                bbox=dict(boxstyle='circle,pad=.2', fc='#b77d00', ec='white'), zorder=11)
        for label_index, (text, color) in enumerate(ref['labels']):
            on_left = lo[1] > 105
            text_x = lo[1]-8 if on_left else min(free.shape[1]-80, hi[1]+8)
            text_y = max(15, lo[0]+8)+label_index*17
            ax.annotate(text, xy=(lo[1]+3, lo[0]+8+label_index*15),
                        xytext=(text_x, text_y),
                        color='white', fontsize=11, weight='bold', ha='right' if on_left else 'left', va='center',
                        bbox=dict(boxstyle='round,pad=.35', fc='#17364d', ec=color, lw=2),
                        arrowprops=dict(arrowstyle='-', color=color, lw=1.6), zorder=12)
    for i, obj in enumerate(task['objects']):
        segment = path[boundaries[i]:boundaries[i+1]+1]
        color = colors[i % len(colors)]
        ax.plot(segment[:, 1], segment[:, 0], color=color, lw=2.8,
                label=f"{i+1}-{i+2}: {obj['description']}",
                path_effects=[pe.Stroke(linewidth=4.6, foreground='white'), pe.Normal()], zorder=5)
        for j in range(15, len(segment)-5, 42):
            a, b = segment[j], segment[j+5]
            ax.annotate('', xy=(b[1], b[0]), xytext=(a[1], a[0]),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2), zorder=6)
        region = np.asarray(task['hard_constraints'][i]['cells'])
        ax.scatter(region[:, 1], region[:, 0], s=5, c=color, alpha=.2, zorder=3)
        r, c = segment[-1]
        ax.scatter(c, r, s=220, c=color, edgecolors='white', linewidths=1.5, zorder=7)
        ax.text(c, r, str(i+2), ha='center', va='center', color='white', weight='bold', fontsize=10, zorder=8)
        rr, cc = obj['center']
        ax.annotate('', xy=(cc, rr), xytext=(c, r),
                    arrowprops=dict(arrowstyle='-', linestyle=':', color=color, lw=1.6), zorder=4)
    ax.scatter(path[0, 1], path[0, 0], marker='*', s=320, c='#21d582', edgecolors='white', zorder=9, label='Start')
    ax.annotate('1 START', (path[0, 1], path[0, 0]), xytext=(-12, -22), textcoords='offset points',
                color='white', weight='bold', fontsize=10,
                bbox=dict(boxstyle='round,pad=.3', facecolor='#14523c', edgecolor='none'), zorder=10)
    distance = np.abs(np.diff(path, axis=0)).sum() * data['metadata']['map_info']['resolution']
    ax.set_title(f"{args.map.stem} | Navigation {task['id']:06d}\nSaved reference trajectory: {len(path)} points / {distance:.2f} m", fontsize=17, pad=16)
    ax.set_xlabel('Column (map grid)')
    ax.set_ylabel('Row (map grid)')
    ax.set_ylim(free.shape[0]-.5, -.5)
    ax.set_xlim(-.5, free.shape[1]-.5)
    if args.highlight_category:
        highlighted = [o for o in data['object_instances'] if o['category'] == args.highlight_category]
        for obj in highlighted:
            r, c = obj.get('grounding_center_grid', obj['center_grid'])
            target = next((o['order']+1 for o in task['objects'] if o['object_id'] == obj['id']), None)
            label = obj['name'] + (f' (Target {target})' if target else '')
            ax.scatter(c, r, s=450, facecolors='none', edgecolors='#ff2020', linewidths=2.5, zorder=20)
            ax.annotate(f'{label}\nrow {r}, col {c}', (c, r),
                        xytext=(c+40, r-25 if target else r+32), fontsize=11, weight='bold', color='#b00000',
                        bbox=dict(boxstyle='round,pad=.4', fc='white', ec='#ff2020'),
                        arrowprops=dict(arrowstyle='->', color='#ff2020', lw=2), zorder=21)
    if args.highlight_object is not None:
        obj = next(o for o in data['object_instances'] if o['id'] == args.highlight_object)
        r, c = obj.get('grounding_center_grid', obj['center_grid'])
        ax.scatter(c, r, s=450, facecolors='none', edgecolors='#ff2020', linewidths=2.5, zorder=20)
        ax.annotate(f"{obj['name']}\nrow {r}, col {c}", (c, r), xytext=(c+55, r+40),
                    fontsize=12, weight='bold', color='#b00000',
                    bbox=dict(boxstyle='round,pad=.4', fc='white', ec='#ff2020'),
                    arrowprops=dict(arrowstyle='->', color='#ff2020', lw=2), zorder=21)
    if args.label_items:
        # Sorted external callouts keep all names legible, including small and structural objects.
        groups = [[], []]
        for obj in data['object_instances']:
            r, c = obj.get('grounding_center_grid', obj['center_grid'])
            groups[int(c >= free.shape[1]/2)].append((r, c, obj))
        labeled = []
        for side, objects in enumerate(groups):
            objects.sort(key=lambda item: (item[0], item[1], item[2]['id']))
            for (r, c, obj), y in zip(objects, np.linspace(.99, .01, len(objects))):
                ink = '#176287' if side == 0 else '#795121'
                ax.annotate(f"{obj['id']:03d}  {obj['name']}", xy=(c, r),
                            xytext=(-.035 if side == 0 else 1.035, y), textcoords='axes fraction',
                            ha='right' if side == 0 else 'left', va='center', fontsize=8,
                            color=ink, annotation_clip=False,
                            arrowprops=dict(arrowstyle='-', color=ink, alpha=.42, lw=.45),
                            bbox=dict(fc='white', ec='none', pad=.5, alpha=.95), zorder=15)
                ax.scatter(c, r, s=8, color=ink, edgecolors='white', linewidths=.25, zorder=14)
                labeled.append(obj['id'])
        assert len(labeled) == len(data['object_instances']) == len(set(labeled))
        ax.set_title(f"{args.map.stem} | Navigation {task['id']:06d} | All {len(labeled)} object instances\n"
                     'Exact saved names, including walls, doorways and windows; leader lines point to object centers',
                     fontsize=19, pad=25)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.06), fontsize=10, framealpha=.96)
    fig.text(.5, .018, '1: start. Targets are numbered 2, 3, 4 in visit order. Labels indicate the applicable route segment.\n'
             'near: approach the shaded reference vicinity. faraway: keep distance from that vicinity.\n'
             'Shadow: approximate reference vicinity (1 m padding for display only). Gold R: reference-object footprint.\n'
             'The planner uses continuous distance preferences; the shadow is not a hard boundary.',
             ha='center', fontsize=10, linespacing=1.5)
    if args.label_items:
        fig.subplots_adjust(left=.27, right=.73, bottom=.18, top=.88)
    else:
        fig.subplots_adjust(left=.075, right=.97, bottom=.23, top=.9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches='tight' if args.label_items else None)
    plt.close(fig)
    print(json.dumps({'output': str(args.output), 'points': len(path), 'distance_m': distance,
                      'arrival_indices': boundaries, 'arrival_cells': path[boundaries].tolist(),
                      'all_cells_free': True, 'four_connected': True}, indent=2))


if __name__ == '__main__':
    main()
