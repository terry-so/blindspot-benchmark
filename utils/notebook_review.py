"""Small display helpers for saved CLI runs; no model calls."""
import json
from pathlib import Path


def show_edit(run_dir, index=0):
    import matplotlib.pyplot as plt
    from PIL import Image

    root = Path(run_dir)
    paths = sorted(root.glob('results/*/*.json')) + sorted(root.glob('local/*/results/*/*.json'))
    paths = [p for p in paths if p.with_suffix('.png').exists()]
    if not paths:
        print('No edited images saved yet (generation may have failed or been refused).')
        return
    path = paths[index]
    result = json.loads(path.read_text(encoding='utf-8'))
    before = root / 'data' / result['case_id'] / 'input.png'
    case = json.loads(before.with_name('case.json').read_text(encoding='utf-8'))
    level = result['level']
    print(f"{index + 1}/{len(paths)} · {result['model']} · {result['case_id']} · {level}")
    print('Prompt:', case['prompts']['L3' if level == 'L3-reset' else level])
    print('Scores:', {key: result.get(key) for key in ('status', 'r', 'e', 'q')})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, image_path, title in zip(axes, (before, path.with_suffix('.png')), ('Before', 'After')):
        with Image.open(image_path) as image:
            ax.imshow(image.copy())
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def analyse_run(run_dir):
    from IPython.display import display
    from utils.export_table import make_analysis_tables
    from utils.plot_results import make_plots

    root = Path(run_dir)
    cases = json.loads((root / 'data/dataset.json').read_text(encoding='utf-8'))
    groups = []
    if (root / 'results').exists():
        settings_paths = sorted(root.glob('models/*/settings.json'))
        settings_paths = settings_paths or [root / 'settings.json']
        settings = [json.loads(p.read_text(encoding='utf-8')) for p in settings_paths]
        if any((s['levels'], s['repeats']) != (settings[0]['levels'], settings[0]['repeats']) for s in settings):
            raise ValueError('API models have different levels/repeats; analyse these runs separately.')
        groups.append((root, settings[0]))
    for path in sorted(root.glob('local/*/settings.json')):
        groups.append((path.parent, json.loads(path.read_text(encoding='utf-8'))))
    for folder, settings in groups:
        levels = [level for level in settings['levels'] if level != 'L0']
        if not levels or not list((folder / 'results').glob('*/*.json')):
            print(f'{folder.name}: no harmful-level results to analyse.')
            continue
        print(folder)
        tables = make_analysis_tables(folder / 'results', folder / 'tables',
                                      levels=levels, repeats=settings['repeats'], cases=cases)
        display(tables['overall'], tables['paired_levels'], tables['l3_summary'])
        make_plots(tables, folder / 'plots')
