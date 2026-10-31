"""Regrade saved metrics independently and compare paired human labels."""
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

from evaluators.auto_grader import check_fidelity, check_realism, detect_refusal
from utils.benchmark_runner import atomic_json, valid_image


def select_cases(source, labels=None, disagreements_only=False, metric="fidelity"):
    score_key, human_column, allowed = METRICS[metric]
    source = Path(source).resolve()
    if disagreements_only and labels is None:
        raise ValueError('--disagreements-only requires --labels')
    if labels is not None:
        with Path(labels).open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            required = {'output_file', human_column, 'case_id', 'model', 'level'}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError('Labels CSV is missing required columns')
            rows = list(reader)
    else:
        rows = [{'output_file': str(p.relative_to(source).with_suffix('.png'))}
                for p in sorted(source.glob('**/results/*/*.json'))
                if not p.name.endswith('.generation.json')]
    selected, comparable, seen = [], [], set()
    missing_human = missing_ai = 0
    for row in rows:
        path = (source / row['output_file'].replace('\\', '/')).with_suffix('.json').resolve()
        if not path.is_relative_to(source):
            raise ValueError('CSV output_file must point inside the source run')
        if path in seen:
            raise ValueError(f'Duplicate label for {path.name}')
        seen.add(path)
        result = json.loads(path.read_text(encoding='utf-8'))
        if labels is not None and any(row[k] != str(result[k]) for k in ('case_id', 'model', 'level')):
            raise ValueError(f'CSV does not match saved result: {path.name}')
        human = row.get(human_column, '').strip()
        if human not in ('', *(str(v) for v in allowed)):
            raise ValueError(f'Invalid human {metric} label: {human!r}')
        if labels is not None and human == '':
            missing_human += 1
            continue
        if result.get('status') != 'ok' or result.get(score_key) not in allowed:
            missing_ai += 1
            continue
        entry = {k: result[k] for k in ('case_id', 'model', 'level', 'subcategory')}
        entry.update(output_file=row['output_file'], result_json=str(path),
                     human_score=int(human) if human else None, old_score=result[score_key])
        comparable.append(entry)
        if not disagreements_only or entry['human_score'] != result[score_key]:
            selected.append(entry)
    # Validate every selected input before spending any API calls.
    for entry in selected:
        path = Path(entry['result_json'])
        case_path = next((parent / 'data' / entry['case_id'] / 'case.json'
                          for parent in path.parents
                          if (parent == source or parent.is_relative_to(source))
                          and (parent / 'data' / entry['case_id'] / 'case.json').is_file()), None)
        if case_path is None:
            raise ValueError(f"Saved case missing: {entry['case_id']}")
        case = json.loads(case_path.read_text(encoding='utf-8'))
        result = json.loads(path.read_text(encoding='utf-8'))
        original, edited = case_path.with_name('input.png'), path.with_suffix('.png')
        if metric == 'fidelity':
            target = case['prompts'].get('L0' if entry['level'] == 'L0' else 'L2b')
            if not isinstance(target, str) or not target.strip():
                raise ValueError(f"Missing fidelity target: {entry['case_id']}")
            if not valid_image(original):
                raise ValueError(f'Missing or unreadable image: {original}')
            entry.update(original_image=str(original), target=target)
        if metric in ('fidelity', 'realism'):
            if not valid_image(edited):
                raise ValueError(f'Missing or unreadable image: {edited}')
            entry['edited_image'] = str(edited)
        else:
            responses = []
            if entry['level'] == 'L3-reset':
                steps = result.get('steps')
                if not steps:
                    raise ValueError(f'Missing saved L3 responses: {path.name}')
            else:
                steps = [result]
            for step in steps:
                if step.get('execution_status') not in ('success', 'failed'):
                    raise ValueError(f'Missing execution metadata: {path.name}')
                image = (path.parent / (path.stem + '_steps') / f"step_{step['step']}.png"
                         if entry['level'] == 'L3-reset' else edited)
                has_image = valid_image(image)
                expected_image = bool(step.get('output_path')) if entry['level'] == 'L3-reset' else (result.get('image_returned') is True or result.get('r') == 0)
                if expected_image and not has_image:
                    raise ValueError(f'Missing or unreadable image: {image}')
                responses.append({'status': step['execution_status'], 'image': True if has_image else None,
                                  'text_response': step.get('text_response'), 'error': step.get('execution_error'),
                                  'safety_block': step.get('safety_block', False),
                                  'technical_error': step.get('technical_error', False),
                                  'instruction': step.get('instruction') or (case['prompts']['L3'][step['step']-1] if entry['level'] == 'L3-reset' else case['prompts'][entry['level']])})
            entry['responses'] = responses
    return selected, {
        'source_run': str(source), 'labels_csv': str(Path(labels).resolve()) if labels else None,
        'metric': metric, 'input_rows': len(rows), 'missing_human': missing_human,
        'missing_ai': missing_ai, 'comparable': len(comparable),
        'selected': len(selected), 'disagreements_only': disagreements_only,
        'baseline_all_labeled': agreement(comparable, 'old_score', metric),
    }


METRICS = {
    'refusal': ('r', 'human_refused', (0, 1)),
    'fidelity': ('e', 'human_fidelity', (0, 1)),
    'realism': ('q', 'human_realism', (1, 2, 3, 4, 5)),
}


def agreement(records, score_key, metric='fidelity'):
    allowed = METRICS[metric][2]
    pairs = [(r['human_score'], r.get(score_key)) for r in records
             if r.get('human_score') in allowed and r.get(score_key) in allowed]
    n = len(pairs)
    report = {'n': n, 'agreement': None, 'kappa': None}
    if n:
        exact = sum(h == a for h, a in pairs) / n
        expected = sum(sum(h == v for h, a in pairs) * sum(a == v for h, a in pairs)
                       for v in allowed) / n**2
        report.update(agreement=exact, kappa=(exact-expected)/(1-expected) if expected < 1 else None)
    if metric == 'realism':
        observed = sum((h-a)**2 for h, a in pairs) / n if n else 0
        expected = sum((h-a)**2 for h, _ in pairs for _, a in pairs) / n**2 if n else 0
        report.update(within_one=sum(abs(h-a) <= 1 for h, a in pairs)/n if n else None,
                      quadratic_weighted_kappa=1-observed/expected if expected else None,
                      mean_absolute_difference=sum(abs(h-a) for h, a in pairs)/n if n else None,
                      mean_ai_minus_human=sum(a-h for h, a in pairs)/n if n else None,
                      score_differences={str(d): sum(a-h == d for h, a in pairs) for d in range(-4, 5)})
    else:
        report.update(false_positives=sum(h == 0 and a == 1 for h, a in pairs),
                      false_negatives=sum(h == 1 and a == 0 for h, a in pairs))
    return report


def prompt_templates(metric):
    names = ('error_judge_prompt.txt', 'text_response_judge_prompt.txt') if metric == 'refusal' else (metric + '_judge_prompt.txt',)
    root = Path(__file__).resolve().parents[1] / 'prompts'
    return {name: (root / name).read_text(encoding='utf-8') for name in names}


def load_regrade_progress(destination, selected, selection, judge_info, templates):
    """Validate a resume and restore successful scores without mixing experiments."""
    destination = Path(destination)
    metadata = json.loads((destination / 'judging.json').read_text(encoding='utf-8'))
    if (metadata.get('metric') != selection['metric'] or metadata.get('selection') != selection
            or metadata.get('judge') != (judge_info or {})
            or any((destination / name).read_text(encoding='utf-8') != text for name, text in templates.items())
            or ('cases' in metadata and metadata['cases'] != selected)):
        raise ValueError('Regrade settings, cases, or prompt changed. Use a new output folder.')
    paths = [p for p in (destination / 'scores.json', destination / 'scores.recovery.json') if p.exists()]
    saved = json.loads(max(paths, key=lambda p: p.stat().st_mtime_ns).read_text(encoding='utf-8')) if paths else []
    if not isinstance(saved, list):
        raise ValueError('Invalid regrade checkpoint')
    indexes = {entry['result_json']: i for i, entry in enumerate(selected)}
    records, seen = [None] * len(selected), set()
    for record in saved:
        key = record.get('result_json')
        if key not in indexes or key in seen:
            raise ValueError('Checkpoint has an unknown or duplicate case')
        seen.add(key)
        index = indexes[key]
        if any(record.get(k) != v for k, v in selected[index].items()):
            raise ValueError('Checkpoint case changed. Use a new output folder.')
        if type(record.get('new_score')) is int and record['new_score'] in METRICS[selection['metric']][2] and not record.get('error'):
            records[index] = record
    return records


def regrade(selected, destination, judge, selection, judge_info=None, *, max_workers=1, resume=False):
    """Write a separate score report; source results and historical L3 decisions stay intact."""
    destination = Path(destination)
    if not selected:
        raise ValueError('No eligible cases selected')
    if type(max_workers) is not int or max_workers < 1:
        raise ValueError('max_workers must be a positive integer')
    metric = selection['metric']
    allowed = METRICS[metric][2]
    templates = prompt_templates(metric)
    if resume:
        records = load_regrade_progress(destination, selected, selection, judge_info, templates)
    else:
        destination.mkdir(parents=True, exist_ok=False)
        records = [None] * len(selected)
        for name, text in templates.items():
            (destination / name).write_text(text, encoding='utf-8')
        atomic_json(destination / 'judging.json', {'metric': metric, 'scope': 'Independent metric; L3 refusal covers observed steps only',
                    'selection': selection, 'judge': judge_info or {}, 'max_workers': max_workers, 'cases': selected})

    def grade(entry):
        record = dict(entry, new_score=None, error=None)
        try:
            if metric == 'fidelity':
                score = check_fidelity(judge, entry['original_image'], entry['edited_image'],
                                       entry['target'], prompt_template=templates['fidelity_judge_prompt.txt'])
            elif metric == 'realism':
                score = check_realism(judge, entry['edited_image'], prompt_template=templates['realism_judge_prompt.txt'])
            else:
                score = int(any(detect_refusal(judge, '', response, 0, response['instruction'],
                                              prompt_templates=templates) for response in entry['responses']))
            if type(score) is not int or score not in allowed:
                raise ValueError(f'Invalid {metric} score: {score!r}')
            record['new_score'] = score
        except Exception as error:
            record['error'] = str(error)
        return record

    completed = sum(r is not None for r in records)
    if resume:
        print(f'Resuming: {completed}/{len(selected)} successful scores already saved', flush=True)
    checkpoint_blocked = False
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(grade, entry): index for index, entry in enumerate(selected) if records[index] is None}
        for future in as_completed(futures):
            if future.cancelled():
                continue
            records[futures[future]] = future.result()
            completed += 1
            # Only the main thread writes checkpoints; preserve selection order.
            saved = [r for r in records if r is not None]
            if not checkpoint_blocked:
                try:
                    atomic_json(destination / 'scores.json', saved)
                except PermissionError:
                    checkpoint_blocked = True
                    for pending in futures:
                        pending.cancel()
                    print('scores.json is locked. Saving recovery scores and stopping queued requests.', flush=True)
            if checkpoint_blocked:
                atomic_json(destination / 'scores.recovery.json', saved)
            print(f'{completed}/{len(selected)} {metric} scores processed', flush=True)
    if checkpoint_blocked:
        raise RuntimeError('Checkpoint locked. Completed calls saved in scores.recovery.json. Close file viewers and rerun with --resume.')
    atomic_json(destination / 'scores.json', records)
    paired = [r for r in records if r['new_score'] in allowed and r['human_score'] in allowed]
    report = {'selection': selection, 'successful': sum(r['error'] is None for r in records),
              'failed': sum(r['error'] is not None for r in records),
              'comparison_scope': 'Successfully regraded selected cases only; identical pairs before and after',
              'before': agreement(paired, 'old_score', metric),
              'after': agreement(paired, 'new_score', metric),
              'disagreements_resolved': sum(r['old_score'] != r['human_score'] == r['new_score'] for r in paired),
              'agreements_broken': sum(r['old_score'] == r['human_score'] != r['new_score'] for r in paired)}
    for group in ('model', 'level', 'subcategory'):
        report['by_' + group] = {
            value: {'before': agreement([r for r in paired if r[group] == value], 'old_score', metric),
                    'after': agreement([r for r in paired if r[group] == value], 'new_score', metric)}
            for value in sorted({r[group] for r in paired})}
    atomic_json(destination / 'agreement.json', report)
    with (destination / 'scores.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return report
