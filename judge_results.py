"""Judge saved edits in a new output folder; never run image generation."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
import os
from pathlib import Path
import shutil

from PIL import Image
from evaluators.auto_grader import grade_output
from utils.benchmark_runner import atomic_json, valid_image


def create_judge(model, provider):
    """Use the same judge configuration for L3 checks and saved-result scoring."""
    from models.judges import OpenAIJudge
    judge_class = OpenAIJudge
    if provider == 'azure' and model.lower().startswith('claude-'):
        from models.judges import ClaudeJudge
        judge_class = ClaudeJudge
    url = os.environ['AZURE_OPENAI_BASE_URL'] if provider == 'azure' else 'https://api.openai.com/v1'
    key = os.environ['Azure_API_KEY' if provider == 'azure' else 'OPENAI_API_KEY']
    return judge_class(model, base_url=url, api_key=key)


def result_paths(source, models=None):
    source = Path(source)
    paths = [p for p in sorted(source.glob('**/results/*/*.json'))
             if not p.name.endswith('.generation.json')]
    if models:
        selected = []
        for path in paths:
            parts = path.relative_to(source).parts
            # Multi-model local runs use local/ALIAS/results/MODEL/file.json.
            name = parts[1] if parts[0] == 'local' else path.parent.name
            if name in models:
                selected.append(path)
        paths = selected
    if not paths:
        raise ValueError('No saved results found for the selected models')
    return paths


def judge_results(source, destination, judge, models=None, *, refusal_judge=None, realism_judge=None, max_workers=1, prompt_templates=None, metadata=None, resume=False, skip_refusal=False):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or (destination.exists() and not resume):
        raise ValueError("Choose a new output folder outside the source run")
    paths = result_paths(source, models)
    if resume:
        saved_metadata = json.loads((destination / "judging.json").read_text(encoding="utf-8"))
        if any(saved_metadata.get(k) != v for k, v in metadata.items()):
            raise ValueError("Judge settings or prompts changed; use a new output folder")
    else:
        shutil.copytree(source, destination)
    if metadata is not None:
        atomic_json(destination / "judging.json", metadata)

    def process(original):
        path = destination / original.relative_to(source)
        result = json.loads(path.read_text(encoding='utf-8'))
        if resume and result.get('status') == 'ok':
            return False
        result = json.loads(original.read_text(encoding='utf-8'))
        level = result['level']
        image_path = path.with_suffix('.png')
        result['output_path'] = str(image_path)
        try:
            if result.get('technical_error') or (result.get('status') == 'error' and not valid_image(image_path)):
                raise RuntimeError(result.get('error') or 'Generation failed')
            # Preserve decisions that determined whether an L3 chain continued.
            refusal = 0 if skip_refusal else (result.get('r') if level == 'L3-reset' else None)
            if level == 'L3-reset' and refusal != 1:
                steps = result.get('steps', [])
                if result.get('failed_step') or len(steps) != result.get('chain_length') or not steps:
                    raise RuntimeError('L3 chain is incomplete; generation is required')
            case_file = next((parent / 'data' / result['case_id'] / 'case.json'
                              for parent in path.parents
                              if (parent / 'data' / result['case_id'] / 'case.json').is_file()), None)
            if case_file is None:
                raise ValueError('Saved source case not found')
            case = json.loads(case_file.read_text(encoding='utf-8'))
            case['image_path'] = str(case_file.with_name('input.png'))
            previous_image = case['image_path']
            for step in result.get('steps', []):
                step['input_path'] = previous_image
                if step.get('output_path'):
                    step['output_path'] = str(path.parent / (path.stem + '_steps') / f"step_{step['step']}.png")
                    previous_image = step['output_path']
            image = None
            if image_path.exists():
                with Image.open(image_path) as saved_image:
                    image = saved_image.copy()
            elif result.get('execution_status') == 'success' and refusal != 1 and result.get('image_returned') is not False:
                raise ValueError('Generated image is missing')
            output = {'status': result.get('execution_status', 'failed'), 'image': image,
                      'text_response': result.get('text_response'),
                      'error': result.get('execution_error'),
                      'safety_block': result.get('safety_block', False)}
            prompt = '\n'.join(case['prompts']['L3']) if level == 'L3-reset' else case['prompts'][level]
            labels = grade_output(judge, case, prompt, output, str(image_path), 0,
                                  refusal_score=refusal, level=level, refusal_judge=refusal_judge,
                                  realism_judge=realism_judge, prompt_templates=prompt_templates)
            result.update(labels)
            result.pop('error', None)
        except Exception as error:
            result.update(status='error', r=result.get('r') if level == 'L3-reset' else None,
                          e=None, q=None, error=str(error))
        atomic_json(path, result)
        return result["status"] == "error"

    failed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(process, path) for path in paths]
        for index, future in enumerate(as_completed(futures), 1):
            failed += future.result()
            print(f"{index}/{len(paths)} judged", flush=True)
    return failed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--judge-model', default=os.environ.get('JUDGE_MODEL'))
    for metric in ('refusal', 'fidelity', 'realism'):
        parser.add_argument(f'--{metric}-judge-model', help='Override --judge-model for this metric')
    parser.add_argument('--judge-provider', choices=['azure', 'openai'], default='azure')
    parser.add_argument('--models', nargs='+', help='Saved model folder names to judge')
    parser.add_argument('--metrics', nargs='+', choices=['refusal', 'fidelity', 'realism'], help='Regrade selected metrics independently into reports')
    parser.add_argument('--labels', type=Path, help='Human labels CSV; selects metric-comparable cases')
    parser.add_argument('--disagreements-only', action='store_true', help='Select only old disagreements with --labels')
    parser.add_argument('--check', action='store_true', help='Validate saved results or metric selection without API calls')
    parser.add_argument('--max-workers', type=int, default=1, help='Concurrent regrading requests (default: 1)')
    parser.add_argument('--skip-refusal', action='store_true', help='Score local outputs for fidelity and realism only')
    parser.add_argument('--resume', action='store_true', help='Reuse successful scores; retry failed and unfinished cases')
    args = parser.parse_args(argv)
    if args.max_workers < 1:
        parser.error('--max-workers must be a positive integer')
    if args.metrics and any(getattr(args, f'{m}_judge_model') for m in ('refusal', 'fidelity', 'realism')):
        parser.error('For metric calibration, use --judge-model for the selected metrics')
    if (args.labels or args.disagreements_only) and not args.metrics:
        parser.error('--labels and --disagreements-only require --metrics')
    if args.models and args.metrics:
        parser.error('--models is for full judging; use --labels to select regrades')
    if not args.metrics:
        if (args.output.exists() and not args.resume) or args.output.resolve() == args.run.resolve() or args.run.resolve() in args.output.resolve().parents:
            parser.error('Choose a new output folder outside the source run')
        paths = result_paths(args.run, args.models)
        if args.check:
            print(f'{len(paths)} saved results selected. No API calls made.')
            return 0
    if args.metrics:
        from evaluators.regrade import select_cases, regrade, load_regrade_progress, prompt_templates
        if len(set(args.metrics)) != len(args.metrics):
            parser.error('--metrics must not contain duplicates')
        if (args.output.exists() and not args.resume) or args.output.resolve() == args.run.resolve() or args.run.resolve() in args.output.resolve().parents:
            parser.error('Choose a new output folder outside the source run')
        selections = {}
        for metric in args.metrics:
            selected, selection = select_cases(args.run, args.labels, args.disagreements_only, metric)
            print(json.dumps(selection, indent=2))
            if not selected:
                parser.error(f'No eligible {metric} cases selected')
            selections[metric] = (selected, selection)
            if args.resume and (args.output / metric).exists():
                records = load_regrade_progress(args.output / metric, selected, selection,
                          {'model': args.judge_model, 'provider': args.judge_provider}, prompt_templates(selection['metric']))
                print(f"{metric}: {sum(r is not None for r in records)} successful scores can be reused.")
        calibration = {'metrics': args.metrics, 'judge': {'model': args.judge_model, 'provider': args.judge_provider},
                       'selections': {m: selection for m, (_, selection) in selections.items()},
                       'prompts': {m: prompt_templates(selections[m][1].get('metric', m)) for m in args.metrics}}
        if args.resume:
            saved = json.loads((args.output / 'calibration.json').read_text(encoding='utf-8'))
            if saved != calibration:
                parser.error('Calibration settings or prompts changed. Use a new output folder.')
        if args.check:
            print('Selections validated. No API calls made.')
            return 0
    if not args.judge_model:
        parser.error('Set --judge-model or JUDGE_MODEL')
    judge = create_judge(args.judge_model, args.judge_provider)
    if args.metrics:
        atomic_json(args.output / 'calibration.json', calibration)
        failed = 0
        for metric, (selected, selection) in selections.items():
            report = regrade(selected, args.output / metric, judge, selection,
                             {'model': args.judge_model, 'provider': args.judge_provider},
                             max_workers=args.max_workers, resume=args.resume and (args.output / metric).exists())
            print(json.dumps({k: v for k, v in report.items() if not k.startswith('by_')}, indent=2))
            print(f'Reports: {args.output / metric}')
            failed += report['failed']
        return int(bool(failed))
    models = {m: getattr(args, f'{m}_judge_model') or args.judge_model
              for m in ('refusal', 'fidelity', 'realism')}
    clients = {args.judge_model: judge}
    for model in set(models.values()):
        if model not in clients:
            clients[model] = create_judge(model, args.judge_provider)
    prompts = {p.name: p.read_text(encoding='utf-8') for p in Path(__file__).with_name('prompts').glob('*judge_prompt.txt')}
    metadata = {'models': models, 'provider': args.judge_provider,
                'source': str(args.run.resolve()), 'realism_scale': '1-5',
                'l3_refusal': 'Preserved from generation', 'prompts': prompts}
    if args.skip_refusal:
        metadata['skip_refusal'] = True
    failed = judge_results(args.run, args.output, clients[models['fidelity']], models=args.models,
                           refusal_judge=clients[models['refusal']], realism_judge=clients[models['realism']],
                           max_workers=args.max_workers, prompt_templates=prompts, metadata=metadata,
                           resume=args.resume, skip_refusal=args.skip_refusal)
    atomic_json(args.output / 'judging.json', dict(metadata, failed=failed))
    return int(bool(failed))


if __name__ == '__main__':
    raise SystemExit(main())
