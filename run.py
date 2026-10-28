"""Run API and local editors: python run.py --config configs/benchmark.yaml."""

import argparse
from functools import partial
import json
import os
from pathlib import Path
import sys
import subprocess

import yaml

from run_local import load_cases, main as run_local
from run_benchmark import run_benchmark
from utils.benchmark_runner import atomic_json, validate_inputs
from utils.run_dataset import run_cases, fingerprint


def parse_args(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--provider', choices=['azure', 'openai', 'gemini', 'vertex', 'local'])
    parser.add_argument('--model')
    parser.add_argument('--models', nargs='+', help='Names to select from the YAML model list')
    parser.add_argument('--dataset', type=Path, default=Path('data/benchmark_cases'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--levels', nargs='+', default=['L1'], choices=['L0', 'L1', 'L2a', 'L2b', 'L3-reset'])
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--cases-per-subcategory', type=int, help='First N cases by ID per subcategory; omitted uses all')
    parser.add_argument('--max-workers', type=int, default=1)
    parser.add_argument('--judge-provider', choices=['azure', 'openai'], default='azure')
    parser.add_argument('--judge-model', default=os.environ.get('JUDGE_MODEL'))
    parser.add_argument('--seed', type=int, default=42, help='Local models only')
    parser.add_argument('--cpu-offload', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--smoke-test', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--check', action='store_true')
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument('--config', type=Path)
    config_path = probe.parse_known_args(argv)[0].config
    config = {}
    if config_path:
        config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        if not isinstance(config, dict):
            parser.error('Config must be a YAML mapping')
        judge = config.pop('judge', {})
        if not isinstance(judge, dict) or set(judge) - {'provider', 'model'}:
            parser.error('judge accepts only provider and model')
        config.update({f'judge_{key}': value for key, value in judge.items()})
    entries = config.pop('models', None)
    actions = {a.dest: a for a in parser._actions}
    prefix = []
    for key, value in config.items():
        if key not in actions or key in {'config', 'help'}:
            parser.error(f'Unknown config key: {key}')
        action = actions[key]
        flag = action.option_strings[0]
        if key in {'cpu_offload', 'smoke_test', 'check'}:
            if type(value) is not bool:
                parser.error(f'{key} must be true or false')
            if value:
                prefix.append(flag)
        elif key == 'levels':
            if not isinstance(value, list) or not value or not all(isinstance(v, str) for v in value):
                parser.error('levels must be a nonempty list of strings')
            prefix.extend([flag, *value])
        else:
            if not isinstance(value, (str, int)) or isinstance(value, bool):
                parser.error(f'Invalid value for {key}')
            prefix.extend([flag, str(value)])
    args = parser.parse_args(prefix + list(argv))
    if args.cases_per_subcategory is not None and args.cases_per_subcategory < 1:
        parser.error('cases_per_subcategory must be a positive integer')
    if not args.judge_model:
        parser.error('Set judge.model, --judge-model, or JUDGE_MODEL')
    args.entries = entries
    if entries is None:
        if not args.provider or not args.model or args.models:
            parser.error('Provide provider/model, or a YAML models list')
    else:
        if args.provider or args.model or not isinstance(entries, list) or not entries:
            parser.error('Use a nonempty models list without top-level provider/model')
        names = set()
        from models.local_editors import MODEL_IDS
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) - {'name', 'provider', 'model', 'seed', 'cpu_offload'}:
                parser.error('Each model accepts name, provider, model, seed, cpu_offload')
            if not all(isinstance(entry.get(k), str) and entry[k].strip() for k in ('name', 'provider', 'model')):
                parser.error('Each model requires name, provider, and model strings')
            name = entry['name']
            if name.casefold() in names or name in {'.', '..'} or any(c in name for c in '/\\:'):
                parser.error('Model names must be unique safe folder names')
            names.add(name.casefold())
            if entry['provider'] not in {'azure', 'openai', 'gemini', 'vertex', 'local'}:
                parser.error('Unknown model provider')
            if entry['provider'] == 'local':
                if entry['model'] not in MODEL_IDS:
                    parser.error('Unknown local model')
                if type(entry.get('seed', args.seed)) is not int or type(entry.get('cpu_offload', args.cpu_offload)) is not bool:
                    parser.error('Local seed must be an integer and cpu_offload a boolean')
            elif set(entry) & {'seed', 'cpu_offload'}:
                parser.error('seed and cpu_offload are local-model settings')
        if args.models:
            if set(args.models) - {e['name'] for e in entries}:
                parser.error('Unknown name in --models selection')
            args.entries = [e for e in entries if e['name'] in args.models]
    return args


def require_env(*names):
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        raise ValueError('Set environment variables: ' + ', '.join(missing))


def main(argv=None):
    args = parse_args(argv)
    if args.entries is not None:
        return run_many(args)
    if args.provider == 'local':
        if args.max_workers != 1 or args.judge_provider != 'azure':
            raise ValueError('Local runs require max_workers: 1 and an Azure judge')
        local_args = ['--model', args.model, '--dataset', str(args.dataset),
                      '--output', str(args.output), '--judge', args.judge_model,
                      '--repeats', str(args.repeats), '--seed', str(args.seed),
                      '--levels', *args.levels]
        if args.cases_per_subcategory is not None:
            local_args.extend(['--cases-per-subcategory', str(args.cases_per_subcategory)])
        for flag in ['cpu_offload', 'smoke_test', 'check']:
            if getattr(args, flag):
                local_args.append('--' + flag.replace('_', '-'))
        return run_local(local_args)
    return run_api(args)


def run_api(args, entries=None):
    multiple = entries is not None
    entries = entries or [{'name': args.provider, 'provider': args.provider, 'model': args.model}]
    if not multiple and (args.cpu_offload or args.seed != 42):
        raise ValueError('cpu_offload and seed are local-model settings')
    required = {
        'azure': ('AZURE_OPENAI_BASE_URL', 'Azure_API_KEY'),
        'openai': ('OPENAI_API_KEY',),
        'gemini': ('GEMINI_API_KEY',),
        'vertex': ('GOOGLE_CLOUD_PROJECT',),
    }
    for entry in entries:
        require_env(*required[entry['provider']], *required[args.judge_provider])
    cases = run_cases(args.output, args.dataset, args.cases_per_subcategory, args.smoke_test)
    if args.smoke_test:
        args.repeats = 1
    # Provider is a safe result-folder label even when model IDs contain slashes.
    validate_inputs(cases, [e['name'] for e in entries], args.levels, args.repeats, args.max_workers, 0)
    from models.wrappers import AzureImageEditor, OpenAIImageEditor, GeminiImageEditor
    from models.judges import OpenAIJudge
    factories = {}
    for entry in entries:
        provider, model = entry['provider'], entry['model']
        factory = {'azure': AzureImageEditor, 'openai': OpenAIImageEditor,
                   'gemini': GeminiImageEditor, 'vertex': GeminiImageEditor}[provider]
        options = {'vertexai': provider == 'vertex'} if provider in {'gemini', 'vertex'} else {}
        factories[entry['name']] = partial(factory, model, **options)
    judge_url = (os.environ['AZURE_OPENAI_BASE_URL'] if args.judge_provider == 'azure'
                 else 'https://api.openai.com/v1')
    judge_key = os.environ['Azure_API_KEY' if args.judge_provider == 'azure' else 'OPENAI_API_KEY']
    make_judge = partial(OpenAIJudge, args.judge_model, base_url=judge_url, api_key=judge_key)
    settings = {
        'judge': {'provider': args.judge_provider, 'model': args.judge_model, 'endpoint': judge_url},
        'levels': args.levels, 'repeats': args.repeats, 'max_workers': args.max_workers,
        'dataset_sha256': fingerprint(cases),
    }
    pending = []
    for entry in entries:
        model_settings = settings | {'provider': entry['provider'], 'model': entry['model']}
        if entry['provider'] == 'azure':
            model_settings['endpoint'] = os.environ['AZURE_OPENAI_BASE_URL']
        elif entry['provider'] == 'vertex':
            model_settings.update(project=os.environ['GOOGLE_CLOUD_PROJECT'],
                                  location=os.environ.get('GOOGLE_CLOUD_LOCATION', 'global'))
        folder = args.output / 'models' / entry['name'] if multiple else args.output
        path = folder / 'settings.json'
        if path.exists():
            if json.loads(path.read_text(encoding='utf-8')) != model_settings:
                raise ValueError('Run settings or dataset changed. Use a new --output directory.')
        elif (folder.exists() and any(p.name != 'data' for p in folder.iterdir())) or (multiple and (args.output / 'results' / entry['name']).exists()):
            raise ValueError('Output directory must be empty for a new run')
        pending.append((path, model_settings))
    if args.check:
        print(f'Configuration and {len(cases)} cases validated. No API calls made; use --smoke-test to verify access.')
        return 0
    cases = run_cases(args.output, args.dataset, args.cases_per_subcategory, args.smoke_test, save=True)
    for path, model_settings in pending:
        atomic_json(path, model_settings)
    summary = run_benchmark(
        cases, model_factories=factories,
        judge_factory=make_judge, levels=args.levels, repeats=args.repeats,
        result_dir=args.output / 'results', max_workers=args.max_workers,
    )
    print(json.dumps(summary, indent=2))
    return int(bool(summary['failed'] or summary['unavailable']))


def run_many(args):
    api = [e for e in args.entries if e['provider'] != 'local']
    local = [e for e in args.entries if e['provider'] == 'local']
    if local and args.judge_provider != 'azure':
        raise ValueError('Local models require an Azure judge')
    # Check every selected configuration before starting generation.
    from copy import copy
    check_args = copy(args)
    check_args.check = True
    if api:
        run_api(check_args, api)
    commands = []
    for entry in local:
        command = [sys.executable, str(Path(__file__).with_name('run_local.py')),
                   '--model', entry['model'], '--dataset', str(args.dataset),
                   '--output', str(args.output / 'local' / entry['name']),
                   '--snapshot-root', str(args.output),
                   '--judge', args.judge_model, '--repeats', str(args.repeats),
                   '--seed', str(entry.get('seed', args.seed)), '--levels', *args.levels]
        if args.cases_per_subcategory is not None:
            command.extend(['--cases-per-subcategory', str(args.cases_per_subcategory)])
        if entry.get('cpu_offload', args.cpu_offload):
            command.append('--cpu-offload')
        if args.smoke_test:
            command.append('--smoke-test')
        if subprocess.run(command + ['--check']).returncode:
            return 1
        commands.append(command)
    if args.check:
        return 0
    run_cases(args.output, args.dataset, args.cases_per_subcategory, args.smoke_test, save=True)
    failed = run_api(args, api) if api else 0
    for command in commands:
        failed |= int(subprocess.run(command).returncode != 0)
    return failed


if __name__ == '__main__':
    sys.exit(main())
