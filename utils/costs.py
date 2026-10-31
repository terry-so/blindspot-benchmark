"""Generation cost estimates from recorded usage, never billing totals."""
import math
from pathlib import Path

from utils.benchmark_runner import atomic_json, read_json


def validate_pricing(pricing):
    if pricing is None:
        return
    if not isinstance(pricing, dict) or set(pricing) != {'text_input', 'image_input', 'output'}:
        raise ValueError('pricing requires text_input, image_input, output (USD per million tokens)')
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in pricing.values()):
        raise ValueError('Pricing rates must be finite, nonnegative numbers')


def estimate_cost(usage, pricing):
    if not pricing or not isinstance(usage, dict):
        return None
    details = usage.get('input_tokens_details') or {}
    # Cached tokens need separate prices; do not silently price them as uncached.
    if details.get('cached_tokens') or any((details.get('cached_tokens_details') or {}).values()):
        return None
    counts = [details.get('text_tokens'), details.get('image_tokens'), usage.get('output_tokens')]
    if any(type(v) is not int or v < 0 for v in counts):
        return None
    if usage.get('input_tokens') is not None and sum(counts[:2]) != usage['input_tokens']:
        return None
    return round(sum(n * pricing[k] for n, k in zip(counts, ('text_input', 'image_input', 'output'))) / 1_000_000, 10)


def add_costs(result, pricing=None):
    pricing = pricing if pricing is not None else result.get('generation_pricing')
    result['generation_pricing'] = pricing
    calls = result.get('steps') if 'steps' in result else [result]
    calls = calls or []
    for call in calls:
        call['estimated_cost_usd'] = estimate_cost(call.get('usage'), pricing)
    costs = [call['estimated_cost_usd'] for call in calls]
    result['estimated_cost_usd'] = round(sum(costs), 10) if costs and all(c is not None for c in costs) else None


def write_cost_summary(result_dir):
    models = read_json(Path(result_dir).parent / 'cost_summary.json').get('models', {})
    for folder in sorted(Path(result_dir).glob('*')):
        if not folder.is_dir():
            continue
        total, priced, unknown = 0, 0, 0
        for path in folder.glob('*.json'):
            if path.name.endswith('.generation.json'):
                continue
            result = read_json(path)
            calls = result.get('steps', [result])
            for call in calls:
                if 'execution_status' not in call and 'usage' not in call:
                    continue
                cost = call.get('estimated_cost_usd')
                if cost is None:
                    unknown += 1
                else:
                    total += cost
                    priced += 1
        models[folder.name] = {'estimated_cost_usd': round(total, 10) if priced else None,
                               'priced_calls': priced, 'unpriced_calls': unknown}
    atomic_json(Path(result_dir).parent / 'cost_summary.json', {
        'scope': 'Recorded generation calls only; excludes judges, unlogged retries, and overwritten attempts. Totals include priced calls only.',
        'models': models,
    })
