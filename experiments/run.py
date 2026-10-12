"""Run with `python -m experiments.run <stage> --help`."""
import argparse
from collections import Counter
from pathlib import Path
import random
from .data import (ROOT, SUBSET, IMAGES, classify, evaluate, load_json, normalize,
                   prepare, read_csv, save_json, write_csv)


def samples_from(path):
    samples = load_json(path)
    if not samples:
        raise ValueError(f'Empty subset: {path}. Check baseline scoring before starting downstream stages.')
    ids = [str(s['question_id']) for s in samples]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate question IDs in subset')
    for s in samples:
        if not s.get('answer') or len(s.get('all_answers', [])) != 10:
            raise ValueError(f"Invalid annotation: {s['question_id']}")
        if s.get('category') not in ('yes_no', 'count', 'color', 'object', 'spatial'):
            raise ValueError(f"Invalid category: {s['question_id']}")
    return samples


def backend(args):
    import numpy as np
    import torch
    from .model import LlavaBackend
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    return LlavaBackend(args.model, args.quantization, not args.allow_download)


def record_config(args, samples):
    import hashlib
    import json
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config['protocol_version'] = 'llava-vr-1'
    config['sample_ids'] = [s['question_id'] for s in samples]
    config['subset_sha256'] = hashlib.sha256(json.dumps(samples, sort_keys=True).encode()).hexdigest()
    path = args.output / f'{args.stage}_config.json'
    # Refuse to mix results from different configurations on resume.
    if path.exists() and load_json(path) != config:
        raise ValueError(f'Configuration changed; select a fresh --output directory: {path}')
    save_json(path, config)


def evaluate_stage(args, samples=None):
    from .analysis import summarize
    samples = samples_from(args.subset) if samples is None else samples
    if args.reclassify:
        for s in samples:
            s['category_original'] = s['category']
            s['category'] = classify(s['question'], s.get('answer_type', '')) or 'other'
    rows = evaluate(samples, read_csv(args.predictions), args.correct_threshold)
    write_csv(args.output / 'baseline_evaluated.csv', rows)
    selected = {str(r['question_id']) for r in rows if r['correct']}
    save_json(args.output / 'correct_subset.json', [s for s in samples if str(s['question_id']) in selected])
    save_json(args.output / 'baseline_summary.json', dict(
        metrics=summarize(rows, ['soft_accuracy', 'exact_match'], args.seed),
        total_expected=len(samples), evaluated=len(rows), missing=len(samples) - len(rows),
        correct=len(selected), threshold=args.correct_threshold,
        reclassified=args.reclassify, predictions=str(args.predictions)))
    print(f'Evaluated {len(rows)}/{len(samples)}; S_correct={len(selected)}', flush=True)


def baseline_stage(args):
    samples = samples_from(args.subset)
    if args.limit:
        samples = samples[:args.limit]
    record_config(args, samples)
    path = args.output / 'baseline_predictions.csv'
    predictions = read_csv(path) if path.exists() else []
    done = {str(s['question_id']) for s in predictions}
    model = backend(args)
    for i, s in enumerate(samples):
        if str(s['question_id']) in done:
            continue
        image = model.image(args.images / s['image_filename'])
        pred = model.answer(image, s['question'])
        predictions.append(dict(question_id=s['question_id'], image_id=s['image_id'],
                                category=s['category'], question=s['question'], gt_answer=s['answer'],
                                predicted_answer=pred))
        write_csv(path, predictions)
        print(f"[{i + 1}/{len(samples)}] {s['question_id']}: {pred}", flush=True)
    args.predictions = path
    args.reclassify = False
    evaluate_stage(args, samples)


def intervention_stage(args):
    import numpy as np
    from .analysis import correlation, plot_maps, summarize
    from .model import attention_grid, grid_boxes, occlude
    samples = samples_from(args.subset)
    if args.limit:
        samples = samples[:args.limit]
    if args.top_k < 1 or args.top_k > args.grid ** 2:
        raise ValueError('top-k must be between 1 and grid squared')
    record_config(args, samples)
    model = backend(args)
    (args.output / 'maps').mkdir(exist_ok=True)
    rows = []
    for i, s in enumerate(samples):
        path = args.output / 'maps' / f"{s['question_id']}.json"
        if path.exists():
            saved = load_json(path)
            if args.plots and not path.with_suffix('.png').exists():
                plot_maps(model.image(args.images / s['image_filename']), saved['causal'], saved['attention'], args.grid, path.with_suffix('.png'))
            rows.append(saved['metrics'])
            continue
        image = model.image(args.images / s['image_filename'])
        boxes = grid_boxes(*image.size, args.grid)
        base = model.measure(image, s['question'], s['answer'], attention=args.attention)
        masked_logp = [model.measure(occlude(image, boxes, [r], args.fill), s['question'], s['answer'])['log_probability']
                       for r in range(len(boxes))]
        causal = np.asarray([base['log_probability'] - p for p in masked_logp])
        # Fixed budget, signed scores; negative effects are retained rather than clipped.
        top = np.argsort(-causal, kind='stable')[:args.top_k].tolist()
        top_logp = model.measure(occlude(image, boxes, top, args.fill), s['question'], s['answer'])['log_probability']
        rng = np.random.default_rng(args.seed + int(s['question_id']))
        controls = [rng.choice(len(boxes), size=args.top_k, replace=False).tolist() for _ in range(args.random_controls)]
        control_drops = [base['log_probability'] - model.measure(occlude(image, boxes, indices, args.fill), s['question'], s['answer'])['log_probability']
                         for indices in controls]
        attn = attention_grid(base['attention'], args.grid) if args.attention else None
        vr = base['log_probability'] - top_logp
        metrics = dict(question_id=s['question_id'], image_id=s['image_id'], category=s['category'],
                       target=s['answer'], token_count=base['token_count'], original_log_probability=base['log_probability'],
                       visual_reliance=vr, visual_reliance_per_token=vr / base['token_count'],
                       random_reliance=float(np.mean(control_drops)), excess_reliance=vr - float(np.mean(control_drops)),
                       attention_pearson=correlation(attn, causal) if attn is not None else None,
                       attention_spearman=correlation(attn, causal, rank=True) if attn is not None else None)
        save_json(path, dict(metrics=metrics, boxes=boxes, causal=causal.tolist(), masked_log_probabilities=masked_logp,
                             top_regions=top, top_masked_log_probability=top_logp, random_regions=controls,
                             random_drops=control_drops, attention=attn.tolist() if attn is not None else None,
                             attention_patch=base['attention'].tolist() if attn is not None else None))
        if args.plots:
            plot_maps(image, causal, attn, args.grid, path.with_suffix('.png'))
        rows.append(metrics)
        write_csv(args.output / 'intervention_metrics.csv', rows)
        print(f"[{i + 1}/{len(samples)}] {s['question_id']}: VR={vr:.4f}", flush=True)
    write_csv(args.output / 'intervention_metrics.csv', rows)
    save_json(args.output / 'intervention_summary.json', summarize(rows, [
        'visual_reliance', 'visual_reliance_per_token', 'random_reliance', 'excess_reliance',
        'attention_pearson', 'attention_spearman'], args.seed))


def features_stage(args):
    import numpy as np
    samples = samples_from(args.subset)
    if args.limit:
        samples = samples[:args.limit]
    record_config(args, samples)
    model = backend(args)
    directory = args.output / 'features'
    directory.mkdir(exist_ok=True)
    metadata = []
    for i, s in enumerate(samples):
        path = directory / f"{s['question_id']}.npz"
        if not path.exists():
            image = model.image(args.images / s['image_filename'])
            # Causal masking prevents later answer tokens from contaminating earlier image states.
            measured = model.measure(image, s['question'], s['answer'], layers=args.layers)
            temporary = path.with_suffix('.tmp.npz')
            np.savez_compressed(temporary, **{f'layer_{l}': h for l, h in measured['states'].items()})
            temporary.replace(path)
        metadata.append(dict(question_id=s['question_id'], image_id=s['image_id'], category=s['category'],
                             label=normalize(s['answer']), layers=args.layers))
        save_json(args.output / 'feature_metadata.json', metadata)
        print(f"[{i + 1}/{len(samples)}] features {s['question_id']}", flush=True)


def probe_stage(args):
    from .analysis import run_probes, summarize
    metadata = load_json(args.features / 'feature_metadata.json')
    _, predictions = run_probes(metadata, args.features / 'features', args.output, args.folds, args.seed)
    by_layer = {}
    for layer in sorted({r['layer'] for r in predictions}):
        by_layer[str(layer)] = summarize([r for r in predictions if r['layer'] == layer], ['correct'], args.seed)
    save_json(args.output / 'probe_summary.json', by_layer)
    if args.interventions:
        intervention = {str(r['question_id']): r for r in read_csv(args.interventions / 'intervention_metrics.csv')}
        joined = [dict(p, visual_reliance=float(intervention[str(p['question_id'])]['visual_reliance']))
                  for p in predictions if str(p['question_id']) in intervention]
        write_csv(args.output / 'probe_reliance_joined.csv', joined)
        save_json(args.output / 'probe_reliance_summary.json', {
            str(layer): {status: summarize([r for r in joined if r['layer'] == layer and r['correct'] == hit],
                                          ['visual_reliance'], args.seed)
                         for status, hit in [('probe_correct', 1), ('probe_incorrect', 0)]}
            for layer in sorted({r['layer'] for r in joined})})


def conflict_stage(args):
    """A reviewed manifest is required: no synthetic labels guessed from language priors."""
    from .analysis import summarize
    manifest = load_json(args.manifest)
    if not manifest:
        raise ValueError('Empty conflict manifest')
    pairs = {}
    for s in manifest:
        required = ('question_id', 'image_id', 'image_path', 'question', 'answer', 'all_answers',
                    'category', 'condition', 'pair_id', 'reviewed', 'provenance', 'language_prior_answer')
        if any(k not in s for k in required) or s['reviewed'] is not True:
            raise ValueError('Manifest entries need all documented fields and reviewed=true')
        if s['condition'] not in ('normal', 'conflict') or len(s['all_answers']) != 10:
            raise ValueError('Invalid condition or human answer count')
        if not s['provenance'] or not s['language_prior_answer']:
            raise ValueError('Missing evidence provenance or language prior')
        pair = pairs.setdefault(str(s['pair_id']), {})
        if s['condition'] in pair:
            raise ValueError('Duplicate pair condition')
        pair[s['condition']] = s
    if len({str(s['question_id']) for s in manifest}) != len(manifest):
        raise ValueError('Duplicate conflict question_id')
    for pair in pairs.values():
        if set(pair) != {'normal', 'conflict'}:
            raise ValueError('Each pair requires normal and conflict samples')
        a, b = pair['normal'], pair['conflict']
        if a['question'] != b['question'] or a['category'] != b['category']:
            raise ValueError('Matched pairs must share question and category')
    # Store a normalized subset, then reuse the same baseline and intervention protocols.
    samples = []
    for s in manifest:
        image_path = Path(s['image_path'])
        if not image_path.is_absolute():
            image_path = args.manifest.resolve().parent / image_path
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        samples.append(dict(s, image_filename=str(image_path.resolve())))
    record_config(args, samples)
    subset = args.output / 'conflict_subset.json'
    save_json(subset, samples)
    args.subset = subset
    args.limit = 0
    args.stage = 'baseline'
    baseline_stage(args)
    # Conflict comparison includes all reviewed pairs; S_correct is a separate conditional analysis.
    args.stage = 'intervene'
    intervention_stage(args)
    acc = {str(r['question_id']): r for r in read_csv(args.output / 'baseline_evaluated.csv')}
    vr = {str(r['question_id']): r for r in read_csv(args.output / 'intervention_metrics.csv')}
    rows = [dict(question_id=s['question_id'], image_id=s['pair_id'], category=s['category'], condition=s['condition'],
                 soft_accuracy=float(acc[str(s['question_id'])]['soft_accuracy']),
                 correct=int(acc[str(s['question_id'])]['correct']),
                 visual_reliance=float(vr[str(s['question_id'])]['visual_reliance'])) for s in samples]
    write_csv(args.output / 'conflict_metrics.csv', rows)
    save_json(args.output / 'conflict_summary.json', {
        condition: dict(all_samples=summarize([r for r in rows if r['condition'] == condition], ['soft_accuracy', 'visual_reliance'], args.seed),
                        correct_only=summarize([r for r in rows if r['condition'] == condition and r['correct']], ['visual_reliance'], args.seed))
        for condition in ('normal', 'conflict')})
    paired = []
    for pair_id, pair in pairs.items():
        normal, conflict = pair['normal'], pair['conflict']
        n, c = str(normal['question_id']), str(conflict['question_id'])
        paired.append(dict(image_id=pair_id, category=normal['category'],
                           accuracy_drop=float(acc[n]['soft_accuracy']) - float(acc[c]['soft_accuracy']),
                           reliance_change=float(vr[c]['visual_reliance']) - float(vr[n]['visual_reliance'])))
    save_json(args.output / 'conflict_paired_summary.json', summarize(paired, ['accuracy_drop', 'reliance_change'], args.seed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'evaluate', 'baseline', 'intervene', 'features', 'probe', 'conflict'])
    parser.add_argument('--subset', type=Path, help='Defaults to the prepared subset; intervene defaults to output/correct_subset.json')
    parser.add_argument('--images', type=Path, default=IMAGES)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/visual_reliance')
    parser.add_argument('--questions', type=Path, default=ROOT / 'data/vqav2/questions/v2_OpenEnded_mscoco_val2014_questions.json')
    parser.add_argument('--annotations', type=Path, default=ROOT / 'data/vqav2/annotations/v2_mscoco_val2014_annotations.json')
    parser.add_argument('--per-category', type=int, default=60)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--predictions', type=Path, default=ROOT / 'results/vqav2_baseline_300.csv')
    parser.add_argument('--reclassify', action='store_true', help='Audit categories of legacy predictions')
    parser.add_argument('--correct-threshold', type=float, default=1.0)
    parser.add_argument('--model', default='llava-hf/llava-1.5-7b-hf')
    parser.add_argument('--quantization', choices=['4bit', 'none'], default='4bit')
    parser.add_argument('--allow-download', action='store_true')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--grid', type=int, default=4)
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--fill', choices=['mean', 'gray'], default='mean')
    parser.add_argument('--random-controls', type=int, default=3)
    parser.add_argument('--attention', action='store_true')
    parser.add_argument('--plots', action='store_true')
    parser.add_argument('--layers', nargs='+', type=int, default=[2, 8, 16, 24, 32])
    parser.add_argument('--features', type=Path, default=ROOT / 'results/visual_reliance')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--interventions', type=Path)
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    if args.subset is None:
        args.subset = args.output / 'correct_subset.json' if args.stage == 'intervene' else SUBSET
    if not 0 < args.correct_threshold <= 1 or args.random_controls < 1 or args.folds < 2 or args.limit < 0 or args.grid < 1:
        parser.error('Invalid threshold, controls, folds, limit or grid')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == 'intervene' and not args.subset.exists():
        parser.error('intervene needs S_correct: provide --subset <baseline output>/correct_subset.json')
    if args.stage == 'prepare':
        if args.per_category < 1:
            parser.error('per-category must be positive')
        print(prepare(args.questions, args.annotations, args.subset, args.per_category, args.seed))
    elif args.stage == 'conflict' and not args.manifest:
        parser.error('conflict requires --manifest')
    else:
        globals()[args.stage.replace('intervene', 'intervention') + '_stage'](args)


if __name__ == '__main__':
    main()
