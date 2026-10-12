"""Group-aware statistics and linear probes implemented with NumPy."""
from collections import Counter
import numpy as np
from .data import save_json, write_csv


def ranks(values):
    values = np.asarray(values)
    order = np.argsort(values, kind='stable')
    result = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start + end - 1) / 2
        start = end
    return result


def correlation(a, b, rank=False):
    a, b = np.asarray(a), np.asarray(b)
    if rank:
        a, b = ranks(a), ranks(b)
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def group_interval(rows, key, seed=42, repetitions=1000):
    valid = [r for r in rows if r.get(key) is not None]
    if not valid:
        return dict(n=0, mean=None, ci95=None)
    groups = {}
    for r in valid:
        groups.setdefault(str(r['image_id']), []).append(float(r[key]))
    values = list(groups.values())
    rng = np.random.default_rng(seed)
    estimates = [np.mean([v for i in rng.integers(len(values), size=len(values)) for v in values[i]])
                 for _ in range(repetitions)]
    return dict(n=len(valid), images=len(values), mean=float(np.mean([float(r[key]) for r in valid])),
                ci95=[float(x) for x in np.quantile(estimates, [.025, .975])])


def summarize(rows, keys, seed=42):
    categories = sorted({r['category'] for r in rows})
    return {category: {key: group_interval(rows if category == 'overall' else [r for r in rows if r['category'] == category], key, seed)
                       for key in keys} for category in ['overall'] + categories}


def linear_predictions(train, labels, test, classes, alpha=10):
    # Standardize using training data only; solve the dual ridge system (n << hidden dimension).
    mean, std = train.mean(0), train.std(0)
    std = np.maximum(std, 1e-6)
    x, z = (train - mean) / std, (test - mean) / std
    y = np.eye(len(classes))[[classes.index(label) for label in labels]]
    prior = y.mean(0)
    weights = np.linalg.solve(x @ x.T + alpha * np.eye(len(x)), y - prior)
    return [classes[i] for i in (z @ x.T @ weights + prior).argmax(1)]


def run_probes(metadata, feature_dir, output, folds=5, seed=42):
    """Image-disjoint OOF predictions; labels never appended to feature extraction prompts."""
    from pathlib import Path
    rows, predictions = [], []
    rng = np.random.default_rng(seed)
    for category in sorted({r['category'] for r in metadata}):
        samples = [r for r in metadata if r['category'] == category]
        groups = sorted({str(r['image_id']) for r in samples})
        rng.shuffle(groups)
        if len(groups) < folds:
            rows.append(dict(category=category, status='insufficient_images', n=len(samples)))
            continue
        assignment = {g: i % folds for i, g in enumerate(groups)}
        layers = sorted(samples[0]['layers'])
        arrays = []
        for s in samples:
            with np.load(Path(feature_dir) / f"{s['question_id']}.npz") as archive:
                arrays.append({layer: archive[f'layer_{layer}'].astype(np.float64) for layer in layers})
        labels = [s['label'] for s in samples]
        classes = sorted(set(labels))
        if len(classes) < 2:
            rows.append(dict(category=category, status='insufficient_classes', n=len(samples)))
            continue
        for layer in layers:
            correct, baseline, shuffled, unseen = [], [], [], []
            for fold in range(folds):
                train_ids = [i for i, s in enumerate(samples) if assignment[str(s['image_id'])] != fold]
                test_ids = [i for i, s in enumerate(samples) if assignment[str(s['image_id'])] == fold]
                x = np.stack([arrays[i][layer] for i in train_ids])
                z = np.stack([arrays[i][layer] for i in test_ids])
                train_labels = [labels[i] for i in train_ids]
                majority = Counter(train_labels).most_common(1)[0][0]
                # Class vocabulary determined on train only; unseen test labels count as failures.
                train_classes = sorted(set(train_labels))
                pred = linear_predictions(x, train_labels, z, train_classes)
                permuted = list(rng.permutation(train_labels))
                null_pred = linear_predictions(x, permuted, z, train_classes)
                for i, p, null in zip(test_ids, pred, null_pred):
                    hit = int(p == labels[i])
                    correct.append(hit)
                    baseline.append(int(majority == labels[i]))
                    shuffled.append(int(null == labels[i]))
                    unseen.append(int(labels[i] not in train_classes))
                    predictions.append(dict(question_id=samples[i]['question_id'], image_id=samples[i]['image_id'],
                                            category=category, layer=layer, fold=fold, label=labels[i],
                                            prediction=p, correct=hit))
            rows.append(dict(category=category, layer=layer, n=len(samples), classes=len(classes), status='ok',
                             accuracy=float(np.mean(correct)), majority_accuracy=float(np.mean(baseline)),
                             shuffled_label_accuracy=float(np.mean(shuffled)), unseen_test_labels=sum(unseen),
                             alpha=10, folds=folds))
    write_csv(Path(output) / 'probe_metrics.csv', rows)
    write_csv(Path(output) / 'probe_predictions.csv', predictions)
    return rows, predictions


def plot_maps(image, causal, attention, grid, path):
    try:
        import matplotlib
    except ImportError:
        return plot_maps_pillow(image, causal, attention, grid, path)
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    maps = [('Causal log-probability drop', causal)]
    if attention is not None:
        maps.append(('Answer-predicting attention', attention))
    fig, axes = plt.subplots(1, len(maps) + 1, figsize=(4 * (len(maps) + 1), 4))
    axes[0].imshow(image)
    axes[0].set_title('Square-resized model input')
    axes[0].axis('off')
    for axis, (title, values) in zip(axes[1:], maps):
        matrix = np.asarray(values).reshape(grid, grid)
        kwargs = dict(cmap='viridis')
        if title.startswith('Causal'):
            bound = max(float(np.abs(matrix).max()), 1e-6)
            kwargs = dict(cmap='coolwarm', vmin=-bound, vmax=bound)
        artist = axis.imshow(matrix, **kwargs)
        axis.set_title(title)
        axis.set_xticks(range(grid))
        axis.set_yticks(range(grid))
        fig.colorbar(artist, ax=axis, shrink=.7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_maps_pillow(image, causal, attention, grid, path):
    """Dependency-light fallback with numeric cell values and explicit color scales."""
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype('arial.ttf', 15)
        heading = ImageFont.truetype('arial.ttf', 18)
    except OSError:
        font = heading = ImageFont.load_default()
    size, padding, top = 336, 20, 65
    maps = [('Causal logP drop', causal)]
    if attention is not None:
        maps.append(('Answer-predicting attention', attention))
    panel_count = 1 + len(maps)
    canvas = Image.new('RGB', (padding + panel_count * (size + padding), top + size + 65), 'white')
    draw = ImageDraw.Draw(canvas)
    canvas.paste(image.resize((size, size)), (padding, top))
    draw.text((padding, 15), 'Square-resized model input', font=heading, fill='black')
    for panel, (title, values) in enumerate(maps, 1):
        x0 = padding + panel * (size + padding)
        array = np.asarray(values).reshape(grid, grid)
        causal_map = title.startswith('Causal')
        bound = max(float(np.abs(array).max() if causal_map else array.max()), 1e-9)
        draw.text((x0, 15), title, font=heading, fill='black')
        for r in range(grid):
            for c in range(grid):
                value = float(array[r, c])
                ratio = min(abs(value) / bound, 1)
                extreme = np.array([190, 35, 45] if value >= 0 else [35, 85, 180]) if causal_map else np.array([75, 35, 130])
                color = tuple((255 * (1 - ratio) + extreme * ratio).astype(int))
                box = (x0 + c * size // grid, top + r * size // grid,
                       x0 + (c + 1) * size // grid - 1, top + (r + 1) * size // grid - 1)
                draw.rectangle(box, fill=color, outline='black')
                text = f'{value:.3f}' if causal_map else f'{value:.2e}'
                center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                draw.text(center, text, font=font, fill='white' if ratio > .65 else 'black', anchor='mm')
        legend = f'Blue: negative; white: 0; red: positive; max |C|={bound:.3f}' if causal_map else f'White: 0; purple: max={bound:.2e}'
        draw.text((x0, top + size + 12), legend, font=font, fill='black')
    canvas.save(path)
