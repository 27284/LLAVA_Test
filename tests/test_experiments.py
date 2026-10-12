import tempfile
import unittest
import sys
from unittest.mock import patch
from pathlib import Path
import numpy as np
from PIL import Image
from experiments.data import classify, evaluate, normalize, soft_accuracy, load_json, save_json
from experiments.analysis import correlation, linear_predictions, ranks, run_probes
from experiments.model import LlavaBackend, attention_grid, grid_boxes, occlude


class MetricsTests(unittest.TestCase):
    def test_vqa_consensus(self):
        for count, expected in [(0, 0), (1, .3), (2, .6), (3, .9), (4, 1), (10, 1)]:
            self.assertAlmostEqual(soft_accuracy('blue', ['blue'] * count + ['red'] * (10 - count)), expected)
        self.assertEqual(soft_accuracy('two', ['2'] * 6 + ['two'] * 4), 1)
        self.assertEqual(soft_accuracy('The chairs are blue.', ['blue'] * 10), 0)
        with self.assertRaises(ValueError):
            soft_accuracy('blue', ['blue'])

    def test_normalization(self):
        self.assertEqual(normalize('The two cats!'), '2 cats')
        self.assertEqual(normalize('1,000'), '1000')
        self.assertEqual(normalize('3.14'), '3.14')
        self.assertEqual(normalize('dont'), "don't")

    def test_categories(self):
        self.assertEqual(classify('What is the color of the surfboard?'), 'color')
        self.assertEqual(classify('How many dogs are under the table?', 'number'), 'count')
        self.assertEqual(classify('Is the dog under the table?', 'yes/no'), 'yes_no')
        self.assertIsNone(classify('How much does this cost?', 'number'))

    def test_selection(self):
        samples = [dict(question_id=1, image_id=2, image_filename='a.jpg', category='color',
                        question='Color?', answer='blue', all_answers=['blue'] * 3 + ['red'] * 7)]
        result = evaluate(samples, [dict(question_id=1, pred_answer='blue')])
        self.assertAlmostEqual(result[0]['soft_accuracy'], .9)
        self.assertEqual(result[0]['correct'], 0)

    def test_geometry_and_attention(self):
        boxes = grid_boxes(7, 9, 4)
        covered = np.zeros((9, 7))
        for x0, y0, x1, y1 in boxes:
            covered[y0:y1, x0:x1] += 1
        np.testing.assert_array_equal(covered, 1)
        image = Image.new('RGB', (7, 9), (255, 255, 255))
        masked = occlude(image, boxes, [0], 'gray')
        self.assertEqual(masked.getpixel((0, 0)), (127, 127, 127))
        self.assertEqual(masked.getpixel((6, 8)), (255, 255, 255))
        attention = np.arange(36, dtype=float)
        self.assertAlmostEqual(attention_grid(attention, 4).sum(), attention.sum())

    def test_correlations(self):
        np.testing.assert_equal(ranks([3, 1, 1]), [2, .5, .5])
        self.assertIsNone(correlation([1, 1], [1, 2]))
        self.assertAlmostEqual(correlation([1, 2, 3], [9, 5, 1], rank=True), -1)

    def test_probe(self):
        x = np.array([[0, 1], [0, 2], [3, 0], [4, 0]], dtype=float)
        self.assertEqual(linear_predictions(x, ['a', 'a', 'b', 'b'], x, ['a', 'b']), ['a', 'a', 'b', 'b'])
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            metadata = []
            for i in range(12):
                label = 'blue' if i % 2 else 'red'
                metadata.append(dict(question_id=i, image_id=i // 2, category='color', label=label, layers=[2]))
                np.savez(directory / f'{i}.npz', layer_2=np.array([i % 2, 1 - i % 2]))
            rows, predictions = run_probes(metadata, directory, directory, folds=3)
            self.assertEqual(rows[0]['accuracy'], 1)
            for image in range(6):
                self.assertEqual(len({p['fold'] for p in predictions if p['image_id'] == image}), 1)


class TinyModelTests(unittest.TestCase):
    def test_likelihood_attention_and_hooks(self):
        import torch
        from transformers import LlavaConfig, LlavaForConditionalGeneration
        config = LlavaConfig(
            vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                               num_attention_heads=2, image_size=28, patch_size=14),
            text_config=dict(vocab_size=64, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2),
            image_token_index=63, image_token_id=63, vision_feature_layer=-2)
        config._attn_implementation = 'eager'
        config.text_config._attn_implementation = 'eager'
        model = LlavaForConditionalGeneration(config).eval()
        class Tokenizer:
            def __call__(self, *args, **kwargs):
                return {'input_ids': torch.tensor([[7, 8]])}
        class Processor:
            tokenizer = Tokenizer()
        backend = LlavaBackend.__new__(LlavaBackend)
        backend.model, backend.processor = model, Processor()
        backend.size, backend.patch_size, backend.image_token_id = 28, 14, 63
        backend.layers = model.model.language_model.layers
        def inputs(image, question):
            return dict(input_ids=torch.tensor([[1, 63, 63, 63, 63, 2, 3]]),
                        attention_mask=torch.ones(1, 7, dtype=torch.long), pixel_values=torch.zeros(1, 3, 28, 28))
        backend.inputs = inputs
        measured = backend.measure(None, 'test', 'red', layers=[1, 2], attention=True)
        data = inputs(None, None)
        data['input_ids'] = torch.cat([data['input_ids'], torch.tensor([[7, 8]])], 1)
        data['attention_mask'] = torch.ones(1, 9, dtype=torch.long)
        with torch.no_grad():
            output = model(**data, output_attentions=True, return_dict=True, use_cache=False)
        expected = output.logits[0, 6:8].log_softmax(-1)[torch.arange(2), torch.tensor([7, 8])].sum().item()
        self.assertAlmostEqual(measured['log_probability'], expected, places=5)
        manual_attention = torch.stack([a[0, :, 6:8, 1:5].mean((0, 1)) for a in output.attentions]).mean(0).numpy()
        np.testing.assert_allclose(measured['attention'], manual_attention, atol=1e-7)
        self.assertEqual(measured['states'][1].shape, (16,))
        self.assertNotIn('capture', [h.__name__ for h in backend.layers[0]._forward_hooks.values()])
        self.assertNotIn('attention_hook', [h.__name__ for h in backend.layers[0].self_attn._forward_hooks.values()])
        self.assertNotIn('require_attention', [h.__name__ for h in backend.layers[0].self_attn._forward_pre_hooks.values()])
        # Visual positions precede the question: changing later question/answer tokens cannot change them.
        previous = measured['states'][1]
        original_inputs = backend.inputs
        def changed_inputs(image, question):
            data = original_inputs(image, question)
            data['input_ids'][0, -2:] = torch.tensor([10, 11])
            return data
        backend.inputs = changed_inputs
        changed = backend.measure(None, 'different', 'blue', layers=[1])
        np.testing.assert_allclose(previous, changed['states'][1], atol=1e-6)


class PipelineTests(unittest.TestCase):
    def test_all_stages_and_conflict_resume(self):
        from experiments.run import main
        class FakeBackend:
            def image(self, path):
                with Image.open(path) as image:
                    return image.convert('RGB')
            def answer(self, image, question):
                return 'red' if np.asarray(image)[0, 0, 0] else 'green'
            def measure(self, image, question, target, layers=(), attention=False):
                value = float(np.asarray(image).mean()) / 255
                return dict(log_probability=value - 2, token_count=1,
                            states={l: np.array([value, 1 - value, float(l)]) for l in layers},
                            attention=np.array([.1, .2, .3, .4]) if attention else None)
        with tempfile.TemporaryDirectory() as temp, patch('experiments.run.backend', return_value=FakeBackend()):
            root = Path(temp)
            samples, manifest = [], []
            for i in range(4):
                label = 'red' if i % 2 == 0 else 'green'
                path = root / f'{i}.png'
                Image.new('RGB', (8, 8), label).save(path)
                sample = dict(question_id=i + 1, image_id=i + 1, image_filename=str(path),
                              question='What color is this?', category='color', answer=label,
                              all_answers=[label] * 10)
                samples.append(sample)
                manifest.append(dict(sample, image_path=str(path), pair_id=i // 2,
                                     condition='normal' if i % 2 == 0 else 'conflict', reviewed=True,
                                     provenance='Synthetic fixture only; not human data', language_prior_answer='red'))
            subset = root / 'subset.json'
            save_json(subset, samples)
            def execute(arguments):
                with patch.object(sys, 'argv', ['run'] + arguments):
                    main()
            out = root / 'out'
            execute(['baseline', '--subset', str(subset), '--output', str(out)])
            self.assertEqual(len(load_json(out / 'correct_subset.json')), 4)
            execute(['intervene', '--output', str(out), '--grid', '2', '--top-k', '1', '--attention', '--random-controls', '1', '--plots'])
            self.assertTrue((out / 'maps/1.png').exists())
            execute(['features', '--subset', str(subset), '--output', str(out), '--layers', '1'])
            execute(['probe', '--features', str(out), '--output', str(out), '--folds', '2', '--interventions', str(out)])
            self.assertTrue((out / 'probe_reliance_joined.csv').exists())
            conflict_manifest = root / 'manifest.json'
            save_json(conflict_manifest, manifest)
            conflict_out = root / 'conflict'
            arguments = ['conflict', '--manifest', str(conflict_manifest), '--output', str(conflict_out),
                         '--grid', '2', '--top-k', '1', '--random-controls', '1', '--attention']
            execute(arguments)
            execute(arguments)
            summary = load_json(conflict_out / 'conflict_summary.json')
            self.assertEqual(summary['normal']['all_samples']['overall']['soft_accuracy']['n'], 2)
            self.assertEqual(summary['conflict']['all_samples']['overall']['visual_reliance']['n'], 2)


if __name__ == '__main__':
    unittest.main()
