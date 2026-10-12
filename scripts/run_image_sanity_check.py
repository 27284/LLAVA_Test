"""Optional original/blank/noise sanity check."""
import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.model import LlavaBackend

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--question', default='What animal is in this image?')
    parser.add_argument('--model', default='llava-hf/llava-1.5-7b-hf')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--allow-download', action='store_true')
    args = parser.parse_args()
    model = LlavaBackend(args.model, local_only=not args.allow_download)
    image = model.image(args.image)
    rng = np.random.default_rng(args.seed)
    images = dict(original=image, blank=Image.new('RGB', image.size, 'white'),
                  noise=Image.fromarray(rng.integers(0, 256, (image.height, image.width, 3), dtype=np.uint8)))
    for name, variant in images.items():
        print(f'{name}: {model.answer(variant, args.question)}', flush=True)

if __name__ == '__main__':
    main()
