"""HF LLaVA 1.5 backend: answer likelihood, attention and pooled visual states."""
import math
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration


def grid_boxes(width, height, grid):
    if grid < 1 or grid > min(width, height):
        raise ValueError('Invalid grid size')
    return [(c * width // grid, r * height // grid, (c + 1) * width // grid, (r + 1) * height // grid)
            for r in range(grid) for c in range(grid)]


def occlude(image, boxes, regions, fill='mean'):
    result = image.copy()
    color = tuple(np.asarray(image).mean(axis=(0, 1)).round().astype(int)) if fill == 'mean' else (127, 127, 127)
    for index in regions:
        result.paste(color, boxes[index])
    return result


class LlavaBackend:
    def __init__(self, model_id, quantization='4bit', local_only=True):
        self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_only)
        kwargs = dict(device_map='auto', attn_implementation='eager', local_files_only=local_only,
                      dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
        if quantization == '4bit':
            if not torch.cuda.is_available():
                raise ValueError('4bit experiments require CUDA; use --quantization none on CPU')
            kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self.model = LlavaForConditionalGeneration.from_pretrained(model_id, **kwargs).eval()
        config = self.model.config
        if config.model_type != 'llava' or config.vision_feature_select_strategy != 'default':
            raise ValueError('This backend supports single-image LLaVA 1.5 patch features only')
        self.size = config.vision_config.image_size
        self.patch_size = config.vision_config.patch_size
        self.processor.patch_size = self.patch_size
        self.processor.vision_feature_select_strategy = 'default'
        self.processor.num_additional_image_tokens = 1
        self.image_token_id = getattr(config, 'image_token_id', getattr(config, 'image_token_index', None))
        language = self.model.model.language_model
        self.layers = language.layers if hasattr(language, 'layers') else language.model.layers

    def image(self, path):
        # Square resize defines a shared coordinate system: no invisible center-cropped regions.
        with Image.open(path) as image:
            return image.convert('RGB').resize((self.size, self.size), Image.Resampling.BICUBIC)

    def inputs(self, image, question):
        prompt = f'USER: <image>\n{question}\nAnswer with a single word or short phrase.\nASSISTANT:'
        inputs = self.processor(text=prompt, images=image, return_tensors='pt')
        return {k: v.to(self.model.device, dtype=self.model.dtype) if v.is_floating_point()
                else v.to(self.model.device) for k, v in inputs.items()}

    @torch.inference_mode()
    def answer(self, image, question):
        inputs = self.inputs(image, question)
        ids = self.model.generate(**inputs, max_new_tokens=20, do_sample=False)
        return self.processor.tokenizer.decode(ids[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

    @torch.inference_mode()
    def measure(self, image, question, target, layers=(), attention=False):
        inputs = self.inputs(image, question)
        prompt_length = inputs['input_ids'].shape[1]
        # Llama's tokenizer supplies the word-boundary prefix itself; manually adding
        # a leading space would insert an extra whitespace token and distort logP.
        answer_ids = self.processor.tokenizer(target, add_special_tokens=False, return_tensors='pt')['input_ids'].to(self.model.device)
        if answer_ids.numel() == 0:
            raise ValueError('Empty target answer')
        inputs['input_ids'] = torch.cat([inputs['input_ids'], answer_ids], dim=1)
        inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
        positions = torch.where(inputs['input_ids'][0] == self.image_token_id)[0]
        expected = (self.size // self.patch_size) ** 2
        if len(positions) != expected:
            raise ValueError(f'Expected {expected} expanded image tokens, found {len(positions)}')
        if any(layer < 1 or layer > len(self.layers) for layer in layers):
            raise ValueError(f'Layer indices must be 1..{len(self.layers)}')
        states, attentions, handles = {}, [], []
        def state_hook(number):
            def capture(module, args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                states[number] = hidden[0, positions.to(hidden.device)].float().mean(0).cpu().numpy()
            return capture
        def attention_hook(module, args, output):
            weights = output[1]
            if weights is None:
                raise RuntimeError('Attention weights unavailable: eager attention is required')
            # Queries that predict each answer token, including the first token's prompt query.
            selected = weights[0, :, prompt_length - 1:prompt_length + answer_ids.shape[1] - 1]
            attentions.append(selected.index_select(-1, positions.to(weights.device)).float().mean((0, 1)).cpu().numpy())
        def require_attention(module, args, kwargs):
            # Older Transformers Llama versions gate returned weights on this flag.
            # Set it on each attention module, without retaining whole-model matrices.
            kwargs['output_attentions'] = True
            return args, kwargs
        try:
            for number in layers:
                handles.append(self.layers[number - 1].register_forward_hook(state_hook(number)))
            if attention:
                for layer in self.layers:
                    handles.append(layer.self_attn.register_forward_pre_hook(require_attention, with_kwargs=True))
                    handles.append(layer.self_attn.register_forward_hook(attention_hook))
            output = self.model(**inputs, use_cache=False, return_dict=True)
            logits = output.logits[0, prompt_length - 1:prompt_length + answer_ids.shape[1] - 1].float()
            logp = logits.log_softmax(-1).gather(-1, answer_ids[0].to(logits.device).unsqueeze(-1)).sum().item()
        finally:
            for handle in handles:
                handle.remove()
        return dict(log_probability=logp, token_count=answer_ids.shape[1], states=states,
                    attention=np.mean(attentions, axis=0) if attentions else None)


def attention_grid(patches, grid):
    side = math.isqrt(len(patches))
    if side * side != len(patches):
        raise ValueError('Visual tokens do not form a square patch grid')
    # Area-weighted mapping also supports grids not divisible into the patch lattice.
    result = np.zeros((grid, grid))
    patch_map = np.asarray(patches).reshape(side, side)
    for r in range(grid):
        for c in range(grid):
            for y in range(side):
                dy = max(0, min((r + 1) / grid, (y + 1) / side) - max(r / grid, y / side)) * side
                for x in range(side):
                    dx = max(0, min((c + 1) / grid, (x + 1) / side) - max(c / grid, x / side)) * side
                    result[r, c] += patch_map[y, x] * dx * dy
    return result.ravel()
