
# Precompute text prototypes (class name + prompt ensembling + optional descriptors)
# Requires either:
#   - open_clip (pip install open-clip-torch)
#   - or torchvision CLIP (PyTorch 2.4+).
#
# This script writes a .npz file with key 'prototypes' holding [C,D] L2-normalized vectors.

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F

PROMPTS = [
    "a photo of a {}",
    "a close-up photo of a {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a bright photo of a {}",
    "a dark photo of a {}",
    "a low-contrast photo of a {}",
    "a high-contrast photo of a {}",
    "a side-view photo of a {}",
    "a photo of a small {}",
    "a photo of a large {}",
    "an indoor photo of a {}",
    "an outdoor photo of a {}",
]

def load_clip_text_encoder(device: torch.device):
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k")
        tokenizer = open_clip.get_tokenizer("ViT-B-16")
        model = model.to(device).eval()
        return model, tokenizer, "open_clip"
    except Exception:
        # Try torchvision CLIP (PyTorch 2.4+)
        try:
            import torchvision.models as tvm
            model, preprocess = tvm.clip.load("ViT-B-16", weights=tvm.ClipViTB16Weights.DEFAULT)
            tokenizer = tvm.clip.tokenize
            model = model.to(device).eval()
            return model, tokenizer, "torchvision"
        except Exception as e:
            raise RuntimeError(
                "Could not load a CLIP text encoder. Install open-clip: `pip install open-clip-torch` "
                "or upgrade to PyTorch/torchvision with CLIP support."
            ) from e

def encode_texts(model, tokenizer, texts, backend: str, device: torch.device):
    with torch.no_grad():
        if backend == "open_clip":
            toks = tokenizer(texts)
            embs = model.encode_text(toks.to(device))
        else:
            toks = tokenizer(texts).to(device)
            embs = model.encode_text(toks)
        embs = F.normalize(embs.float(), dim=1)
        return embs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classnames", type=str, required=True, help="Path to a JSON list of class names (length C)")
    parser.add_argument("--descriptors", type=str, default=None, help="Optional JSON: dict {class_name: [phrases,...]}")
    parser.add_argument("--out", type=str, required=True, help="Output .npz path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, tokenizer, backend = load_clip_text_encoder(device)

    classnames = json.load(open(args.classnames, "r"))
    if not isinstance(classnames, list):
        raise ValueError("classnames must be a JSON list")

    desc_map = {}
    if args.descriptors is not None and len(args.descriptors) > 0:
        desc_map = json.load(open(args.descriptors, "r"))
        if not isinstance(desc_map, dict):
            raise ValueError("descriptors must be a JSON object mapping class -> list[str]")

    all_protos = []
    for cname in classnames:
        prompts = [p.format(cname) for p in PROMPTS]
        # Optional: add short factual descriptors
        for d in desc_map.get(cname, []):
            prompts.append(d.strip())

        # Batch encode prompts and average
        embs = []
        bs = 64
        for i in range(0, len(prompts), bs):
            embs.append(encode_texts(model, tokenizer, prompts[i:i+bs], backend, device).cpu())
        embs = torch.cat(embs, dim=0)
        proto = embs.mean(dim=0, keepdim=True)
        proto = torch.nn.functional.normalize(proto, dim=1)
        all_protos.append(proto)

    import torch as _torch
    P = _torch.cat(all_protos, dim=0).cpu().numpy()  # [C, D]
    np.savez(args.out, prototypes=P)
    print(f"Saved prototypes to {args.out}, shape={P.shape}")

if __name__ == "__main__":
    main()
