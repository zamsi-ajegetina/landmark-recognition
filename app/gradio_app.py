"""
Gradio demo for African Landmark Recognition.

Launch:
    python app/gradio_app.py

Environment variables (optional):
    CHECKPOINT_PATH   - path to TorchScript .pt (default: checkpoints/A4_resnet50_full_exported.pt)
    RAW_CHECKPOINT    - path to raw state-dict .pt for GradCAM (default: auto-detected)
    N_CLASSES         - number of classes (default: 50)
"""

import os
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.helpers import compute_mean_and_std
from src.transfer import get_model_transfer_learning

# ── Constants ─────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
_CKPT_DIR = _ROOT / "checkpoints"

CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT_PATH",
    str(_CKPT_DIR / "A4_resnet50_full_exported.pt"),
)
N_CLASSES = int(os.environ.get("N_CLASSES", 50))

# Auto-detect best available raw checkpoint for GradCAM.
_RAW_CANDIDATES = [
    ("A4_resnet50_full.pt", "resnet50"),
    ("A2_resnet18_full.pt", "resnet18"),
    ("A7_resnet50_minimal_aug.pt", "resnet50"),
    ("A8_resnet50_weighted.pt", "resnet50"),
]
_raw_ckpt_path = None
_raw_ckpt_arch = None

if "RAW_CHECKPOINT" in os.environ:
    _raw_ckpt_path = os.environ["RAW_CHECKPOINT"]
    _raw_ckpt_arch = "resnet50"
else:
    for _fname, _arch in _RAW_CANDIDATES:
        _candidate = _CKPT_DIR / _fname
        if _candidate.exists():
            _raw_ckpt_path = str(_candidate)
            _raw_ckpt_arch = _arch
            break


mean, std = compute_mean_and_std()
_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

_model_ts = None
_model_raw = None
_class_names = None


def _get_class_names():
    global _class_names
    if _class_names is not None:
        return _class_names
    try:
        from src.helpers import get_data_location
        from torchvision.datasets import ImageFolder
        ds = ImageFolder(Path(get_data_location()) / "test")
        _class_names = ds.classes
    except Exception:
        _class_names = [str(i) for i in range(N_CLASSES)]
    return _class_names


def _load_ts_model():
    global _model_ts
    if _model_ts is not None:
        return _model_ts
    if not Path(CHECKPOINT_PATH).exists():
        raise FileNotFoundError(
            f"TorchScript checkpoint not found at {CHECKPOINT_PATH}. "
            "Run `python ablation_runner.py --config configs/A4_resnet50_full.yaml` first."
        )
    _model_ts = torch.jit.load(CHECKPOINT_PATH, map_location="cpu")
    _model_ts.eval()
    return _model_ts


def _load_raw_model():
    global _model_raw
    if _model_raw is not None:
        return _model_raw
    if _raw_ckpt_path is None or not Path(_raw_ckpt_path).exists():
        return None
    model = get_model_transfer_learning(
        _raw_ckpt_arch, n_classes=N_CLASSES, finetune_strategy="full"
    )
    state = torch.load(_raw_ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    _model_raw = model
    print(f"GradCAM model loaded: {_raw_ckpt_arch} from {_raw_ckpt_path}")
    return _model_raw


def _compute_gradcam(pil_image: Image.Image, class_idx: int):
    try:
        from torchcam.methods import GradCAM
        from torchcam.utils import overlay_mask
        from torchvision.transforms.functional import to_pil_image
    except ImportError:
        return None

    model = _load_raw_model()
    if model is None:
        return None

    extractor = GradCAM(model, target_layer="layer4")
    tensor = _transform(pil_image).unsqueeze(0)
    logits = model(tensor)
    activation_map = extractor(class_idx, logits)[0]
    extractor.remove_hooks()

    heatmap = to_pil_image(activation_map, mode="F")
    result = overlay_mask(pil_image.resize((224, 224)), heatmap, alpha=0.5)
    return result


def _wikipedia_info(landmark_name: str) -> str:
    """Return a markdown string with the Wikipedia summary and link."""
    try:
        import wikipedia
        query = landmark_name.replace("_", " ").strip()

        wikipedia.set_lang("en")

        candidates = wikipedia.search(query, results=3)
        if not candidates:
            return ""

        for title in candidates:
            try:
                page = wikipedia.page(title, auto_suggest=False)
                summary = page.summary.split("\n")[0]
                return f"**Wikipedia:** {summary}\n\n[Read more]({page.url})"
            except wikipedia.DisambiguationError as e:
                if e.options:
                    try:
                        page = wikipedia.page(e.options[0], auto_suggest=False)
                        summary = page.summary.split("\n")[0]
                        return f"**Wikipedia:** {summary}\n\n[Read more]({page.url})"
                    except Exception:
                        continue
            except wikipedia.PageError:
                continue

        return ""
    except Exception:
        return ""


def predict(image: Image.Image):
    if image is None:
        return {}, None, "No image provided."

    pil_image = image.convert("RGB")
    tensor = _transform(pil_image).unsqueeze(0)

    try:
        model = _load_ts_model()
    except FileNotFoundError as e:
        return {}, None, str(e)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    class_names = _get_class_names()
    top_probs, top_idxs = torch.topk(probs, k=min(5, probs.size(0)))
    top_probs = top_probs.tolist()
    top_idxs = top_idxs.tolist()

    confidences = {class_names[i]: round(p, 4) for p, i in zip(top_probs, top_idxs)}
    top1_name = class_names[top_idxs[0]]

    gradcam_img = _compute_gradcam(pil_image, top_idxs[0])

    info_text = _wikipedia_info(top1_name)
    print('info text gotten',info_text)
    print('top class name ', top1_name)

    return confidences, gradcam_img, info_text


def build_interface():
    import gradio as gr

    if _raw_ckpt_path is None:
        gradcam_note = (
            "\n> **GradCAM unavailable** — no raw state-dict checkpoint found."
        )
    else:
        arch_label = "ResNet-50" if _raw_ckpt_arch == "resnet50" else "ResNet-18"
        gradcam_note = f"\n> GradCAM uses **{arch_label}** ({Path(_raw_ckpt_path).name})."

    with gr.Blocks(title="African Landmark Recognition") as demo:
        gr.Markdown(
            "# African Landmark Recognition\n"
            "Upload a landmark photo to get the top-5 predictions, "
            "a GradCAM heatmap, and a Wikipedia summary of the predicted landmark."
            + gradcam_note
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Upload Landmark Image")
                submit_btn = gr.Button("Classify", variant="primary")

            with gr.Column(scale=1):
                label_output = gr.Label(num_top_classes=5, label="Top-5 Predictions")
                gradcam_output = gr.Image(label="GradCAM Heatmap", type="pil")
                info_output = gr.Markdown(label="Landmark Info")

        submit_btn.click(
            fn=predict,
            inputs=[image_input],
            outputs=[label_output, gradcam_output, info_output],
        )

    return demo


if __name__ == "__main__":
    print(f"TorchScript checkpoint : {CHECKPOINT_PATH}")
    print(f"GradCAM raw checkpoint : {_raw_ckpt_path or 'NOT FOUND'} ({_raw_ckpt_arch or '-'})")
    demo = build_interface()
    demo.launch(share=False, server_name="0.0.0.0")
