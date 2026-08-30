# VL-KAN — Variable-Length CAPTCHA Recognition with Kolmogorov–Arnold Networks

Reference PyTorch 2.x implementation of **VL-KAN**, a variable-length CAPTCHA
recognizer that fuses a multi-scale convolutional trunk with **Kolmogorov–Arnold
Network (KAN)** layers and a length-/boundary-aware CTC decoder. It handles
numeric (10 classes) and case-sensitive alphanumeric (62 classes) codes of
length **3–10**.

The architecture is trained end-to-end with a three-phase curriculum and a
multi-task objective combining **Length-Aware Dynamic CTC**, a **length** head,
and a **boundary** head, with optional online **Adversarial CAPTCHA Augmentation
(ACA)**.

---

## 1. Project layout

```
.
├── datasets/        # vocabulary, procedural generator, augmentations, Dataset/collate
├── data/            # adversarial (PGD) augmentation utilities
├── models/
│   ├── backbone/    # ResNet18Seq sequence trunk
│   ├── kan/         # spline, KANLayer, KANBlock, multi-scale pyramid, A-KAN block
│   ├── heads/       # boundary head + soft-target builder
│   ├── deepcaptcha.py                 # fixed-length CNN baseline (Deep-CAPTCHA)
│   ├── variable_length_deepcaptcha.py # CRNN + CTC + length head (VarLen baseline)
│   └── kan_captcha.py                 # VLKAN (full model)
├── losses/          # Length-Aware Dynamic CTC loss
├── decode/          # length/boundary-aware CTC beam search
├── training/        # Trainer engine (AMP, grad-accum, curriculum, checkpointing)
├── evaluation/      # metrics (CAcc/SAcc/LAcc/CER), latency, decode adapters
├── experiments/     # automated structural ablation harness
├── train.py         # curriculum training entry point
└── infer.py         # inference loop
```

---

## 2. Installation

**pip**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**conda**

```bash
conda env create -f environment.yml
conda activate vlkan
```

> **CUDA:** install the matching PyTorch build from the official index
> (`pip install torch --index-url https://download.pytorch.org/whl/cu121`) before
> the remaining requirements. The code auto-selects CUDA when available; pass
> `--device cpu` to force CPU.

A quick environment check:

```bash
python -c "import torch, PIL; print('torch', torch.__version__)"
```

---

## 3. Data / preprocessing

CAPTCHA samples are generated **procedurally and deterministically** — there is no
download or offline preprocessing step. `datasets/CaptchaDataset` renders glyphs
(per-glyph rotation, tracking, baseline jitter) and `datasets/CaptchaAugmentor`
applies on-device, batched distortions (elastic, perspective, Perlin clutter,
spline occlusion). Determinism is keyed by `(base_seed, epoch, index)` so runs are
reproducible and resumable.

Sanity-check the generator and augmentation pipeline:

```bash
python -c "
from datasets.dataset import CaptchaDataset
from datasets.vocabulary import build_vocabulary
v = build_vocabulary('alphanumeric')
ds = CaptchaDataset(v, charset='0123456789', samples_per_epoch=4)
s = ds[0]
print('text=', s['text'], 'image=', tuple(s['image'].shape))
"
```

---

## 4. Curriculum training

`train.py` runs the full three-phase curriculum over a **fixed 62-class
vocabulary** (so the CTC head dimension is constant across phases):

| Phase | Charset | Lengths | ACA |
| --- | --- | --- | --- |
| **I** — numeric, short | `0–9` | 3–5 | off |
| **II** — numeric, full | `0–9` | 3–10 | off |
| **III** — alphanumeric | `0–9A–Za–z` | 3–10 | on (10%) |

Features: `torch.amp` mixed precision (bfloat16 default; float16 auto-enables a
`GradScaler`), gradient accumulation with global-norm clipping (≤ 5.0),
multi-task weighting `α=1.0` (CTC) · `β=0.3` (length) · `γ=0.2` (boundary),
atomic checkpointing, and TensorBoard telemetry.

**Launch a run**

```bash
python train.py --output-dir artifacts/run1
```

**Common overrides**

```bash
python train.py --output-dir artifacts/run1 \
  --batch-size 256 --grad-accum 2 --amp-dtype bfloat16 \
  --phase1-epochs 20 --phase2-epochs 40 --phase3-epochs 60 \
  --lr 3e-4 --weight-decay 0.05 --num-workers 8
```

**Pause / resume** (checkpoints are written to `<output-dir>/checkpoints/last.pt`):

```bash
python train.py --output-dir artifacts/run1 \
  --resume artifacts/run1/checkpoints/last.pt
```

**Tiny end-to-end validation run (CPU):**

```bash
python train.py --smoke --output-dir /tmp/vlkan_smoke
```

**Monitor:**

```bash
tensorboard --logdir artifacts/run1/logs
```

**Validation during training**

By default an 80/20 train/validation split is applied and validation runs at the
end of every epoch. The validation set is generated from a separate seed space so
it never overlaps with training data. Metrics logged under `val/` in TensorBoard:

| Tag | Meaning |
|-----|---------|
| `val/loss_lad_ctc` | LAD-CTC loss on clean validation images |
| `val/CAcc` | Character Accuracy `= 1 – CER` |
| `val/SAcc` | Sequence (exact-match) Accuracy |
| `val/LAcc` | Length Accuracy |
| `val/CER` | Character Error Rate |

The best `SAcc` checkpoint is saved to `<output-dir>/checkpoints/best.pt`
automatically.

Override the defaults:

```bash
# Larger validation set (30 %), evaluated every 5 epochs
python train.py --output-dir artifacts/run1 --val-split 0.3 --val-every 5

# Disable periodic validation (val-every larger than any phase's epoch count)
python train.py --output-dir artifacts/run1 --val-every 9999
```

---

## 5. Structural ablations

`experiments/ablation.py` steps through four variants spanning the
variable-length and KAN axes, warm-trains each in its native regime, evaluates on
a fixed held-out set, and emits Markdown comparison tables.

| Variant | Variable length | KAN | Model |
| --- | :---: | :---: | --- |
| Baseline | – | – | `DeepCaptcha` (fixed-length CNN) |
| Baseline + VarLen | yes | – | `VariableLengthDeepCaptcha` (CRNN + CTC) |
| Baseline + KAN | yes | partial | `VLKAN` without A-KAN attention |
| Full VL-KAN | yes | yes | `VLKAN` (pyramid + A-KAN blocks) |

**Run:**

```bash
python -m experiments.ablation \
  --train-steps 200 --eval-samples 512 --batch-size 64 \
  --output artifacts/ablation.md
```

**Tiny validation run (CPU):**

```bash
python -m experiments.ablation --smoke
```

The report (`artifacts/ablation.md`) contains a capability matrix plus a metrics
table with **CAcc**, **SAcc**, **LAcc**, **CER**, per-image **latency**, and
**throughput**.

---

## 6. Inference

`infer.py` loads a trained checkpoint and transcribes images using the
length-/boundary-aware beam search (greedy fallback via `--greedy`).

**Transcribe image files:**

```bash
python infer.py --checkpoint artifacts/run1/checkpoints/last.pt \
  --images "samples/*.png"
```

**Generate and transcribe synthetic samples (no input files):**

```bash
python infer.py --checkpoint artifacts/run1/checkpoints/last.pt --synthetic 8
```

Output lists `name  prediction  truth` per image and a latency summary
(mean / p95 / throughput).

**Programmatic loop:**

```python
import torch
from datasets.vocabulary import build_vocabulary
from evaluation.decoding import beam_ctc_decode
from infer import load_model, _synthetic_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab = build_vocabulary("alphanumeric")
model = load_model("artifacts/run1/checkpoints/last.pt", vocab.size, device)

images, truths = _synthetic_batch(8, seed=0)
with torch.no_grad():
    preds = beam_ctc_decode(model(images.to(device)), vocab)
print(list(zip(preds, truths)))
```

---

## 7. Metrics

`evaluation/metrics.py` provides the headline measures:

- **CAcc** — Character Accuracy, `1 − CER`, from corpus-level Levenshtein edit distance.
- **SAcc** — Sequence (exact-match) Accuracy.
- **LAcc** — Length Accuracy (predicted length equals ground-truth length).
- **CER** — aggregate Character Error Rate.

Latency/throughput are profiled with `Timer` and `LatencyTracker`
(mean / p50 / p95 / p99 and items/sec).

```python
from evaluation.metrics import evaluate_predictions
m = evaluate_predictions(["abc", "1234"], ["abc", "1235"])
print(m.as_dict())
```

---

## 8. Reproducibility

- Procedural data is seeded by `(base_seed, epoch, index)`; pass `--seed` to fix runs.
- Checkpoints persist model / optimizer / scheduler / loss / GradScaler / RNG
  state and the curriculum cursor for bit-faithful resume.
- Dependencies are locked in `requirements.txt` / `environment.yml`.
