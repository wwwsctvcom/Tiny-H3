# Contributing to Tiny-H3

Thanks for your interest in improving Tiny-H3! This document covers the basics;
the design rationale lives in [`docs/design.md`](docs/design.md).

## Development setup

```bash
git clone <this repo> && cd tiny-h3
bash scripts/setup_env.sh        # deps + pinned diffusers + pip install -e .
source scripts/env.sh            # env vars (HF mirror, data paths)
bash scripts/download_assets.sh  # H3 VAEs + Qwen3-0.6B (~12 GB, GPU runs need them)
```

The package sources live flat under `src/` and keep the import name `tiny_h3`
(wired up via `package-dir` in `pyproject.toml`). `pip install -e .` is what puts
`tiny_h3` on the path; `tools/*.py` also work from a bare checkout.

## Running the tests

```bash
pytest -q        # pure-CPU unit tests (packing, scheduler, data generator)
```

GPU acceptance (one RTX 5090 or similar, ~30 GB):

```bash
python tools/reconstruct_data.py --data-dir $TINY_H3_DATA/synth --out outputs/reconstruction --count 4
python tools/smoke_test.py --latents $TINY_H3_DATA/synth/latents --with-vae --preset smoke
```

## Conventions

* Keep the H3 contract intact: latent shapes (`in_channels=24`, `audio_in_channels=32`,
  patch `(1,2,2)`), the packed row order and tags, `v = x0 - noise`, and the VAE
  normalisation in `src/vae.py`. If you must touch one of these, say so loudly in the PR.
* `scripts/pins.env` pins the diffusers commit the project is tested against.
* Format with `ruff` (`line-length = 120`, config in `pyproject.toml`).
* Pull requests: one logical change per PR, include the command you verified it with.
