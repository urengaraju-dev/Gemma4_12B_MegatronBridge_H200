# Contributing

Thanks for your interest in improving Gemma-4-12B support on Megatron-Bridge.
This repo values **evidence over assertion** — every capability claim is backed by a runnable parity/smoke harness in `tests/`, compared against the HuggingFace reference on identical inputs. Please keep that bar.

## Ground rules

1. **Verify against the installed stack, not docs.** Megatron-Bridge 0.6.1 / Megatron-Core 0.19.1 / transformers 5.12.1 inside `nvcr.io/nvidia/nemo:26.08`. APIs drift between versions; check what is actually imported.
2. **Every behavioral change ships with a test.** If you add or fix something, add (or extend) a harness under `tests/` that a reviewer can run to reproduce your numbers.
3. **Report parity honestly.** State cosine / top-1 / max|Δ| and the exact command. If something is bf16-limited or single-GPU-only, say so — see `docs/DESIGN.md §4`.
4. **Never commit secrets or weights.** No `hf_…`, `glpat-…`, `github_pat_…` tokens, and no model checkpoints. `.gitignore` covers common cases; double-check `git diff --staged` before committing.

## Development loop

```bash
make setup           # one-time: container + weights
make phase0-parity   # text parity
make phase1-parity   # image-conditioned parity
make phase1-vision   # vision embedder bit-exactness
```

All targets run inside the container (repo mounted at `/workspace`). See the [Makefile](Makefile) for the full list (`make help`).

## Code style

- Python, PEP 8, 4-space indent. Keep modules small and single-purpose (mirror the existing `src/` split: bridge / projector / mask / VL wrapper).
- Match the surrounding code's idiom and comment density. Comments should explain *why* (especially the non-obvious Megatron/HF divergences), not restate the code.
- Keep the diff minimal and focused; unrelated cleanups belong in a separate change.

## Pull / merge requests

- Describe the change, the motivation, and the **before/after parity numbers** with the command used.
- Update `CHANGELOG.md` (Unreleased section) and, if the behavior or design changed, `docs/DESIGN.md`.
- One logical change per MR.

## Scope

This project tracks the three-phase plan (text → vision → audio) toward full `Gemma4UnifiedForConditionalGeneration` support and its eventual upstreaming into Megatron-Bridge's MIMO framework. The current productionization targets are listed in `CHANGELOG.md → Unreleased` and the README's "Known limitations / follow-ups".
