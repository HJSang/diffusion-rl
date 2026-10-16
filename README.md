# Diffusion RL

Papers, tutorials and code for reinforcement learning on diffusion and flow-matching models.

## Layout

```
papers/      one folder per paper: <arxiv-id>-<short-name>/ with the PDF and, when available, the LaTeX source
tutorials/   study guides with runnable examples
code/        upstream implementations as git submodules
```

## Papers

| arXiv | Title | Folder | Code |
| --- | --- | --- | --- |
| [2610.12572](https://arxiv.org/abs/2610.12572) | DiffusionART: Efficient, Stable and Scalable Reinforcement of Diffusion Models | [papers/2610.12572-diffusionart](papers/2610.12572-diffusionart) (PDF + LaTeX) | [code/DiffusionART](code/DiffusionART) |
| [2610.14136](https://arxiv.org/abs/2610.14136) | Correction Training for Diffusion Models | [papers/2610.14136-correction-training](papers/2610.14136-correction-training) (PDF) | — |

## Tutorials

### [DiffusionART study guide](tutorials/diffusionart)

A five-step path from LLM RL to the DiffusionART paper:

1. Flow matching, ODE/SDE sampling and classifier-free guidance
2. From LLM RL to diffusion RL: the denoising MDP, step log-probs, rewards, credit assignment
3. The baselines: DDPO, Flow-GRPO, DiffusionNFT, ReFL, Diffusion-DPO
4. The DiffusionART paper, section by section
5. Reading and running the DiffusionART code

Open `tutorials/diffusionart/DiffusionART-tutorial.html` in a browser for the full guide on one page, or `tutorials/diffusionart/site/step-N.html` for a single step.

The chapters are HTML fragments in `fragments/`, assembled by `build.py`:

```bash
python3 tutorials/diffusionart/build.py
```

`examples/stepN/` holds the toy scripts behind each chapter's code blocks, with their recorded outputs (`*.out`). They need PyTorch and run on CPU in seconds.

## Code

Clone with submodules to get the upstream implementations:

```bash
git clone --recurse-submodules git@github.com:HJSang/diffusion-rl.git
```

## Adding a paper

1. Create `papers/<arxiv-id>-<short-name>/` and put the PDF there (from `https://arxiv.org/pdf/<id>`; LaTeX source from `https://arxiv.org/src/<id>`, unpacked into `src/`).
2. If it has public code, add it with `git submodule add <url> code/<Name>`.
3. Add a row to the papers table above.
