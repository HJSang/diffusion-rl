import torch
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16).to("cuda")
w = 4.5                       # CFG scale; the paper's SD3.5 RL rollouts use w = 1 (no CFG)

# 1) prompt c -> embeddings: CLIP-L + CLIP-G (tokens + pooled) and T5-XXL (tokens).
#    With CFG, the "negative" embeddings come from the empty prompt.
pos, neg, pooled_pos, pooled_neg = pipe.encode_prompt(
    prompt="a red cube on top of a blue sphere", prompt_2=None, prompt_3=None,
    device="cuda", do_classifier_free_guidance=True, negative_prompt="")

# 2) shifted sigma grid from 1 (noise) down to 0 (data); len = steps + 1
sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
sched.set_timesteps(28)
sigmas = sched.sigmas.to("cuda")

# 3) x_1 = pure noise in the normalized VAE latent space: 16 x 512/8 x 512/8
x = torch.randn(1, 16, 64, 64, device="cuda", dtype=torch.bfloat16)
for s, s_next in zip(sigmas[:-1], sigmas[1:]):
    out = pipe.transformer(
        hidden_states=torch.cat([x, x]),                  # [uncond; cond] batch
        timestep=(s * 1000).expand(2),                    # model time = sigma * 1000
        encoder_hidden_states=torch.cat([neg, pos]),
        pooled_projections=torch.cat([pooled_neg, pooled_pos]),
        return_dict=False)[0]
    v_unc, v_cond = out.chunk(2)
    v_sd3 = v_unc + w * (v_cond - v_unc)                  # CFG; SD3 sign: v_sd3 ~ eps - x0
    x = x + (s_next - s) * v_sd3                          # = x_t + Delta * v, with v = -v_sd3

# 4) undo the latent normalization, then decode to pixels in [-1, 1]
cfg = pipe.vae.config
image = pipe.vae.decode(x / cfg.scaling_factor + cfg.shift_factor, return_dict=False)[0]
