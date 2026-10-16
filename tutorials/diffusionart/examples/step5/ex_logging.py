from dataclasses import dataclass, field

import torch
from visual_rl.algorithms import art_loss, art_scalar_covariance
from visual_rl.backends.tensor_fixture import TensorPolicy
from visual_rl.sampling import shifted_schedule
from visual_rl.training.objectives import ARTObjective, _indices, disjoint_windows
from visual_rl.training.rollout import collect_rollout
from visual_rl.training.update import accumulated_step


@dataclass
class LoggedARTObjective(ARTObjective):
    diagnostics: list = field(default_factory=list)

    def __call__(self, items):
        rows, slots, steps = _indices(items, self.cache)
        state = self.cache.states[rows, slots].to(self.device)
        sigma = self.cache.sigmas[steps].to(self.device)
        velocity = self.predict(state, sigma, rows.to(self.device))
        old = self.cache.old_velocities[rows, slots].to(self.device)
        dt = (self.cache.sigmas[steps + 1] - self.cache.sigmas[steps]).to(self.device)
        terms = art_loss(velocity, old, sigma, self.row_directions[rows].to(self.device),
                         self.row_magnitudes[rows].to(self.device), state=state,
                         ode_velocity_coefficient=dt,
                         restore_anchor_mask=self.row_anchors[rows].to(self.device), **self.options)
        self.diagnostics.append({k: terms[k].detach().cpu() for k in ("masked", "movement_score")})
        return terms["per_item_loss"]

    def pop_metrics(self):
        masked = torch.cat([d["masked"] for d in self.diagnostics]).float()
        score = torch.cat([d["movement_score"] for d in self.diagnostics])
        self.diagnostics.clear()
        return {"train/art/masked_fraction": masked.mean().item(),
                "train/art/movement_score_mean": score.mean().item()}


torch.manual_seed(42)
model = TensorPolicy()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
gen = torch.Generator().manual_seed(42)
p, k = 2, 4
groups = torch.arange(p).repeat_interleave(k); cond = groups.float() / p
sigmas = shifted_schedule(4, 1.0); selected = (0, 1, 2)
predict = lambda s, t, rows: model(s, t, cond[rows])
cache = collect_rollout(predict, torch.randn((p * k, 4), generator=gen), sigmas, selected=selected, microbatch=4)
rewards = -(cache.endpoints - cond[:, None]).square().mean(1)
st = art_scalar_covariance(cache.endpoints, rewards, groups, p)
obj = LoggedARTObjective(cache, predict, st["group_directions"][groups], st["group_guidance_magnitudes"][groups],
                         torch.zeros(p * k, dtype=torch.bool), torch.device("cpu"))
for items in disjoint_windows(p * k, len(selected)):
    m = accumulated_step(model, opt, items, obj, microbatch=3, denominator=len(items), max_grad_norm=1.0)
    m.update(obj.pop_metrics())
    print({key: round(v, 8) for key, v in m.items()})
