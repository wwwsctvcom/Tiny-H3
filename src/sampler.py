"""Flow-matching samplers for Tiny-H3, in MiniMax-H3's own convention.

H3 parameterizes the flow as ``x_t = t * x0 + (1 - t) * noise`` with ``t = 1 - sigma`` and
the model predicts the **data-ward** velocity ``v = x0 - noise`` (so ``x0_hat = x_t + sigma * v``).
That is the opposite sign from the usual flow-match convention -- get it wrong and training
still converges to something that samples as noise.  Both the schedules and the Euler step
mirror ``MiniMaxH3Scheduler``; the SDE step with log-probability is the Flow-GRPO update
adapted from miles-diffusion (``miles/utils/sde_log_prob.py``) to this sign convention.

* :func:`build_schedule` -- exponential sigma shift (H3 uses 12.0 for video, 3.0 for audio).
* :func:`euler_step`     -- deterministic ODE step, used for the demo.
* :func:`sde_step`       -- one reverse-SDE step that also returns ``log p(next | current)``,
  which is what the GRPO ratio needs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

VIDEO_FLOW_SHIFT = 12.0
AUDIO_FLOW_SHIFT = 3.0


def build_schedule(num_steps: int, shift: float, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(sigmas, timesteps)``; ``num_steps`` model evaluations, sigmas ending at 0."""
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    base = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float32)
    sigmas = shift * base / (1.0 + (shift - 1.0) * base)
    sigmas = torch.unique_consecutive(sigmas)
    timesteps = 1.0 - sigmas[:-1]
    if device is not None:
        sigmas, timesteps = sigmas.to(device), timesteps.to(device)
    return sigmas, timesteps


def euler_step(velocity: torch.Tensor, sample: torch.Tensor, sigma: float, sigma_next: float) -> torch.Tensor:
    """One deterministic step: ``x0_hat = x + sigma*v`` then blend ``x_next = r*x + (1-r)*x0_hat``."""
    compute = torch.float32 if sample.dtype in (torch.float16, torch.bfloat16) else sample.dtype
    sample_c = sample.to(compute)
    denoised = sample_c + float(sigma) * velocity.to(compute)
    ratio = float(sigma_next) / float(sigma)
    return (ratio * sample_c + (1.0 - ratio) * denoised).to(sample.dtype)


@dataclass
class SDEStep:
    prev_sample: torch.Tensor
    log_prob: torch.Tensor  # (batch,)
    mean: torch.Tensor
    std_dev: torch.Tensor


def sde_transition_stats(
    velocity: torch.Tensor,
    sample: torch.Tensor,
    sigma: float,
    sigma_next: float,
    *,
    sigma_max: float,
    noise_level: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable ``(mean, std)`` of one H3 reverse-SDE transition."""
    sample_c = sample.float()
    velocity_c = velocity.float()
    dt = float(sigma_next) - float(sigma)
    if dt >= 0:
        raise ValueError(f"schedule must be decreasing: sigma={sigma}, sigma_next={sigma_next}")
    safe_sigma = sigma_max if abs(sigma - 1.0) < 1e-6 else sigma
    std_dev_t = math.sqrt(sigma / (1.0 - safe_sigma)) * noise_level
    view = (-1,) + (1,) * (sample_c.dim() - 1)
    std_dev_t_t = torch.full((sample_c.shape[0],), std_dev_t, device=sample_c.device, dtype=torch.float32).view(view)
    mean = (
        sample_c * (1.0 + std_dev_t_t**2 / (2.0 * sigma) * dt)
        - velocity_c * (1.0 + std_dev_t_t**2 * (1.0 - sigma) / (2.0 * sigma)) * dt
    )
    return mean, std_dev_t_t * math.sqrt(-dt)


def transition_log_prob(prev_sample: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Mean log-probability over non-batch dimensions, matching miles-diffusion/Flow-GRPO."""
    log_prob = (
        -((prev_sample.detach().float() - mean) ** 2) / (2.0 * std**2)
        - torch.log(std)
        - math.log(math.sqrt(2.0 * math.pi))
    )
    return log_prob.reshape(log_prob.shape[0], -1).mean(dim=1)


def sde_step(
    velocity: torch.Tensor,
    sample: torch.Tensor,
    sigma: float,
    sigma_next: float,
    *,
    sigma_max: float,
    noise_level: float,
    noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> SDEStep:
    """One reverse-SDE step with its log-probability (Flow-GRPO's exploration step).

    With ``noise_level=0`` this degenerates to :func:`euler_step`; the flow-matching GDP
    ``sigma`` schedule gives the diffusion coefficient ``std_dev = sqrt(sigma/(1-sigma)) * eta``
    and the drift picks up the score term, written here for H3's ``v = x0 - noise`` sign.
    """
    mean, std = sde_transition_stats(
        velocity, sample, sigma, sigma_next, sigma_max=sigma_max, noise_level=noise_level
    )
    if noise is None:
        noise = torch.randn(mean.shape, device=mean.device, dtype=torch.float32, generator=generator)
    prev_sample = mean + std * noise
    log_prob = transition_log_prob(prev_sample, mean, std)
    # Keep the pre-sqrt coefficient for KL, matching miles-diffusion's return contract.
    std_dev_t = std / math.sqrt(float(sigma) - float(sigma_next))
    return SDEStep(prev_sample=prev_sample.to(sample.dtype), log_prob=log_prob, mean=mean, std_dev=std_dev_t)


@dataclass
class SDEConfig:
    """Rollout knobs for Flow-GRPO."""

    noise_level: float = 0.7
    record: bool = True


@dataclass
class SampleResult:
    video_latents: torch.Tensor
    audio_latents: torch.Tensor
    trajectory: list[dict] = field(default_factory=list)  # one entry per denoising step


def packed_velocity(
    dit: torch.nn.Module,
    layout,
    text_embeds: torch.Tensor,
    video_latents: torch.Tensor,
    audio_latents: torch.Tensor,
    video_sigma: float,
    audio_sigma: float,
):
    """Run the DiT once and return velocities in **latent** shapes.

    The diffusers H3 transformer consumes/returns packed rows.  Keeping the row<->latent
    conversion at this boundary means every caller (SFT, ODE sampling, SDE rollout, GRPO)
    works with the intuitive ``(B,C,T,H,W)`` / ``(B,stereo,C,T)`` tensors.
    """
    from .packing import pack_audio_rows, pack_video_rows, unpack_audio_rows, unpatchify_video_latents

    batch = text_embeds.shape[0]
    video_rows = pack_video_rows(video_latents, layout.patch_size)
    audio_rows = torch.cat([pack_audio_rows(audio_latents[b]) for b in range(batch)], dim=0)

    timesteps, timestep_indices = layout.row_timesteps(1.0 - video_sigma, 1.0 - audio_sigma)
    out = dit(
        hidden_states=video_rows.to(text_embeds.dtype),
        audio_hidden_states=audio_rows.to(text_embeds.dtype),
        encoder_hidden_states=text_embeds,
        timestep=timesteps.to(text_embeds.device, text_embeds.dtype),
        timestep_indices=timestep_indices.to(text_embeds.device),
        token_tags=layout.token_tags.to(text_embeds.device),
        position_ids=layout.position_ids.to(text_embeds.device),
        video_indices=layout.video_indices.to(text_embeds.device),
        audio_indices=layout.audio_indices.to(text_embeds.device),
        text_indices=layout.text_indices.to(text_embeds.device),
        return_dict=True,
    )
    videos = []
    for b in range(batch):
        videos.append(
            unpatchify_video_latents(
                out.sample[b], layout.patch_size, layout.in_channels,
                layout.latent_frames, layout.latent_height, layout.latent_width,
            )[0]
        )
    video_velocity = torch.stack(videos)
    audio_velocity = torch.stack(
        [unpack_audio_rows(out.audio_sample[b : b + 1], layout.audio_channels, layout.num_audio_latents) for b in range(batch)]
    )
    return video_velocity, audio_velocity


@torch.no_grad()
def sample_av(
    dit: torch.nn.Module,
    layout,
    text_embeds: torch.Tensor,
    video_shape: tuple[int, int, int, int],
    audio_shape: tuple[int, int, int],
    *,
    num_steps: int = 24,
    video_shift: float = VIDEO_FLOW_SHIFT,
    audio_shift: float = AUDIO_FLOW_SHIFT,
    seed: int = 0,
    sde: SDEConfig | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    video_latents: torch.Tensor | None = None,
    audio_latents: torch.Tensor | None = None,
    progress: bool = False,
) -> SampleResult:
    """Text -> (video latents, audio latents).

    ODE sampling (the default) is diffusers' own ``MiniMaxH3Scheduler`` -- one instance per
    modality, because H3 uses shift 12 for video and 3 for audio.  Passing ``sde`` switches to
    the Flow-GRPO reverse-SDE sampler, which needs per-step sigmas and log-probabilities that
    the deterministic scheduler does not expose.
    """
    device = torch.device(device)
    gen_device = device.type if device.type == "cuda" else "cpu"
    gen = torch.Generator(device=gen_device).manual_seed(seed)
    if video_latents is None:
        video_latents = torch.randn(video_shape, generator=gen, device=device, dtype=dtype)
    if audio_latents is None:
        audio_latents = torch.randn(audio_shape, generator=gen, device=device, dtype=dtype)

    if sde is None:
        from diffusers import MiniMaxH3Scheduler

        scheduler_v = MiniMaxH3Scheduler(shift=video_shift)
        scheduler_a = MiniMaxH3Scheduler(shift=audio_shift)
        scheduler_v.set_timesteps(num_steps, device=device)
        scheduler_a.set_timesteps(num_steps, device=device)
        sigmas_v = scheduler_v.sigmas
        sigmas_a = scheduler_a.sigmas
    else:
        scheduler_v = scheduler_a = None
        sigmas_v, _ = build_schedule(num_steps, video_shift)
        sigmas_a, _ = build_schedule(num_steps, audio_shift)
    sigma_max_v = float(sigmas_v[1]) if len(sigmas_v) > 1 else 1.0
    sigma_max_a = float(sigmas_a[1]) if len(sigmas_a) > 1 else 1.0

    trajectory: list[dict] = []
    # Both schedules end at sigma 0: build_schedule gives num_steps+1 points for num_steps
    # evaluations, and MiniMaxH3Scheduler carries the terminal zero inside `num_steps`
    # (n points -> n-1 evaluations).  Walk sigma points, never a fixed count.
    step_count = min(len(sigmas_v), len(sigmas_a)) - 1
    step_iter = range(step_count)
    if progress:
        from tqdm.auto import tqdm

        step_iter = tqdm(step_iter, desc="denoise")
    for i in step_iter:
        sv, sv_next = float(sigmas_v[i]), float(sigmas_v[i + 1])
        sa, sa_next = float(sigmas_a[i]), float(sigmas_a[i + 1])
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype in (torch.float16, torch.bfloat16)):
            v_video, v_audio = packed_velocity(dit, layout, text_embeds, video_latents, audio_latents, sv, sa)

        if sde is None:
            video_latents = scheduler_v.step(v_video, scheduler_v.timesteps[i], video_latents, return_dict=False)[0]
            audio_latents = scheduler_a.step(v_audio, scheduler_a.timesteps[i], audio_latents, return_dict=False)[0]
        else:
            step_v = sde_step(v_video, video_latents, sv, sv_next, sigma_max=sigma_max_v,
                              noise_level=sde.noise_level, generator=gen)
            step_a = sde_step(v_audio, audio_latents, sa, sa_next, sigma_max=sigma_max_a,
                              noise_level=sde.noise_level, generator=gen)
            if sde.record:
                trajectory.append(
                    {
                        "video": {"latent": video_latents[0].detach().cpu(), "next_latent": step_v.prev_sample[0].detach().cpu(),
                                  "timestep": 1.0 - sv, "next_timestep": 1.0 - sv_next,
                                  "sigma": sv, "sigma_next": sv_next,
                                  "mean_old": step_v.mean[0].detach().cpu(),
                                  "log_prob": step_v.log_prob.detach().cpu()},
                        "audio": {"latent": audio_latents[0].detach().cpu(), "next_latent": step_a.prev_sample[0].detach().cpu(),
                                  "timestep": 1.0 - sa, "next_timestep": 1.0 - sa_next,
                                  "sigma": sa, "sigma_next": sa_next,
                                  "mean_old": step_a.mean[0].detach().cpu(),
                                  "log_prob": step_a.log_prob.detach().cpu()},
                    }
                )
            video_latents = step_v.prev_sample
            audio_latents = step_a.prev_sample

    return SampleResult(video_latents=video_latents, audio_latents=audio_latents, trajectory=trajectory)
