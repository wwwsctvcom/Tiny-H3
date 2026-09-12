"""Numpy-only checks: no torch needed, so they run anywhere (including CPU-only boxes)."""

from __future__ import annotations

import numpy as np

from tiny_h3.data.synth import MOTIONS, generate_scene
from tiny_h3.rewards.prompt_match import prompt_match_reward
from tiny_h3.rewards.sync import av_sync_reward


def test_synth_reward_prefers_matching_prompt():
    scene = generate_scene(seed=11, size=96, frames=22, motion="swinging")
    matching, _ = prompt_match_reward(scene.frames, scene.wave, scene.fps, scene.sample_rate, scene.prompt)
    wrong = "a blue square pulsing on a green background, with steady beats at a fast tempo"
    mismatching, _ = prompt_match_reward(scene.frames, scene.wave, scene.fps, scene.sample_rate, wrong)
    assert matching >= mismatching
    assert av_sync_reward(scene.frames, scene.wave, scene.fps, scene.sample_rate) >= 0


def test_all_motions_have_audio():
    for seed, motion in enumerate(MOTIONS):
        scene = generate_scene(seed=seed, size=64, frames=22, motion=motion)
        assert float(np.sqrt(np.mean(scene.wave**2))) > 0.01, motion
        assert scene.frames.shape == (22, 64, 64, 3)
