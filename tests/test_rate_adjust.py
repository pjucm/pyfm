#!/usr/bin/env python3
"""Tests for adaptive rate-adjust clamping logic."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from pjfm import FMRadio


def test_rate_adjust_clamp_fm_mode():
    assert FMRadio._clamp_rate_adjust(1.02, weather_mode=False) == FMRadio.RATE_ADJ_MAX
    assert FMRadio._clamp_rate_adjust(0.98, weather_mode=False) == FMRadio.RATE_ADJ_MIN
    assert FMRadio._clamp_rate_adjust(1.0, weather_mode=False) == 1.0


def test_rate_adjust_clamp_weather_mode_tighter():
    assert FMRadio._clamp_rate_adjust(1.02, weather_mode=True) == FMRadio.NBFM_RATE_ADJ_MAX
    assert FMRadio._clamp_rate_adjust(0.98, weather_mode=True) == FMRadio.NBFM_RATE_ADJ_MIN
    assert FMRadio._clamp_rate_adjust(1.0004, weather_mode=True) == 1.0004


def test_normalize_pll_kernel_mode():
    assert FMRadio._normalize_pll_kernel_mode("python") == "python"
    assert FMRadio._normalize_pll_kernel_mode("AUTO") == "auto"
    assert FMRadio._normalize_pll_kernel_mode("numba") == "numba"
    assert FMRadio._normalize_pll_kernel_mode("invalid") == "python"


def test_iq_loss_flush_suppressed_during_startup_grace():
    now_s = 10.0
    stream_start_s = 9.0  # within default 2.0 s startup grace
    should_flush = FMRadio._should_flush_iq_loss(
        recent_loss=FMRadio.IQ_LOSS_FLUSH_THRESHOLD,
        now_s=now_s,
        last_flush_s=0.0,
        stream_start_s=stream_start_s,
    )
    assert not should_flush


def test_iq_loss_flush_requires_threshold_and_cooldown():
    now_s = 10.0
    stream_start_s = 0.0  # disable startup grace for this case

    assert not FMRadio._should_flush_iq_loss(
        recent_loss=FMRadio.IQ_LOSS_FLUSH_THRESHOLD - 1,
        now_s=now_s,
        last_flush_s=0.0,
        stream_start_s=stream_start_s,
    )
    assert not FMRadio._should_flush_iq_loss(
        recent_loss=FMRadio.IQ_LOSS_FLUSH_THRESHOLD,
        now_s=now_s,
        last_flush_s=now_s - (FMRadio.IQ_LOSS_FLUSH_COOLDOWN_S * 0.5),
        stream_start_s=stream_start_s,
    )
    assert FMRadio._should_flush_iq_loss(
        recent_loss=FMRadio.IQ_LOSS_FLUSH_THRESHOLD,
        now_s=now_s,
        last_flush_s=now_s - (FMRadio.IQ_LOSS_FLUSH_COOLDOWN_S + 0.1),
        stream_start_s=stream_start_s,
    )
