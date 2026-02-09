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
