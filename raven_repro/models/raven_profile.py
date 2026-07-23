from __future__ import annotations

from typing import Dict, Iterable, Optional

from .raven import PaperRAVENConfig


PAPER_PARAMETER_TARGETS = {
    "channel_ssm": 5_000,
    "antenna_mixer": 65_000,
    "chirp_ssm": 514_000,
    "decoder": 927_000,
}

PAPER_MAC_TARGETS = {
    "channel_ssm": 516_000_000,
    "antenna_mixer": 186_000_000,
    "chirp_ssm": 60_000_000,
    "decoder": 261_000_000,
}


def _mamba_macs(
    tokens: int,
    d_model: int,
    d_state: int,
    d_conv: int,
    expand: int,
) -> int:
    """Dense projections, depthwise convolution, and selective scan MACs."""

    inner = d_model * expand
    dt_rank = max(1, (d_model + 15) // 16)
    per_token = (
        d_model * (2 * inner)
        + inner * d_conv
        + inner * (dt_rank + 2 * d_state)
        + dt_rank * inner
        + inner * d_model
        + inner * (d_state + 1)
    )
    return int(tokens * per_token)


def _conv2d_macs(
    in_channels: int,
    out_channels: int,
    height: int,
    width: int,
    kernel_size: int = 3,
) -> int:
    return int(height * width * out_channels * in_channels * kernel_size * kernel_size)


def _active_decode_lengths(
    num_chirps: int,
    prefix_lengths: Optional[Iterable[int]],
) -> list[int]:
    if prefix_lengths is None:
        return [num_chirps]
    prefixes = {
        min(int(length), num_chirps)
        for length in prefix_lengths
        if int(length) > 0
    }
    prefixes.add(num_chirps)
    return sorted(prefixes)


def estimate_paper_raven_macs(
    cfg: PaperRAVENConfig,
    prefix_lengths: Optional[Iterable[int]] = None,
) -> Dict[str, int]:
    """CPU-free analytical MAC count for one input frame.

    The convention counts multiplications in Linear/Conv/attention operations
    and the recurrent selective scan. LayerNorm, activations, interpolation,
    softmax, and bias additions are excluded. This is intentionally explicit
    because the paper does not disclose its profiler or Mamba counting rules.
    """

    fast_positions = cfg.num_chirps * cfg.num_samples * cfg.num_rx
    channel_mamba = cfg.fast_time_layers * _mamba_macs(
        fast_positions,
        d_model=2,
        d_state=cfg.ssm_state_dim,
        d_conv=cfg.ssm_conv_kernel,
        expand=cfg.ssm_expansion,
    )
    channel_pool = fast_positions * 2  # sample-axis mean of I and Q
    channel_ssm = channel_mamba + channel_pool

    d = cfg.mixer_dim
    antenna_input = cfg.num_chirps * cfg.num_rx * 2 * d
    antenna_projections = cfg.num_chirps * (
        cfg.num_tx * d * d
        + cfg.num_rx * d * d
        + cfg.num_rx * d * d
        + cfg.num_tx * d * d
    )
    antenna_attention = cfg.num_chirps * 2 * cfg.num_tx * cfg.num_rx * d
    ffn_hidden = d * cfg.mixer_ffn_expansion
    antenna_ffn = cfg.num_chirps * cfg.num_tx * (
        d * ffn_hidden + ffn_hidden * d
    )
    antenna_pair = cfg.num_chirps * cfg.num_rx * cfg.num_tx * (2 * d) * 2
    antenna_mixer = (
        antenna_input
        + antenna_projections
        + antenna_attention
        + antenna_ffn
        + antenna_pair
    )

    virtual_dim = 2 * cfg.num_rx * cfg.num_tx
    chirp_reduce = cfg.num_chirps * virtual_dim * cfg.chirp_dim
    chirp_pre = cfg.num_chirps * cfg.chirp_dim * cfg.chirp_dim
    chirp_mamba = cfg.slow_time_layers * _mamba_macs(
        cfg.num_chirps,
        d_model=cfg.chirp_dim,
        d_state=cfg.ssm_state_dim,
        d_conv=cfg.ssm_conv_kernel,
        expand=cfg.ssm_expansion,
    )
    chirp_ssm = chirp_reduce + chirp_pre + chirp_mamba

    height, width = cfg.bev_grid
    spatial_cells = height * width
    det_projection_macs = cfg.num_chirps * cfg.chirp_dim * spatial_cells
    seg_projection_macs = cfg.num_chirps * cfg.chirp_dim * spatial_cells
    projection_macs = det_projection_macs + seg_projection_macs

    tail = cfg.decoder_channels // 2
    det_height, det_width = cfg.detection_output_size
    mid_height, mid_width = det_height // 2, det_width // 2
    det_cnn = _conv2d_macs(
        cfg.det_temporal_bins,
        cfg.decoder_channels,
        height,
        width,
    )
    det_cnn += _conv2d_macs(
        cfg.decoder_channels,
        cfg.decoder_channels,
        height,
        width,
    )
    det_cnn += _conv2d_macs(
        cfg.decoder_channels,
        tail,
        mid_height,
        mid_width,
    )
    det_cnn += _conv2d_macs(tail, tail, det_height, det_width)
    det_cnn += det_height * det_width * tail * (1 + cfg.regression_channels)

    seg_cnn = _conv2d_macs(
        cfg.seg_temporal_bins,
        cfg.decoder_channels,
        height,
        width,
    )
    seg_cnn += _conv2d_macs(
        cfg.decoder_channels,
        cfg.decoder_channels,
        mid_height,
        mid_width,
    )
    seg_cnn += _conv2d_macs(
        cfg.decoder_channels,
        tail,
        mid_height,
        mid_width,
    )
    seg_cnn += _conv2d_macs(tail, tail, det_height, det_width)
    seg_cnn += det_height * det_width * tail

    decode_lengths = _active_decode_lengths(cfg.num_chirps, prefix_lengths)
    pool_macs = sum(2 * length * spatial_cells for length in decode_lengths)
    decoder = projection_macs + pool_macs + len(decode_lengths) * (det_cnn + seg_cnn)

    profile = {
        "channel_ssm": int(channel_ssm),
        "sample_ssm": 0,
        "antenna_mixer": int(antenna_mixer),
        "chirp_ssm": int(chirp_ssm),
        "projection": int(projection_macs),
        "decoder": int(decoder),
        "other": 0,
        "decoder_projection": int(projection_macs),
        "decoder_pool": int(pool_macs),
        "decoder_cnn": int(len(decode_lengths) * (det_cnn + seg_cnn)),
        "decode_passes": len(decode_lengths),
        "detail.channel_ssm.mamba": int(channel_mamba),
        "detail.channel_ssm.pool": int(channel_pool),
        "detail.antenna_mixer.input": int(antenna_input),
        "detail.antenna_mixer.qkvo": int(antenna_projections),
        "detail.antenna_mixer.attention_matmul": int(antenna_attention),
        "detail.antenna_mixer.ffn": int(antenna_ffn),
        "detail.antenna_mixer.pair": int(antenna_pair),
        "detail.chirp_ssm.Wred": int(chirp_reduce),
        "detail.chirp_ssm.Wpre": int(chirp_pre),
        "detail.chirp_ssm.mamba": int(chirp_mamba),
        "detail.decoder.det_projection": int(det_projection_macs),
        "detail.decoder.seg_projection": int(seg_projection_macs),
        "detail.decoder.pool": int(pool_macs),
        "detail.decoder.detection_cnn": int(len(decode_lengths) * det_cnn),
        "detail.decoder.segmentation_cnn": int(len(decode_lengths) * seg_cnn),
    }
    profile["total"] = sum(profile[name] for name in (
        "channel_ssm",
        "antenna_mixer",
        "chirp_ssm",
        "decoder",
    ))
    return profile

