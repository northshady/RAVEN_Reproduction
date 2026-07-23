from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except Exception:  # pragma: no cover - depends on the training environment.
    Mamba = None


@dataclass
class PaperRAVENConfig:
    """RAVEN architecture parameters, separating reported and inferred values.

    The paper explicitly reports the antenna-mixer and Mamba settings and the
    32x56 projection grid. It does not disclose ``chirp_dim``, ``T_det``,
    ``T_seg``, or the decoder width. Their defaults below are inferred from the
    supplementary parameter budget and the two released SSMRadNet prototypes.
    """

    architecture_version: str = "paper_equations_v1"
    num_chirps: int = 256
    num_samples: int = 512
    num_rx: int = 16
    num_tx: int = 12

    mixer_dim: int = 64
    mixer_heads: int = 8
    mixer_ffn_expansion: int = 4
    mixer_embedding_std: float = 1.0
    mixer_projection_bias: bool = False

    chirp_dim: int = 234
    ssm_state_dim: int = 16
    ssm_conv_kernel: int = 4
    ssm_expansion: int = 2
    fast_time_layers: int = 1
    slow_time_layers: int = 1
    backbone_projection_bias: bool = False

    bev_grid: Tuple[int, int] = (32, 56)
    det_temporal_bins: int = 32
    seg_temporal_bins: int = 16
    decoder_channels: int = 16
    decoder_conv_bias: bool = False
    spatial_projection_bias: bool = True
    regression_channels: int = 2
    detection_prior_probability: Optional[float] = None

    detection_output_size: Tuple[int, int] = (128, 224)
    segmentation_output_size: Tuple[int, int] = (256, 224)
    input_layout: str = "chirp_sample_channel"
    input_transform: str = "none"
    ssm_backend: str = "auto"

    def __post_init__(self) -> None:
        self.bev_grid = tuple(int(value) for value in self.bev_grid)
        self.detection_output_size = tuple(int(value) for value in self.detection_output_size)
        self.segmentation_output_size = tuple(int(value) for value in self.segmentation_output_size)
        if self.architecture_version != "paper_equations_v1":
            raise ValueError(
                f"Unsupported paper RAVEN architecture {self.architecture_version!r}."
            )
        if self.mixer_dim % self.mixer_heads:
            raise ValueError("mixer_dim must be divisible by mixer_heads.")
        if self.det_temporal_bins <= 0 or self.seg_temporal_bins <= 0:
            raise ValueError("T_det and T_seg must be positive.")
        if self.decoder_channels < 2 or self.decoder_channels % 2:
            raise ValueError("decoder_channels must be an even integer of at least two.")
        if self.input_layout not in {"chirp_sample_channel", "sample_chirp_channel"}:
            raise ValueError(f"Unsupported input layout {self.input_layout!r}.")
        if self.input_transform != "none":
            raise ValueError("The paper-equation model accepts raw ADC and has no model-side FFT.")
        if self.ssm_backend not in {"auto", "mamba", "fallback"}:
            raise ValueError("ssm_backend must be 'auto', 'mamba', or 'fallback'.")


class PaperMambaBlock(nn.Module):
    """Direct Mamba transform used by the paper equations.

    Unlike the two SSMRadNet prototype files, this wrapper does not append a
    second output projection, residual connection, normalization, or dropout.
    None of those operations is specified for RAVEN's SSM blocks.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        use_mamba = backend != "fallback" and Mamba is not None
        if backend == "mamba" and Mamba is None:
            raise RuntimeError("ssm_backend='mamba' requires the mamba_ssm package.")
        self.uses_mamba = use_mamba
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        if use_mamba:
            self.core = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            return

        # Shape-compatible CPU fallback for tests. It mirrors Mamba's parameter
        # families but is not a numerically equivalent selective scan.
        inner = d_model * expand
        dt_rank = max(1, (d_model + 15) // 16)
        self.inner = inner
        self.in_proj = nn.Linear(d_model, 2 * inner, bias=False)
        self.depthwise_conv = nn.Conv1d(
            inner,
            inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=inner,
            bias=True,
        )
        self.x_proj = nn.Linear(inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, inner, bias=True)
        self.A_log = nn.Parameter(torch.zeros(inner, d_state))
        self.D = nn.Parameter(torch.ones(inner))
        self.out_proj = nn.Linear(inner, d_model, bias=False)
        self.A_log._no_weight_decay = True
        self.D._no_weight_decay = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.uses_mamba:
            return self.core(x)

        value, gate = self.in_proj(x).chunk(2, dim=-1)
        sequence_length = value.shape[1]
        value = self.depthwise_conv(value.transpose(1, 2))
        value = value[..., :sequence_length].transpose(1, 2)
        value = F.silu(value)
        dt, state_b, state_c = torch.split(
            self.x_proj(value),
            [self.dt_proj.in_features, self.d_state, self.d_state],
            dim=-1,
        )
        dt = torch.sigmoid(self.dt_proj(dt))
        state = torch.tanh(state_b).mean(dim=-1, keepdim=True)
        state = state * torch.sigmoid(state_c).mean(dim=-1, keepdim=True)
        decay = torch.exp(-torch.exp(self.A_log).mean(dim=-1)).view(1, 1, self.inner)
        value = value * dt * decay + value * self.D.view(1, 1, -1)
        value = value * torch.sigmoid(gate) * (1.0 + state)
        return self.out_proj(value)


class PerRXFastTimeSSM(nn.Module):
    """Sixteen independent I/Q SSMs followed by sample-axis average pooling."""

    def __init__(self, cfg: PaperRAVENConfig) -> None:
        super().__init__()
        self.num_rx = cfg.num_rx
        self.rx_ssms = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        PaperMambaBlock(
                            d_model=2,
                            d_state=cfg.ssm_state_dim,
                            d_conv=cfg.ssm_conv_kernel,
                            expand=cfg.ssm_expansion,
                            backend=cfg.ssm_backend,
                        )
                        for _ in range(cfg.fast_time_layers)
                    ]
                )
                for _ in range(cfg.num_rx)
            ]
        )

    def forward(self, adc: torch.Tensor) -> torch.Tensor:
        # adc: [B, Nc, Ns, 2*Nrx], all real RX channels followed by imaginary.
        batch, num_chirps, num_samples, channels = adc.shape
        if channels != 2 * self.num_rx:
            raise ValueError(f"Expected {2 * self.num_rx} I/Q channels, got {channels}.")

        real = adc[..., : self.num_rx]
        imag = adc[..., self.num_rx :]
        per_rx = torch.stack((real, imag), dim=-1).permute(0, 1, 3, 2, 4)

        tokens: List[torch.Tensor] = []
        for rx_index, encoder in enumerate(self.rx_ssms):
            sequence = per_rx[:, :, rx_index].reshape(batch * num_chirps, num_samples, 2)
            encoded = encoder(sequence)
            tokens.append(encoded.mean(dim=1).reshape(batch, num_chirps, 2))
        return torch.stack(tokens, dim=2)  # [B, Nc, Nrx, 2]


class TXQueryAntennaMixer(nn.Module):
    """Paper cross-attention: TX queries attend to per-chirp RX tokens."""

    def __init__(self, cfg: PaperRAVENConfig) -> None:
        super().__init__()
        self.num_rx = cfg.num_rx
        self.num_tx = cfg.num_tx
        self.dim = cfg.mixer_dim

        self.rx_projection = nn.Linear(2, self.dim, bias=cfg.mixer_projection_bias)
        self.rx_embedding = nn.Parameter(torch.empty(self.num_rx, self.dim))
        self.tx_queries = nn.Parameter(torch.empty(self.num_tx, self.dim))
        nn.init.normal_(self.rx_embedding, mean=0.0, std=cfg.mixer_embedding_std)
        nn.init.normal_(self.tx_queries, mean=0.0, std=cfg.mixer_embedding_std)

        self.query_norm = nn.LayerNorm(self.dim)
        self.key_norm = nn.LayerNorm(self.dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.dim,
            num_heads=cfg.mixer_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(self.dim)
        hidden = self.dim * cfg.mixer_ffn_expansion
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.dim),
        )
        self.pair_projection = nn.Linear(
            2 * self.dim,
            2,
            bias=cfg.mixer_projection_bias,
        )
        self.output_norm = nn.LayerNorm(2 * self.num_rx * self.num_tx)

    def forward(self, rx_tokens: torch.Tensor) -> torch.Tensor:
        batch, num_chirps, num_rx, token_dim = rx_tokens.shape
        if num_rx != self.num_rx or token_dim != 2:
            raise ValueError(
                f"Expected RX tokens [B,Nc,{self.num_rx},2], got {tuple(rx_tokens.shape)}."
            )

        flat = rx_tokens.reshape(batch * num_chirps, num_rx, 2)
        rx = self.rx_projection(flat) + self.rx_embedding.unsqueeze(0)
        queries = self.tx_queries.unsqueeze(0).expand(batch * num_chirps, -1, -1)
        attended, _ = self.attention(
            self.query_norm(queries),
            self.key_norm(rx),
            rx,
            need_weights=False,
        )
        tx = queries + attended
        tx = tx + self.ffn(self.ffn_norm(tx))

        rx_pairs = rx.unsqueeze(2).expand(-1, self.num_rx, self.num_tx, -1)
        tx_pairs = tx.unsqueeze(1).expand(-1, self.num_rx, self.num_tx, -1)
        pair_features = self.pair_projection(torch.cat((rx_pairs, tx_pairs), dim=-1))
        virtual = pair_features.reshape(batch, num_chirps, 2 * self.num_rx * self.num_tx)
        return self.output_norm(virtual)


class ChirpSSMBackbone(nn.Module):
    """Virtual-MIMO reduction, pre-projection, and causal chirp-wise SSM."""

    def __init__(self, cfg: PaperRAVENConfig) -> None:
        super().__init__()
        virtual_dim = 2 * cfg.num_rx * cfg.num_tx
        self.reduce = nn.Linear(
            virtual_dim,
            cfg.chirp_dim,
            bias=cfg.backbone_projection_bias,
        )
        self.pre = nn.Linear(
            cfg.chirp_dim,
            cfg.chirp_dim,
            bias=cfg.backbone_projection_bias,
        )
        self.ssm_layers = nn.ModuleList(
            [
                PaperMambaBlock(
                    d_model=cfg.chirp_dim,
                    d_state=cfg.ssm_state_dim,
                    d_conv=cfg.ssm_conv_kernel,
                    expand=cfg.ssm_expansion,
                    backend=cfg.ssm_backend,
                )
                for _ in range(cfg.slow_time_layers)
            ]
        )

    def forward(self, virtual_features: torch.Tensor) -> torch.Tensor:
        chirp_features = F.silu(self.reduce(virtual_features))
        chirp_features = F.silu(self.pre(chirp_features))
        for ssm in self.ssm_layers:
            chirp_features = ssm(chirp_features)
        return chirp_features


class SequenceToBEV(nn.Module):
    """Conv1D D->H*W, chirp AdaptiveAvgPool, then T x H x W reshape."""

    def __init__(
        self,
        chirp_dim: int,
        grid: Tuple[int, int],
        temporal_bins: int,
        bias: bool,
    ) -> None:
        super().__init__()
        self.grid = tuple(grid)
        self.temporal_bins = temporal_bins
        height, width = self.grid
        self.projection = nn.Conv1d(chirp_dim, height * width, kernel_size=1, bias=bias)
        self.pool = nn.AdaptiveAvgPool1d(temporal_bins)

    def project(self, chirp_features: torch.Tensor) -> torch.Tensor:
        return self.projection(chirp_features.transpose(1, 2))

    def to_grid(self, projected: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        if length is not None:
            active = max(1, min(int(length), projected.shape[-1]))
            projected = projected[..., :active]
        pooled = self.pool(projected)
        height, width = self.grid
        return pooled.transpose(1, 2).reshape(
            projected.shape[0],
            self.temporal_bins,
            height,
            width,
        )


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class ConvNormSiLU(nn.Module):
    """The Conv-LN-SiLU order stated in the RAVEN paper."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=bias,
        )
        self.norm = LayerNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.norm(self.conv(x)))


class PaperDetectionDecoder(nn.Module):
    def __init__(self, cfg: PaperRAVENConfig) -> None:
        super().__init__()
        width = cfg.decoder_channels
        tail = width // 2
        self.coarse1 = ConvNormSiLU(cfg.det_temporal_bins, width, cfg.decoder_conv_bias)
        self.coarse2 = ConvNormSiLU(width, width, cfg.decoder_conv_bias)
        self.mid = ConvNormSiLU(width, tail, cfg.decoder_conv_bias)
        self.fine = ConvNormSiLU(tail, tail, cfg.decoder_conv_bias)
        self.output_size = cfg.detection_output_size
        self.cls_head = nn.Conv2d(tail, 1, kernel_size=1)
        self.reg_head = nn.Conv2d(tail, cfg.regression_channels, kernel_size=1)
        if cfg.detection_prior_probability is not None:
            probability = float(cfg.detection_prior_probability)
            if not 0.0 < probability < 1.0:
                raise ValueError("detection_prior_probability must be between zero and one.")
            prior = torch.log(torch.tensor(probability / (1.0 - probability)))
            nn.init.constant_(self.cls_head.bias, float(prior))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.coarse2(self.coarse1(x))
        mid_size = tuple(size // 2 for size in self.output_size)
        x = F.interpolate(x, size=mid_size, mode="bilinear", align_corners=False)
        x = self.mid(x)
        x = F.interpolate(x, size=self.output_size, mode="bilinear", align_corners=False)
        x = self.fine(x)
        return torch.cat((torch.sigmoid(self.cls_head(x)), self.reg_head(x)), dim=1)


class PaperSegmentationDecoder(nn.Module):
    def __init__(self, cfg: PaperRAVENConfig) -> None:
        super().__init__()
        width = cfg.decoder_channels
        tail = width // 2
        self.coarse = ConvNormSiLU(cfg.seg_temporal_bins, width, cfg.decoder_conv_bias)
        self.mid1 = ConvNormSiLU(width, width, cfg.decoder_conv_bias)
        self.mid2 = ConvNormSiLU(width, tail, cfg.decoder_conv_bias)
        self.fine = ConvNormSiLU(tail, tail, cfg.decoder_conv_bias)
        self.feature_size = cfg.detection_output_size
        self.output_size = cfg.segmentation_output_size
        self.head = nn.Conv2d(tail, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.coarse(x)
        mid_size = tuple(size // 2 for size in self.feature_size)
        x = F.interpolate(x, size=mid_size, mode="bilinear", align_corners=False)
        x = self.mid2(self.mid1(x))
        x = F.interpolate(x, size=self.feature_size, mode="bilinear", align_corners=False)
        logits = self.head(self.fine(x))
        return F.interpolate(logits, size=self.output_size, mode="bilinear", align_corners=False)


class PaperRAVEN(nn.Module):
    """Clean RAVEN implementation matching Sections 3.2.1-3.2.4."""

    profile_family = "raven"
    profile_includes_scan = True
    profile_group_order = ("channel_ssm", "antenna_mixer", "chirp_ssm", "decoder")
    published_parameter_count = 1_511_000
    published_macs = 1_023_000_000
    mac_group_roots = {
        "fast_time": "channel_ssm",
        "mixer": "antenna_mixer",
        "chirp_backbone": "chirp_ssm",
        "det_projection": "decoder",
        "seg_projection": "decoder",
        "det_decoder": "decoder",
        "seg_decoder": "decoder",
    }

    def __init__(self, config: Optional[PaperRAVENConfig | Dict] = None, **kwargs) -> None:
        super().__init__()
        if config is None:
            cfg = PaperRAVENConfig(**kwargs)
        elif isinstance(config, dict):
            values = {**config, **kwargs}
            values.pop("family", None)
            cfg = PaperRAVENConfig(**values)
        else:
            cfg = config
        self.cfg = cfg

        self.fast_time = PerRXFastTimeSSM(cfg)
        self.mixer = TXQueryAntennaMixer(cfg)
        self.chirp_backbone = ChirpSSMBackbone(cfg)
        self.det_projection = SequenceToBEV(
            cfg.chirp_dim,
            cfg.bev_grid,
            cfg.det_temporal_bins,
            cfg.spatial_projection_bias,
        )
        self.seg_projection = SequenceToBEV(
            cfg.chirp_dim,
            cfg.bev_grid,
            cfg.seg_temporal_bins,
            cfg.spatial_projection_bias,
        )
        self.det_decoder = PaperDetectionDecoder(cfg)
        self.seg_decoder = PaperSegmentationDecoder(cfg)

    @property
    def architecture_assumptions(self) -> Dict[str, str]:
        return {
            "mixer_dim_heads_ffn": "reported in RAVEN supplement Table 2",
            "mamba_state_conv_expand": "reported in RAVEN supplement Table 2",
            "bev_grid": "reported in RAVEN supplement Table 2",
            "chirp_dim": "not reported; D=234 inferred from the 0.514M chirp-block budget",
            "temporal_bins": "not reported; T_det=32 and T_seg=16 taken from the supplied antenna-mixer prototype",
            "decoder_channels": "not reported; width=16 taken from the supplied antenna-mixer prototype",
            "embedding_init": "N(0,1) reported in the supplement; applied to RX embeddings and TX queries",
            "adc_amplitude": "unscaled raw ADC counts; no model-side amplitude transform",
        }

    def ssm_style_description(self) -> str:
        return "direct Mamba output; no external norm, residual, activation, projection, or dropout"

    def _normalize_layout(self, adc: torch.Tensor) -> torch.Tensor:
        if adc.ndim != 4:
            raise ValueError(f"Expected a 4D ADC tensor, got {tuple(adc.shape)}.")
        if self.cfg.input_layout == "sample_chirp_channel":
            adc = adc.permute(0, 2, 1, 3).contiguous()
        expected = (self.cfg.num_chirps, self.cfg.num_samples, 2 * self.cfg.num_rx)
        if tuple(adc.shape[1:]) != expected:
            raise ValueError(f"Expected ADC shape [B,{expected}], got {tuple(adc.shape)}.")
        return adc

    def encode(self, adc: torch.Tensor) -> torch.Tensor:
        adc = self._normalize_layout(adc)
        rx_tokens = self.fast_time(adc)
        virtual_features = self.mixer(rx_tokens)
        return self.chirp_backbone(virtual_features)

    def decode_projected(
        self,
        det_projected: torch.Tensor,
        seg_projected: torch.Tensor,
        length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        det_grid = self.det_projection.to_grid(det_projected, length=length)
        seg_grid = self.seg_projection.to_grid(seg_projected, length=length)
        return {
            "Detection": self.det_decoder(det_grid),
            "Segmentation": self.seg_decoder(seg_grid),
        }

    def decode(self, chirp_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.decode_projected(
            self.det_projection.project(chirp_features),
            self.seg_projection.project(chirp_features),
        )

    @staticmethod
    def _parameter_count(modules: Iterable[nn.Module]) -> int:
        seen = set()
        count = 0
        for module in modules:
            for parameter in module.parameters():
                if id(parameter) not in seen:
                    seen.add(id(parameter))
                    count += parameter.numel()
        return count

    def logical_parameter_counts(self) -> Dict[str, int]:
        groups = {
            "channel_ssm": [self.fast_time],
            "antenna_mixer": [self.mixer],
            "chirp_ssm": [self.chirp_backbone],
            "decoder": [
                self.det_projection,
                self.seg_projection,
                self.det_decoder,
                self.seg_decoder,
            ],
        }
        return {name: self._parameter_count(modules) for name, modules in groups.items()}

    def detailed_parameter_counts(self) -> Dict[str, int]:
        mixer_input = self._parameter_count([self.mixer.rx_projection])
        mixer_input += self.mixer.rx_embedding.numel() + self.mixer.tx_queries.numel()
        return {
            "channel_ssm.rx_mambas": self._parameter_count([self.fast_time.rx_ssms]),
            "antenna_mixer.input_embeddings": mixer_input,
            "antenna_mixer.attention_norms": self._parameter_count(
                [self.mixer.query_norm, self.mixer.key_norm, self.mixer.attention]
            ),
            "antenna_mixer.ffn": self._parameter_count(
                [self.mixer.ffn_norm, self.mixer.ffn]
            ),
            "antenna_mixer.pair_output": self._parameter_count(
                [self.mixer.pair_projection, self.mixer.output_norm]
            ),
            "chirp_ssm.Wred": self._parameter_count([self.chirp_backbone.reduce]),
            "chirp_ssm.Wpre": self._parameter_count([self.chirp_backbone.pre]),
            "chirp_ssm.mamba": self._parameter_count([self.chirp_backbone.ssm_layers]),
            "decoder.det_projection": self._parameter_count([self.det_projection]),
            "decoder.seg_projection": self._parameter_count([self.seg_projection]),
            "decoder.detection": self._parameter_count([self.det_decoder]),
            "decoder.segmentation": self._parameter_count([self.seg_decoder]),
        }

    def analytical_profile(
        self,
        prefix_lengths: Optional[Iterable[int]] = None,
    ) -> Dict[str, int]:
        from .raven_profile import estimate_paper_raven_macs

        return estimate_paper_raven_macs(self.cfg, prefix_lengths=prefix_lengths)

    def forward(
        self,
        adc: torch.Tensor,
        prefix_lengths: Optional[Iterable[int]] = None,
        return_latents: bool = False,
    ) -> Dict[str, torch.Tensor | Dict[int, Dict[str, torch.Tensor]]]:
        chirp_features = self.encode(adc)
        det_projected = self.det_projection.project(chirp_features)
        seg_projected = self.seg_projection.project(chirp_features)
        full_outputs = self.decode_projected(det_projected, seg_projected)
        outputs: Dict[str, torch.Tensor | Dict[int, Dict[str, torch.Tensor]]] = dict(full_outputs)

        if prefix_lengths is not None:
            prefixes: Dict[int, Dict[str, torch.Tensor]] = {}
            for requested in sorted(set(int(value) for value in prefix_lengths)):
                if requested <= 0:
                    continue
                length = min(requested, chirp_features.shape[1])
                if length == chirp_features.shape[1]:
                    prefixes[length] = full_outputs
                else:
                    prefixes[length] = self.decode_projected(
                        det_projected,
                        seg_projected,
                        length=length,
                    )
            outputs["prefixes"] = prefixes

        if return_latents:
            outputs["Latents"] = chirp_features
        return outputs
