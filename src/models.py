"""Neural architectures for token-level hallucination detection.

Includes convolutional, recurrent, transformer, state-space (Mamba),
xLSTM, and CRF-augmented variants used in the ablation and architecture
comparison experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# Basic architectures (ablation study)
# ------------------------------------------------------------------


class CNN1D(nn.Module):
    """Multi-kernel 1-D CNN with LayerNorm."""

    def __init__(self, dim: int, h: int = 64) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(dim, h, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(h, h, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(h, h, kernel_size=7, padding=3)
        self.norm1 = nn.LayerNorm(h)
        self.norm2 = nn.LayerNorm(h)
        self.head = nn.Linear(h, 1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.act(self.conv1(x))
        x = x.transpose(1, 2)
        x = self.norm1(x)
        x = x.transpose(1, 2)
        x = self.act(self.conv2(x))
        x = x.transpose(1, 2)
        x = self.norm2(x)
        x = x.transpose(1, 2)
        x = self.act(self.conv3(x))
        x = x.transpose(1, 2)
        return self.head(x).squeeze(-1)


class ForwardGRU(nn.Module):
    """Forward-only (causal) GRU for onset detection experiments."""

    def __init__(self, dim: int, h: int = 64) -> None:
        super().__init__()
        self.gru = nn.GRU(
            dim, h, num_layers=2, batch_first=True,
            bidirectional=False, dropout=0.1,
        )
        self.head = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.gru(x)[0]).squeeze(-1)


class BackwardGRU(nn.Module):
    """Backward-only GRU (processes sequence in reverse)."""

    def __init__(self, dim: int, h: int = 64) -> None:
        super().__init__()
        self.gru = nn.GRU(
            dim, h, num_layers=2, batch_first=True,
            bidirectional=False, dropout=0.1,
        )
        self.head = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_rev = x.flip(1)
        out = self.head(self.gru(x_rev)[0]).squeeze(-1)
        return out.flip(1)


class BiGRU(nn.Module):
    """Bidirectional GRU with a two-layer MLP head."""

    def __init__(
        self, dim: int, h: int = 64, num_layers: int = 2
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            dim,
            h,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(h * 2, h), nn.GELU(), nn.Linear(h, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.gru(x)[0]).squeeze(-1)


class BiLSTM(nn.Module):
    """Bidirectional LSTM with a two-layer MLP head."""

    def __init__(self, dim: int, h: int = 64) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            dim,
            h,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.head = nn.Sequential(
            nn.Linear(h * 2, h), nn.GELU(), nn.Linear(h, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.lstm(x)[0]).squeeze(-1)


class TransformerEnc(nn.Module):
    """Transformer encoder with Pre-LN (norm_first=True)."""

    def __init__(
        self,
        dim: int,
        h: int = 64,
        nhead: int = 4,
        nlayers: int = 2,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, h)
        layer = nn.TransformerEncoderLayer(
            d_model=h,
            nhead=nhead,
            dim_feedforward=h * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.head = nn.Sequential(
            nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(self.proj(x))).squeeze(-1)


# ------------------------------------------------------------------
# CRF layer and CRF-augmented models
# ------------------------------------------------------------------


class CRF(nn.Module):
    """Linear-chain CRF for binary sequence labeling."""

    def __init__(self, num_tags: int = 2) -> None:
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_trans = nn.Parameter(torch.randn(num_tags))
        self.end_trans = nn.Parameter(torch.randn(num_tags))

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Negative log-likelihood (mean over batch)."""
        return (
            self._normalizer(emissions, mask)
            - self._score(emissions, tags, mask)
        ).mean()

    def decode(
        self, emissions: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Viterbi decoding."""
        batch_size, seq_len, _num_tags = emissions.shape
        score = self.start_trans + emissions[:, 0]
        history: list[torch.Tensor] = []
        for i in range(1, seq_len):
            ns = (
                score.unsqueeze(2)
                + self.transitions
                + emissions[:, i].unsqueeze(1)
            )
            ns, idx = ns.max(dim=1)
            score = torch.where(mask[:, i].unsqueeze(1), ns, score)
            history.append(idx)
        score = score + self.end_trans
        _, best = score.max(dim=1)
        path = [best]
        for hist in reversed(history):
            best = hist[torch.arange(batch_size, device=best.device), best]
            path.append(best)
        path.reverse()
        return torch.stack(path, dim=1)

    def _score(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = tags.shape[0]
        ar = torch.arange(batch_size, device=tags.device)
        score = self.start_trans[tags[:, 0]] + emissions[ar, 0, tags[:, 0]]
        for i in range(1, tags.shape[1]):
            score = score + (
                self.transitions[tags[:, i - 1], tags[:, i]]
                + emissions[ar, i, tags[:, i]]
            ) * mask[:, i].float()
        ends = mask.long().sum(1) - 1
        score = score + self.end_trans[tags[ar, ends]]
        return score

    def _normalizer(
        self, emissions: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        score = self.start_trans + emissions[:, 0]
        for i in range(1, emissions.shape[1]):
            ns = torch.logsumexp(
                score.unsqueeze(2)
                + self.transitions
                + emissions[:, i].unsqueeze(1),
                dim=1,
            )
            score = torch.where(mask[:, i].unsqueeze(1), ns, score)
        return torch.logsumexp(score + self.end_trans, dim=1)


class BiGRU_CRF(nn.Module):
    """BiGRU encoder with a CRF output layer."""

    has_crf = True

    def __init__(self, dim: int, h: int = 64) -> None:
        super().__init__()
        self.gru = nn.GRU(
            dim,
            h,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.head = nn.Linear(h * 2, 2)
        self.crf = CRF(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.gru(x)[0])


class BiLSTM_CRF(nn.Module):
    """BiLSTM encoder with a CRF output layer."""

    has_crf = True

    def __init__(self, dim: int, h: int = 64) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            dim,
            h,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.head = nn.Linear(h * 2, 2)
        self.crf = CRF(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.lstm(x)[0])


class TransCRF(nn.Module):
    """Transformer encoder with a CRF output layer."""

    has_crf = True

    def __init__(self, dim: int, h: int = 64) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, h)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=h,
            nhead=4,
            dim_feedforward=h * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.head = nn.Linear(h, 2)
        self.crf = CRF(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(self.proj(x)))


# ------------------------------------------------------------------
# Extended architectures (fair architecture comparison)
# ------------------------------------------------------------------


class DilatedCNN(nn.Module):
    """Dilated-convolution stack with exponentially growing receptive field."""

    def __init__(self, dim: int, h: int = 64) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Conv1d(
                    dim if i == 0 else h,
                    h,
                    kernel_size=3,
                    padding=2**i,
                    dilation=2**i,
                )
                for i in range(5)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(h) for _ in range(5)])
        self.act = nn.GELU()
        self.head = nn.Sequential(
            nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        for conv, norm in zip(self.layers, self.norms):
            x = conv(x)
            x = x.transpose(1, 2)
            x = norm(x)
            x = self.act(x)
            x = x.transpose(1, 2)
        x = x.transpose(1, 2)
        return self.head(x).squeeze(-1)


class BiGRU_Attention(nn.Module):
    """BiGRU followed by multi-head self-attention."""

    def __init__(self, dim: int, h: int = 64) -> None:
        super().__init__()
        self.gru = nn.GRU(
            dim, h, num_layers=1, batch_first=True, bidirectional=True
        )
        self.attn = nn.MultiheadAttention(
            h * 2, num_heads=4, batch_first=True, dropout=0.1
        )
        self.norm = nn.LayerNorm(h * 2)
        self.head = nn.Sequential(
            nn.Linear(h * 2, h), nn.GELU(), nn.Linear(h, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_seq, _ = self.gru(x)
        attn_out, _ = self.attn(h_seq, h_seq, h_seq)
        h_seq = self.norm(h_seq + attn_out)
        return self.head(h_seq).squeeze(-1)


# ------------------------------------------------------------------
# Mamba (S6-style selective state-space model)
# ------------------------------------------------------------------


class MambaBlock(nn.Module):
    """Single Mamba block with selective scan (Gu & Dao, 2023)."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        d_inner = d_model * expand
        self.d_state = d_state
        self.d_inner = d_inner
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            d_inner,
            d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_inner,
        )
        self.x_proj = nn.Linear(d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)
        log_a = torch.arange(1, d_state + 1, dtype=torch.float32)
        log_a = log_a.unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(torch.log(log_a))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _batch, seq_len, _dim = x.shape
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)
        x_ssm = x_ssm.transpose(1, 2)
        x_ssm = self.conv1d(x_ssm)[:, :, :seq_len]
        x_ssm = x_ssm.transpose(1, 2)
        x_ssm = F.silu(x_ssm)
        bc = self.x_proj(x_ssm)
        b_param, c_param = bc.chunk(2, dim=-1)
        dt = F.softplus(self.dt_proj(x_ssm)).clamp(min=1e-4, max=10.0)
        neg_a = -torch.exp(self.A_log)
        y = self._selective_scan(x_ssm, dt, neg_a, b_param, c_param)
        y = y + x_ssm * self.D.unsqueeze(0).unsqueeze(0)
        y = y * F.silu(z)
        return self.out_proj(y)

    def _selective_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        neg_a: torch.Tensor,
        b_param: torch.Tensor,
        c_param: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, d_inner = x.shape
        dt_a = dt.unsqueeze(-1) * neg_a.unsqueeze(0).unsqueeze(0)
        d_a = torch.exp(dt_a)
        d_bx = (dt * x).unsqueeze(-1) * b_param.unsqueeze(2)
        h = torch.zeros(
            batch_size, d_inner, self.d_state, device=x.device, dtype=x.dtype
        )
        ys: list[torch.Tensor] = []
        for i in range(seq_len):
            h = h * d_a[:, i] + d_bx[:, i]
            y_i = (h * c_param[:, i].unsqueeze(1)).sum(-1)
            ys.append(y_i)
        return torch.stack(ys, dim=1)


class MambaSeqLabeler(nn.Module):
    """Bidirectional Mamba for sequence labeling."""

    def __init__(
        self, dim: int, h: int = 64, n_layers: int = 2
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, h)
        self.fwd_layers = nn.ModuleList(
            [MambaBlock(h, d_state=16, expand=2) for _ in range(n_layers)]
        )
        self.bwd_layers = nn.ModuleList(
            [MambaBlock(h, d_state=16, expand=2) for _ in range(n_layers)]
        )
        self.fwd_norms = nn.ModuleList(
            [nn.LayerNorm(h) for _ in range(n_layers)]
        )
        self.bwd_norms = nn.ModuleList(
            [nn.LayerNorm(h) for _ in range(n_layers)]
        )
        self.head = nn.Sequential(
            nn.Linear(h * 2, h), nn.GELU(), nn.Linear(h, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        h_fwd = x
        for layer, norm in zip(self.fwd_layers, self.fwd_norms):
            h_fwd = norm(h_fwd + layer(h_fwd))
        h_bwd = x.flip(1)
        for layer, norm in zip(self.bwd_layers, self.bwd_norms):
            h_bwd = norm(h_bwd + layer(h_bwd))
        h_bwd = h_bwd.flip(1)
        return self.head(torch.cat([h_fwd, h_bwd], dim=-1)).squeeze(-1)


# ------------------------------------------------------------------
# xLSTM (Beck et al., 2024) -- exponential gating variant
# ------------------------------------------------------------------


class xLSTMCell(nn.Module):
    """Single xLSTM cell with exponential input gate and LayerNorm."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(input_size + hidden_size, 4 * hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _dim = x_seq.shape
        h = torch.zeros(
            batch_size, self.hidden_size,
            device=x_seq.device, dtype=x_seq.dtype,
        )
        c = torch.zeros(
            batch_size, self.hidden_size,
            device=x_seq.device, dtype=x_seq.dtype,
        )
        outputs: list[torch.Tensor] = []
        for t in range(seq_len):
            combined = torch.cat([x_seq[:, t], h], dim=-1)
            gates = self.W(combined)
            ig, fg, og, gg = gates.chunk(4, dim=-1)
            ig = torch.exp(ig.clamp(max=5))
            fg = torch.sigmoid(fg)
            og = torch.sigmoid(og)
            gg = torch.tanh(gg)
            c = fg * c + ig * gg
            h = og * self.layer_norm(c)
            outputs.append(h)
        return torch.stack(outputs, dim=1)


class BixLSTM(nn.Module):
    """Bidirectional xLSTM for sequence labeling."""

    def __init__(
        self, dim: int, h: int = 64, num_layers: int = 2
    ) -> None:
        super().__init__()
        self.fwd_cells = nn.ModuleList()
        self.bwd_cells = nn.ModuleList()
        for layer_i in range(num_layers):
            inp_size = dim if layer_i == 0 else h
            self.fwd_cells.append(xLSTMCell(inp_size, h))
            self.bwd_cells.append(xLSTMCell(inp_size, h))
        self.head = nn.Sequential(
            nn.Linear(h * 2, h), nn.GELU(), nn.Linear(h, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_fwd = x
        for cell in self.fwd_cells:
            h_fwd = cell(h_fwd)
        h_bwd = x.flip(1)
        for cell in self.bwd_cells:
            h_bwd = cell(h_bwd)
        h_bwd = h_bwd.flip(1)
        return self.head(torch.cat([h_fwd, h_bwd], dim=-1)).squeeze(-1)


# ------------------------------------------------------------------
# Focal Loss (Lin et al., 2017)
# ------------------------------------------------------------------


class FocalLoss(nn.Module):
    """Binary focal loss -- down-weights easy examples."""

    def __init__(
        self, gamma: float = 2.0, alpha: float | None = None
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        if self.alpha is not None:
            alpha_t = (
                self.alpha * targets + (1 - self.alpha) * (1 - targets)
            )
            focal_weight = alpha_t * focal_weight
        loss = focal_weight * bce
        return (loss * mask).sum() / mask.sum()
