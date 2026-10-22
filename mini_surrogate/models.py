"""The two surrogates compared in part 5. Both take x [b, NT, 5] and return ru [b, NZ, NT]."""
import torch
from torch import nn
from neuralop.models import FNO


class FNOColumn(nn.Module):
    """2D FNO on the (depth, time) grid. Time-series inputs are broadcast over depth;
    neuralop adds the (depth, time) coordinates as two extra channels."""

    def __init__(self, nz=20, n_in=5, modes=(10, 64), width=32, layers=4):
        super().__init__()
        self.nz = nz
        self.fno = FNO(n_modes=modes, in_channels=n_in, out_channels=1, hidden_channels=width,
                       n_layers=layers, domain_padding=[0.25, 0.125])

    def forward(self, x):
        b, nt, c = x.shape
        x = x.permute(0, 2, 1).unsqueeze(2).expand(b, c, self.nz, nt)
        return self.fno(x).squeeze(1)


class FNO1DColumn(nn.Module):
    """1D FNO over time; the 20 depths are output channels (same layout as the LSTM)."""

    def __init__(self, nz=20, n_in=5, modes=128, width=32, layers=4):
        super().__init__()
        self.fno = FNO(n_modes=(modes,), in_channels=n_in, out_channels=nz, hidden_channels=width,
                       n_layers=layers, domain_padding=[0.25])

    def forward(self, x):
        return self.fno(x.permute(0, 2, 1))


class LSTMColumn(nn.Module):
    """LSTM over time; at each step it outputs ru at all depths."""

    def __init__(self, nz=20, n_in=5, hidden=128, layers=2, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, layers, batch_first=True, bidirectional=bidirectional)
        self.head = nn.Linear(hidden * (2 if bidirectional else 1), nz)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.head(h).transpose(1, 2)


def build(name, **kw):
    """Named model; keyword arguments override the defaults (used by the sweep)."""
    if kw:
        return {"fno": FNOColumn, "fno1d": FNO1DColumn, "lstm": LSTMColumn}[name](**kw)
    # FNO settings chosen by the validation sweep (mini_surrogate/sweep.py, results/part5/sweep.json)
    return {"fno": lambda: FNOColumn(modes=(10, 500), width=12),
            "fno1d": lambda: FNO1DColumn(modes=500, width=32),
            # LSTMs sized to roughly the FNOs' 1.0-1.4 M parameters, so the comparison is fair
            "lstm": lambda: LSTMColumn(hidden=256, layers=3),
            "bilstm": lambda: LSTMColumn(hidden=160, layers=3, bidirectional=True)}[name]()


def n_params(model):
    return sum(p.numel() for p in model.parameters())
