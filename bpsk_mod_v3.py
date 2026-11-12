"""
BPSK Modulator - Embedded Python Block for GNU Radio

Input : bits (byte stream, each item is 0 or 1)
Output: complex samples, 'sps' samples per bit

Steps:
  1. differential encoding (if differential=True, must match the demodulator)
                                        d[k] = d[k-1] XOR b[k]: a 1 flips, a 0 keeps
  2. mapping                            0 -> -1, 1 -> +1
  3. upsampling                         insert sps-1 zeros after each symbol
  4. pulse shaping                      root-raised-cosine (RRC) FIR filter
"""

import numpy as np
from gnuradio import gr


def rrc_taps(sps, alpha, span):
    """Root-raised-cosine taps, `span` symbols long, normalised to unit energy."""
    n = np.arange(-span * sps // 2, span * sps // 2 + 1, dtype=float)
    t = n / sps                                   # time in symbol periods
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:                       # t = 0
            h[i] = 1.0 - alpha + 4 * alpha / np.pi
        elif alpha > 0 and abs(abs(ti) - 1 / (4 * alpha)) < 1e-12:  # t = +-1/(4a)
            h[i] = (alpha / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha)))
        else:
            num = (np.sin(np.pi * ti * (1 - alpha))
                   + 4 * alpha * ti * np.cos(np.pi * ti * (1 + alpha)))
            den = np.pi * ti * (1 - (4 * alpha * ti) ** 2)
            h[i] = num / den
    return h / np.sqrt(np.sum(h ** 2))            # unit energy


class blk(gr.interp_block):
    """BPSK modulator: bits (0/1) in, RRC-shaped complex samples out."""

    def __init__(self, sps=4, alpha=0.35, span=11, differential=False):
        gr.interp_block.__init__(
            self,
            name="My BPSK Mod",
            in_sig=[np.uint8],
            out_sig=[np.complex64],
            interp=int(sps),
        )
        self.sps = int(sps)
        self.alpha = float(alpha)
        self.span = int(span)
        self.differential = bool(differential)

        self.taps = rrc_taps(self.sps, self.alpha, self.span)
        self.history = np.zeros(len(self.taps) - 1)  # FIR memory between work() calls
        self.prev_bit = 0                             # differential-encoder memory

    def work(self, input_items, output_items):
        bits = input_items[0] & 1
        out = output_items[0]
        n_sym = len(out) // self.sps
        bits = bits[:n_sym]

        # 1. differential encoding (removes the 180 degree ambiguity at the receiver)
        if self.differential:
            enc = np.empty(n_sym, dtype=np.uint8)
            prev = self.prev_bit
            for k in range(n_sym):
                prev ^= int(bits[k])
                enc[k] = prev
            self.prev_bit = prev
            bits = enc

        # 2. mapping: 0 -> -1, 1 -> +1
        symbols = 2.0 * bits.astype(float) - 1.0

        # 3. upsampling
        up = np.zeros(n_sym * self.sps)
        up[::self.sps] = symbols

        # 4. RRC filtering, keeping filter state across calls
        x = np.concatenate((self.history, up))
        y = np.convolve(x, self.taps, mode="valid")
        self.history = x[len(x) - (len(self.taps) - 1):]

        out[:n_sym * self.sps] = y.astype(np.complex64)
        return n_sym * self.sps
