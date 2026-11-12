"""
BPSK Demodulator

Input   : complex samples, 'sps' samples per symbol
Output 0: recovered bits (byte stream, 0 or 1)
Output 1: corrected symbols (complex) - connect to a QT GUI Constellation Sink

Steps:
  1. matched filter      same RRC filter as the transmitter
  2. timing recovery     Gardner loop: finds the centre of each symbol by itself,
                         even if the transmitter's clock is slightly off
  3. Costas loop         removes phase rotation and small frequency offsets
  4. decision            bit = 1 if real part > 0 else 0
  5. differential decoding (if differential=True, must match the modulator)
                         removes the 180 degree ambiguity of the Costas loop

Note: the first ~span bits after start-up are filter/loop settling (ignore them).
"""

import numpy as np
from gnuradio import gr


def rrc_taps(sps, alpha, span):
    """Root-raised-cosine taps (identical to the modulator's)."""
    n = np.arange(-span * sps // 2, span * sps // 2 + 1, dtype=float)
    t = n / sps
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            h[i] = 1.0 - alpha + 4 * alpha / np.pi
        elif alpha > 0 and abs(abs(ti) - 1 / (4 * alpha)) < 1e-12:
            h[i] = (alpha / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha)))
        else:
            num = (np.sin(np.pi * ti * (1 - alpha))
                   + 4 * alpha * ti * np.cos(np.pi * ti * (1 + alpha)))
            den = np.pi * ti * (1 - (4 * alpha * ti) ** 2)
            h[i] = num / den
    return h / np.sqrt(np.sum(h ** 2))


def loop_gains(bw, zeta=0.707):
    """Proportional and integral gains of a 2nd-order loop (as in GNU Radio)."""
    denom = 1 + 2 * zeta * bw + bw ** 2
    return 4 * zeta * bw / denom, 4 * bw ** 2 / denom


def interp(buf, pos):
    """Value of 'buf' at fractional index 'pos' (cubic interpolation).

    Example: pos = 10.25 gives a value a quarter of the way from buf[10] to buf[11],
    following the curve through buf[9], buf[10], buf[11], buf[12].
    """
    i = int(np.floor(pos))
    mu = pos - i
    ym1, y0, y1, y2 = buf[i - 1], buf[i], buf[i + 1], buf[i + 2]
    return (y0
            + mu * (y1 - ym1) / 2
            + mu ** 2 * (ym1 - 2.5 * y0 + 2 * y1 - 0.5 * y2)
            + mu ** 3 * (1.5 * (y0 - y1) + 0.5 * (y2 - ym1)))


class blk(gr.basic_block):
    """BPSK demodulator with timing recovery: complex samples in, bits out."""

    def __init__(self, sps=4, alpha=0.35, span=11, loop_bw=0.05,
                 timing_bw=0.005, differential=False):
        gr.basic_block.__init__(
            self,
            name="My BPSK Demod",
            in_sig=[np.complex64],
            out_sig=[np.uint8, np.complex64],
        )
        self.sps = int(sps)
        self.span = int(span)
        self.differential = bool(differential)
        self.set_relative_rate(1, self.sps)      # roughly sps samples in per symbol out

        # matched filter
        self.taps = rrc_taps(self.sps, float(alpha), self.span)
        self.history = np.zeros(len(self.taps) - 1, dtype=complex)

        # timing recovery state
        self.buf = np.zeros(4, dtype=complex)    # filtered samples not yet used up
        self.pos = 4.0                           # where the next symbol centre is in buf
        self.prev_on = 0j                        # previous symbol-centre sample
        self.tk_p, self.tk_i = loop_gains(float(timing_bw))
        self.tk_p /= 2.0                         # divide by the Gardner detector's gain
        self.tk_i /= 2.0
        self.t_int = 0.0                         # timing loop integrator

        # Costas loop state
        self.k_phase, self.k_freq = loop_gains(float(loop_bw))
        self.phase = 0.0
        self.freq = 0.0

        self.prev_bit = 0                        # differential-decoder memory

    def forecast(self, noutput_items, ninputs):
        # to make noutput_items symbols we need about sps input samples each
        return [noutput_items * self.sps] * ninputs

    def general_work(self, input_items, output_items):
        x_in = input_items[0]
        out_bits = output_items[0]
        out_syms = output_items[1]
        n_out_max = min(len(out_bits), len(out_syms))

        # 1. matched filter on the new samples (with memory across calls)
        n_in = min(len(x_in), n_out_max * self.sps)
        x = np.concatenate((self.history, x_in[:n_in]))
        y = np.convolve(x, self.taps, mode="valid")
        self.history = x[len(x) - (len(self.taps) - 1):]
        self.consume(0, n_in)                    # tell GNU Radio we used n_in inputs
        buf = np.concatenate((self.buf, y))

        sps, half = self.sps, self.sps / 2.0
        pos, prev_on, t_int = self.pos, self.prev_on, self.t_int
        phase, freq, prev_bit = self.phase, self.freq, self.prev_bit
        k = 0
        while k < n_out_max and pos + 3 < len(buf):
            # 2. timing recovery (Gardner)
            on = interp(buf, pos)                # sample at the estimated centre
            mid = interp(buf, pos - half)        # sample half-way to the previous centre
            t_err = np.real(np.conj(mid) * (prev_on - on))
            t_err = float(np.clip(t_err, -1.0, 1.0))
            t_int += self.tk_i * t_err
            pos += sps + (self.tk_p * t_err + t_int)   # step to the next centre
            prev_on = on

            # 3. Costas loop
            z = on * np.exp(-1j * phase)
            err = np.sign(z.real) * z.imag
            freq += self.k_freq * err
            phase += freq + self.k_phase * err
            phase = (phase + np.pi) % (2 * np.pi) - np.pi

            # 4. decision
            bit = 1 if z.real > 0 else 0

            # 5. differential decoding: a change means 1, no change means 0
            if self.differential:
                bit, prev_bit = bit ^ prev_bit, bit

            out_bits[k] = bit
            out_syms[k] = z
            k += 1

        # keep only the part of buf still needed for the next symbols
        keep_from = max(int(np.floor(pos - half)) - 2, 0)
        self.buf = buf[keep_from:]
        self.pos = pos - keep_from
        self.prev_on, self.t_int = prev_on, t_int
        self.phase, self.freq, self.prev_bit = phase, freq, prev_bit
        return k
