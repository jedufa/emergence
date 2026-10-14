from __future__ import annotations

import time

class ProgressLogger:
    def __init__(self, total_gens: int, print_every: int = 1, ema_alpha: float = 0.2):
        self.total = int(total_gens)
        self.print_every = int(print_every)
        self.ema_alpha = float(ema_alpha)
        self.t0 = time.time()
        self.last_t = self.t0
        self.ema_gen = None

    def log(self, gen_idx: int):
        now = time.time()
        dt = now - self.last_t
        self.last_t = now

        self.ema_gen = dt if self.ema_gen is None else (
            self.ema_alpha * dt + (1 - self.ema_alpha) * self.ema_gen
        )

        if gen_idx % self.print_every == 0:
            elapsed = now - self.t0
            remaining = self.total - (gen_idx + 1)
            eta = max(0.0, remaining * (self.ema_gen or 0.0))
            print(
                f"[gen {gen_idx+1}/{self.total}] "
                f"elapsed {self._fmt(elapsed)} | "
                f"avg/gen {self.ema_gen:.2f}s | "
                f"ETA {self._fmt(eta)}",
                flush=True,
            )

    @staticmethod
    def _fmt(seconds: float) -> str:
        if seconds < 10:
            return f"00:00:{seconds:05.2f}"
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
