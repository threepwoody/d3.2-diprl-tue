import math
from typing import Any, Dict, Optional, Union

import torch as th

from stable_baselines3.common.utils import get_device


class ProgramArchitectureEntropy:
    def __init__(
        self,
        auto_tune: bool = False,
        device: Union[th.device, str] = "auto",
    ):
        self.auto_tune = auto_tune
        self.device = get_device(device)
        self.log_alpha_arch: Optional[th.nn.Parameter] = None
        self.alpha_arch_optimizer: Optional[th.optim.Adam] = None
        self.h_target: Optional[float] = None
        self.h_max: Optional[float] = None
        self.h_ema: Optional[float] = None
        self.h_ema_tau = 0.995
        self._is_setup = False

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._is_setup = getattr(self, "_is_setup", False)

    def setup(self, max_depth: int) -> None:
        if not self.auto_tune:
            return
        self.log_alpha_arch = th.nn.Parameter(th.tensor(-10.0, device=self.device))
        self.alpha_arch_optimizer = th.optim.Adam([self.log_alpha_arch], lr=1e-4)
        self.h_max = math.log(max_depth)
        self.h_target = 0.1 * self.h_max
        self.h_ema = None
        self._is_setup = True
        print(
            f"[ProgramArchitectureEntropy] h_max={self.h_max:.4f}, "
            f"h_target={self.h_target:.4f}, max_depth={max_depth}"
        )

    def update_alpha(self, entropy: th.Tensor) -> float:
        if not self.auto_tune:
            return 0.0
        if self.log_alpha_arch is None or self.alpha_arch_optimizer is None:
            raise RuntimeError("Call setup() before update_alpha().")
        if self.h_target is None:
            raise RuntimeError("Call setup() before update_alpha().")
        if self.h_ema is None:
            self.h_ema = entropy.item()
        else:
            self.h_ema = (
                self.h_ema * self.h_ema_tau + (1 - self.h_ema_tau) * entropy.item()
            )
        loss_alpha = -(self.log_alpha_arch * (self.h_ema - self.h_target))
        self.alpha_arch_optimizer.zero_grad()
        loss_alpha.backward()
        self.alpha_arch_optimizer.step()
        self.log_alpha_arch.data.clamp_(-10.0, 10.0)
        return self.h_target

    def get_alpha_tensor(self) -> Optional[th.Tensor]:
        if self.log_alpha_arch is None:
            return None
        return th.exp(self.log_alpha_arch)

    def get_metrics(self) -> Dict[str, Any]:
        metrics = {
            "auto_tune": self.auto_tune,
        }
        if self.log_alpha_arch is not None:
            metrics["log_alpha_arch"] = self.log_alpha_arch.item()
            metrics["alpha_arch"] = th.exp(self.log_alpha_arch).item()
        if self.h_ema is not None:
            metrics["h_ema"] = self.h_ema
        if self.h_max is not None:
            metrics["h_max"] = self.h_max
        if self.h_target is not None:
            metrics["h_target"] = self.h_target
        return metrics

    @property
    def is_auto_tune(self) -> bool:
        return self.auto_tune

    @property
    def is_setup(self) -> bool:
        return self._is_setup
