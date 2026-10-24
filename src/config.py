"""Конфигурация прогона: pydantic-модели, загрузка YAML, хэш (MODEL_SPEC §3, §6, §10)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class RetryParams(BaseModel):
    """Параметры политик повторов R1–R3 (+ORBIT для V3)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_retries: int = 3
    fixed_delay: float = 1.0  # R1
    base: float = 1.0  # R2, R3
    multiplier: float = 2.0
    cap: float = 16.0
    budget_ratio: float = 0.1  # R3 (D-006)
    budget_window: float = 100.0
    budget_reserve: float = 10.0
    orbit_rate: float = 1.0  # ORBIT: интенсивность повтора одного запроса на орбите


class ProtoParams(BaseModel):
    """Параметры протоколов A/B и метрик. Триггер (m_s, D_s) калибруется на pilot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    t_warm: float = 200.0  # t0 разогрева (A)
    trigger_mult: float = 4.0  # m_s: множитель всплеска
    trigger_len: float = 50.0  # D_s: длительность всплеска
    h_rec: float = 1500.0  # H_rec
    overload_rho: float = 1.05  # B: нагрузка в фазе отказа
    overload_len: float = 400.0  # B: длительность перегрузки
    window: float = 100.0  # W для g(t)
    w_last: float = 300.0  # окно итогового наблюдения
    theta: float = 0.5  # порог отказа
    rec_level: float = 0.9
    rec_hold: float = 50.0
    tail: float | None = None  # None: автоматически (D-008)


class Config(BaseModel):
    """Полная конфигурация системы и протоколов. ρ_test, протокол, seed — не её часть."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    c: int = 8
    K: int = 128
    T: float = 10.0
    R: Literal["R0", "R1", "R2", "R3", "ORBIT"] = "R1"
    Q: Literal["FIFO", "LIFO", "ALIFO"] = "FIFO"
    D: Literal["off", "on"] = "off"
    S: Literal["Exp", "LogN", "Pareto"] = "Exp"
    theta_q: float = 0.5
    logn_cv: float = 2.0
    pareto_alpha: float = 2.5
    retry: RetryParams = RetryParams()
    proto: ProtoParams = ProtoParams()

    @field_validator("D", mode="before")
    @classmethod
    def _d_from_yaml_bool(cls, v: Any) -> Any:
        # YAML 1.1 читает off/on как False/True
        if v is True:
            return "on"
        if v is False:
            return "off"
        return v

    @model_validator(mode="after")
    def _check(self) -> Config:
        if self.c < 1 or self.K < 0:
            raise ValueError("c >= 1, K >= 0")
        if self.T <= 0:
            raise ValueError("T > 0")
        if not 0.0 < self.theta_q <= 1.0:
            raise ValueError("theta_q in (0, 1]")
        return self


def config_hash(cfg: Config) -> str:
    """SHA-256 (первые 16 hex) канонического JSON конфигурации."""
    payload = json.dumps(cfg.model_dump(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return Config.model_validate(data)
