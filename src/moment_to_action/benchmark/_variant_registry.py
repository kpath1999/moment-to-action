from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path  # noqa: TC003

import attrs
import platformdirs

from moment_to_action.benchmark._types import CostProfile, VariantID, VariantProfile
from moment_to_action.hardware import ComputeUnit
from moment_to_action.models import ModelID


class VariantRegistry:
    """Persistent registry for benchmarked model variants."""

    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            path = (
                platformdirs.user_cache_path("moment_to_action", "GATech") / "variant_registry.json"
            )
        self._path = path
        self._profiles: dict[VariantID, VariantProfile] = {}

    @property
    def path(self) -> Path:
        """Default persistence path for this registry."""
        return self._path

    def register(self, profile: VariantProfile) -> None:
        """Upsert a profile into the registry."""
        self._profiles[profile.variant_id] = profile

    def get(self, variant_id: VariantID) -> VariantProfile | None:
        """Lookup a profile by exact VariantID."""
        return self._profiles.get(variant_id)

    def query(
        self,
        *,
        model_id: ModelID | None = None,
        compute_unit: ComputeUnit | None = None,
        max_latency_ms: float | None = None,
        min_accuracy: float | None = None,
        hardware_target: str | None = None,
    ) -> list[VariantProfile]:
        """Query profiles with optional filters."""
        matches: list[VariantProfile] = []
        for profile in self._profiles.values():
            if model_id is not None and profile.variant_id.model_id != model_id:
                continue
            if compute_unit is not None and profile.variant_id.compute_unit != compute_unit:
                continue
            if max_latency_ms is not None and profile.inference_mean_ms > max_latency_ms:
                continue
            if min_accuracy is not None and (
                profile.accuracy is None or profile.accuracy < min_accuracy
            ):
                continue
            if hardware_target is not None and profile.hardware_target != hardware_target:
                continue
            matches.append(profile)
        return matches

    def best_variant(self, model_id: ModelID, objective: str) -> VariantProfile | None:
        """Return the best known variant for a model under a target objective."""
        candidates = self.query(model_id=model_id)
        if not candidates:
            return None

        if objective == "latency":
            return min(candidates, key=lambda item: item.inference_mean_ms)
        if objective == "accuracy":
            with_accuracy = [item for item in candidates if item.accuracy is not None]
            return (
                max(with_accuracy, key=lambda item: item.accuracy or 0.0) if with_accuracy else None
            )
        if objective == "efficiency":
            with_energy = [
                item for item in candidates if item.cost.energy_per_inference_mj is not None
            ]
            return (
                min(with_energy, key=lambda item: item.cost.energy_per_inference_mj or float("inf"))
                if with_energy
                else None
            )

        msg = f"Unknown objective: {objective}"
        raise ValueError(msg)

    def all_profiles(self) -> list[VariantProfile]:
        """Return all registered variant profiles."""
        return list(self._profiles.values())

    def save(self, path: Path | None = None) -> None:
        """Serialize the registry to JSON."""
        out_path = path or self._path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        payload: list[dict[str, object]] = []
        for profile in self._profiles.values():
            profile_dict = attrs.asdict(profile)
            profile_dict["variant_id"] = {
                "model_id": profile.variant_id.model_id.value,
                "compute_unit": profile.variant_id.compute_unit.value,
            }
            profile_dict["profiled_at"] = profile.profiled_at.isoformat()
            payload.append(profile_dict)

        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self, path: Path | None = None) -> None:
        """Load registry state from JSON."""
        in_path = path or self._path
        if not in_path.exists():
            return

        raw = json.loads(in_path.read_text(encoding="utf-8"))
        profiles: dict[VariantID, VariantProfile] = {}

        for item in raw:
            variant_data = item["variant_id"]
            cost_data = item["cost"]
            variant_id = VariantID(
                model_id=ModelID(variant_data["model_id"]),
                compute_unit=ComputeUnit(variant_data["compute_unit"]),
            )
            profile = VariantProfile(
                variant_id=variant_id,
                accuracy=item["accuracy"],
                load_latency_ms=item["load_latency_ms"],
                inference_mean_ms=item["inference_mean_ms"],
                inference_p50_ms=item["inference_p50_ms"],
                inference_p95_ms=item["inference_p95_ms"],
                inference_p99_ms=item["inference_p99_ms"],
                peak_memory_mb=item["peak_memory_mb"],
                max_batch_size=item["max_batch_size"],
                hardware_target=item["hardware_target"],
                cost=CostProfile(
                    power_mw=cost_data["power_mw"],
                    energy_per_inference_mj=cost_data["energy_per_inference_mj"],
                ),
                model_size_bytes=item["model_size_bytes"],
                n_runs=item["n_runs"],
                profiled_at=datetime.fromisoformat(item["profiled_at"]),
                accuracy_details=item.get("accuracy_details"),
            )
            profiles[variant_id] = profile

        self._profiles = profiles
