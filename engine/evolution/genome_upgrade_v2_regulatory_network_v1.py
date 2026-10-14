"""Deterministic fixed-shape regulatory mechanics for Genome Upgrade V2."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Final, Sequence

from .genome_upgrade_v2 import (
    CONTEXT_CHANNELS,
    CONTEXT_COUNT,
    POLYGENIC_WEIGHTS_Q10,
    PRODUCT_CHANNELS,
    PRODUCT_COUNT,
    Q10_SCALE,
)


REGULATORY_SCHEMA_VERSION: Final[str] = (
    "genome_upgrade_v2_fixed_regulatory_network_v1"
)
REGULATORY_UPDATE_SCHEDULE: Final[str] = "once_before_policy_each_live_tick_v1"
REGULATOR_COUNT: Final[int] = 4
MAX_FAN_IN: Final[int] = 4
PLEIOTROPIC_INFLUENCE_BUDGET_Q10: Final[int] = 3 * Q10_SCALE
MODIFIER_FACTOR_MIN_Q10: Final[int] = 3 * Q10_SCALE // 4
MODIFIER_FACTOR_MAX_Q10: Final[int] = 5 * Q10_SCALE // 4

NODE_PRODUCT_INPUT: Final[str] = "product_input"
NODE_CONTEXT_INPUT: Final[str] = "context_input"
NODE_REGULATOR: Final[str] = "regulator"
NODE_PRODUCT_OUTPUT: Final[str] = "product_output"
NODE_KINDS: Final[frozenset[str]] = frozenset(
    {
        NODE_PRODUCT_INPUT,
        NODE_CONTEXT_INPUT,
        NODE_REGULATOR,
        NODE_PRODUCT_OUTPUT,
    }
)

EDGE_ACTIVATE: Final[str] = "activate"
EDGE_INHIBIT: Final[str] = "inhibit"
EDGE_MODULATE: Final[str] = "modulate"
EDGE_EFFECTS: Final[frozenset[str]] = frozenset(
    {EDGE_ACTIVATE, EDGE_INHIBIT, EDGE_MODULATE}
)


class GenomeUpgradeV2RegulatoryError(ValueError):
    """Stable validation failure for the fixed regulatory contract."""


def _trunc_div(value: int, divisor: int) -> int:
    magnitude = abs(int(value)) // divisor
    return -magnitude if value < 0 else magnitude


def _q10_product(*values: int) -> int:
    result = Q10_SCALE
    for value in values:
        result = _trunc_div(result * int(value), Q10_SCALE)
    return result


def _clamp_q10(value: int) -> int:
    return min(Q10_SCALE, max(-Q10_SCALE, int(value)))


def _validated_q10_vector(
    values: Sequence[Any], *, count: int, name: str
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or len(values) != count:
        raise GenomeUpgradeV2RegulatoryError(f"{name} has the wrong length")
    result: list[int] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int):
            raise GenomeUpgradeV2RegulatoryError(
                f"{name}[{index}] must be an integer"
            )
        if value < -Q10_SCALE or value > Q10_SCALE:
            raise GenomeUpgradeV2RegulatoryError(
                f"{name}[{index}] is outside the Q10 envelope"
            )
        result.append(int(value))
    return tuple(result)


@dataclass(frozen=True)
class RegulatoryNodeV1:
    node_id: str
    kind: str
    channel_index: int
    threshold_q10: int = 0

    def __post_init__(self) -> None:
        if not self.node_id or self.kind not in NODE_KINDS:
            raise GenomeUpgradeV2RegulatoryError("regulatory node identity is invalid")
        if isinstance(self.channel_index, bool) or not isinstance(
            self.channel_index, int
        ):
            raise GenomeUpgradeV2RegulatoryError("node channel index must be an integer")
        limits = {
            NODE_PRODUCT_INPUT: PRODUCT_COUNT,
            NODE_CONTEXT_INPUT: CONTEXT_COUNT,
            NODE_REGULATOR: REGULATOR_COUNT,
            NODE_PRODUCT_OUTPUT: PRODUCT_COUNT,
        }
        if not 0 <= self.channel_index < limits[self.kind]:
            raise GenomeUpgradeV2RegulatoryError("node channel index is outside its kind")
        if (
            isinstance(self.threshold_q10, bool)
            or not isinstance(self.threshold_q10, int)
            or not 0 <= self.threshold_q10 <= Q10_SCALE
        ):
            raise GenomeUpgradeV2RegulatoryError("node threshold is outside [0, 1024]")
        if self.kind != NODE_REGULATOR and self.threshold_q10 != 0:
            raise GenomeUpgradeV2RegulatoryError(
                "only regulator nodes may define thresholds"
            )


@dataclass(frozen=True)
class RegulatoryEdgeV1:
    source_node_id: str
    target_node_id: str
    effect: str
    weight_q10: int

    def __post_init__(self) -> None:
        if self.effect not in EDGE_EFFECTS:
            raise GenomeUpgradeV2RegulatoryError("regulatory edge effect is invalid")
        if (
            isinstance(self.weight_q10, bool)
            or not isinstance(self.weight_q10, int)
            or not 0 < self.weight_q10 <= Q10_SCALE
        ):
            raise GenomeUpgradeV2RegulatoryError(
                "regulatory edge weight is outside (0, 1024]"
            )


@dataclass(frozen=True)
class EpistaticModifierEdgeV1:
    source_product_index: int
    modifier_index: int
    target_product_index: int
    effect: str
    weight_q10: int

    def __post_init__(self) -> None:
        for value in (
            self.source_product_index,
            self.modifier_index,
            self.target_product_index,
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < PRODUCT_COUNT
            ):
                raise GenomeUpgradeV2RegulatoryError(
                    "epistatic channel index is outside the product envelope"
                )
        if self.source_product_index == self.target_product_index:
            raise GenomeUpgradeV2RegulatoryError(
                "epistatic edges must act across product loci"
            )
        if self.effect not in {EDGE_ACTIVATE, EDGE_INHIBIT}:
            raise GenomeUpgradeV2RegulatoryError("epistatic edge effect is invalid")
        if (
            isinstance(self.weight_q10, bool)
            or not isinstance(self.weight_q10, int)
            or not 0 < self.weight_q10 <= Q10_SCALE
        ):
            raise GenomeUpgradeV2RegulatoryError(
                "epistatic edge weight is outside (0, 1024]"
            )


@dataclass(frozen=True)
class FixedRegulatoryNetworkV1:
    nodes: tuple[RegulatoryNodeV1, ...]
    edges: tuple[RegulatoryEdgeV1, ...]
    epistatic_edges: tuple[EpistaticModifierEdgeV1, ...]
    maximum_fan_in: int = MAX_FAN_IN
    active_node_cost_q10: int = 32
    active_edge_cost_q10: int = 8
    signal_cost_q10: int = 64
    regulatory_budget_q10: int = 640
    update_schedule: str = REGULATORY_UPDATE_SCHEDULE

    def __post_init__(self) -> None:
        node_by_id = {node.node_id: node for node in self.nodes}
        if len(node_by_id) != len(self.nodes):
            raise GenomeUpgradeV2RegulatoryError("regulatory node IDs are not unique")
        expected_counts = {
            NODE_PRODUCT_INPUT: PRODUCT_COUNT,
            NODE_CONTEXT_INPUT: CONTEXT_COUNT,
            NODE_REGULATOR: REGULATOR_COUNT,
            NODE_PRODUCT_OUTPUT: PRODUCT_COUNT,
        }
        actual_counts = Counter(node.kind for node in self.nodes)
        if any(actual_counts[kind] != count for kind, count in expected_counts.items()):
            raise GenomeUpgradeV2RegulatoryError(
                "regulatory node kinds do not fill the fixed envelope"
            )
        for kind, count in expected_counts.items():
            if {
                node.channel_index for node in self.nodes if node.kind == kind
            } != set(range(count)):
                raise GenomeUpgradeV2RegulatoryError(
                    f"{kind} channel indices do not fill the fixed envelope"
                )
        fan_in: Counter[str] = Counter()
        for edge in self.edges:
            source = node_by_id.get(edge.source_node_id)
            target = node_by_id.get(edge.target_node_id)
            if source is None or target is None:
                raise GenomeUpgradeV2RegulatoryError(
                    "regulatory edge references an unknown node"
                )
            input_kinds = {NODE_PRODUCT_INPUT, NODE_CONTEXT_INPUT}
            if target.kind == NODE_REGULATOR:
                if source.kind not in input_kinds or edge.effect == EDGE_MODULATE:
                    raise GenomeUpgradeV2RegulatoryError(
                        "regulator fan-in must activate or inhibit from an input"
                    )
            elif target.kind == NODE_PRODUCT_OUTPUT:
                if source.kind != NODE_REGULATOR:
                    raise GenomeUpgradeV2RegulatoryError(
                        "product output fan-in must come from a regulator"
                    )
            else:
                raise GenomeUpgradeV2RegulatoryError(
                    "regulatory edges may only target regulators or outputs"
                )
            fan_in[target.node_id] += 1
        if (
            isinstance(self.maximum_fan_in, bool)
            or not isinstance(self.maximum_fan_in, int)
            or not 1 <= self.maximum_fan_in <= MAX_FAN_IN
            or any(value > self.maximum_fan_in for value in fan_in.values())
        ):
            raise GenomeUpgradeV2RegulatoryError("regulatory fan-in exceeds its bound")
        for name in (
            "active_node_cost_q10",
            "active_edge_cost_q10",
            "signal_cost_q10",
            "regulatory_budget_q10",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= 4 * Q10_SCALE
            ):
                raise GenomeUpgradeV2RegulatoryError(
                    f"{name} is outside the fixed cost envelope"
                )
        if self.update_schedule != REGULATORY_UPDATE_SCHEDULE:
            raise GenomeUpgradeV2RegulatoryError(
                "regulatory update schedule is not the frozen P1.0 schedule"
            )


def _fixed_nodes() -> tuple[RegulatoryNodeV1, ...]:
    return (
        *(
            RegulatoryNodeV1(f"product-input-{index}", NODE_PRODUCT_INPUT, index)
            for index in range(PRODUCT_COUNT)
        ),
        *(
            RegulatoryNodeV1(f"context-input-{index}", NODE_CONTEXT_INPUT, index)
            for index in range(CONTEXT_COUNT)
        ),
        RegulatoryNodeV1("regulator-0", NODE_REGULATOR, 0, 128),
        RegulatoryNodeV1("regulator-1", NODE_REGULATOR, 1, 128),
        RegulatoryNodeV1("regulator-2", NODE_REGULATOR, 2, 128),
        RegulatoryNodeV1("regulator-3", NODE_REGULATOR, 3, 128),
        *(
            RegulatoryNodeV1(f"product-output-{index}", NODE_PRODUCT_OUTPUT, index)
            for index in range(PRODUCT_COUNT)
        ),
    )


DEFAULT_FIXED_REGULATORY_NETWORK_V1: Final[FixedRegulatoryNetworkV1] = (
    FixedRegulatoryNetworkV1(
        nodes=_fixed_nodes(),
        edges=(
            RegulatoryEdgeV1(
                "context-input-0", "regulator-0", EDGE_ACTIVATE, 512
            ),
            RegulatoryEdgeV1(
                "product-input-0", "regulator-0", EDGE_ACTIVATE, 256
            ),
            RegulatoryEdgeV1(
                "context-input-1", "regulator-1", EDGE_ACTIVATE, 640
            ),
            RegulatoryEdgeV1(
                "product-input-4", "regulator-1", EDGE_INHIBIT, 256
            ),
            RegulatoryEdgeV1(
                "context-input-5", "regulator-2", EDGE_ACTIVATE, 768
            ),
            RegulatoryEdgeV1(
                "context-input-2", "regulator-2", EDGE_INHIBIT, 256
            ),
            RegulatoryEdgeV1(
                "context-input-6", "regulator-3", EDGE_ACTIVATE, 768
            ),
            RegulatoryEdgeV1(
                "context-input-7", "regulator-3", EDGE_INHIBIT, 768
            ),
            RegulatoryEdgeV1(
                "product-input-10", "regulator-3", EDGE_ACTIVATE, 1024
            ),
            RegulatoryEdgeV1(
                "regulator-0", "product-output-0", EDGE_ACTIVATE, 256
            ),
            RegulatoryEdgeV1(
                "regulator-0", "product-output-4", EDGE_INHIBIT, 128
            ),
            RegulatoryEdgeV1(
                "regulator-1", "product-output-3", EDGE_ACTIVATE, 384
            ),
            RegulatoryEdgeV1(
                "regulator-1", "product-output-1", EDGE_INHIBIT, 256
            ),
            RegulatoryEdgeV1(
                "regulator-2", "product-output-9", EDGE_ACTIVATE, 512
            ),
            RegulatoryEdgeV1(
                "regulator-2", "product-output-10", EDGE_INHIBIT, 256
            ),
            RegulatoryEdgeV1(
                "regulator-3", "product-output-10", EDGE_MODULATE, 512
            ),
        ),
        epistatic_edges=(
            EpistaticModifierEdgeV1(9, 9, 3, EDGE_ACTIVATE, 512),
            EpistaticModifierEdgeV1(0, 0, 4, EDGE_INHIBIT, 256),
        ),
    )
)


@dataclass(frozen=True)
class RegulatoryExpressionV1:
    regulated_products_q10: tuple[int, ...]
    regulator_nodes_q10: tuple[int, ...]
    epistatic_delta_q10: tuple[int, ...]
    regulatory_cost_q10: int
    active_regulator_count: int
    active_edge_count: int
    developmental_stage: str


def developmental_stage_from_context_v1(context_q10: Sequence[Any]) -> str:
    context = _validated_q10_vector(
        context_q10, count=CONTEXT_COUNT, name="context_q10"
    )
    juvenile = context[CONTEXT_CHANNELS.index("juvenile")] > 0
    senescent = context[CONTEXT_CHANNELS.index("senescent")] > 0
    if juvenile and senescent:
        raise GenomeUpgradeV2RegulatoryError(
            "juvenile and senescent context cannot both be active"
        )
    if juvenile:
        return "juvenile"
    if senescent:
        return "senescent"
    return "mature"


def express_regulatory_reference_v1(
    products_q10: Sequence[Any],
    context_q10: Sequence[Any],
    modifiers_q10: Sequence[Any],
    *,
    network: FixedRegulatoryNetworkV1 = DEFAULT_FIXED_REGULATORY_NETWORK_V1,
) -> RegulatoryExpressionV1:
    products = _validated_q10_vector(
        products_q10, count=PRODUCT_COUNT, name="products_q10"
    )
    context = _validated_q10_vector(
        context_q10, count=CONTEXT_COUNT, name="context_q10"
    )
    modifiers = _validated_q10_vector(
        modifiers_q10, count=PRODUCT_COUNT, name="modifiers_q10"
    )
    stage = developmental_stage_from_context_v1(context)
    node_by_id = {node.node_id: node for node in network.nodes}

    prepared_products = list(products)
    stability_factor = min(
        MODIFIER_FACTOR_MAX_Q10,
        max(
            MODIFIER_FACTOR_MIN_Q10,
            Q10_SCALE + _trunc_div(modifiers[2], 4),
        ),
    )
    prepared_products[2] = _clamp_q10(
        _q10_product(prepared_products[2], stability_factor)
    )
    saturation_limit = Q10_SCALE - abs(modifiers[11]) // 8
    prepared_products[7] = min(
        saturation_limit,
        max(-saturation_limit, prepared_products[7]),
    )
    products = tuple(prepared_products)

    epistatic_delta = [0] * PRODUCT_COUNT
    active_epistatic_edges = 0
    for edge in network.epistatic_edges:
        magnitude = _q10_product(
            products[edge.source_product_index],
            modifiers[edge.modifier_index],
            edge.weight_q10,
        )
        delta = abs(magnitude) * (-1 if edge.effect == EDGE_INHIBIT else 1)
        epistatic_delta[edge.target_product_index] += delta
        active_epistatic_edges += int(delta != 0)
    epistatic_products = tuple(
        _clamp_q10(value + delta)
        for value, delta in zip(products, epistatic_delta)
    )

    regulator_accumulators = [0] * REGULATOR_COUNT
    active_input_edges = 0
    for edge in network.edges:
        target = node_by_id[edge.target_node_id]
        if target.kind != NODE_REGULATOR:
            continue
        source = node_by_id[edge.source_node_id]
        signal = (
            epistatic_products[source.channel_index]
            if source.kind == NODE_PRODUCT_INPUT
            else context[source.channel_index]
        )
        contribution = _q10_product(signal, edge.weight_q10)
        if (
            edge.source_node_id == "context-input-0"
            and edge.target_node_id == "regulator-0"
        ):
            contribution = _q10_product(
                contribution,
                Q10_SCALE + _trunc_div(modifiers[8], 4),
            )
        elif (
            edge.source_node_id == "context-input-5"
            and edge.target_node_id == "regulator-2"
        ):
            contribution = _q10_product(
                contribution,
                Q10_SCALE + _trunc_div(modifiers[6], 4),
            )
        elif (
            edge.source_node_id == "context-input-7"
            and edge.target_node_id == "regulator-3"
        ):
            contribution = _q10_product(
                contribution,
                Q10_SCALE + _trunc_div(modifiers[10], 4),
            )
        if edge.effect == EDGE_INHIBIT:
            contribution = -abs(contribution)
        regulator_accumulators[target.channel_index] += contribution
        active_input_edges += int(contribution != 0)

    regulators = [0] * REGULATOR_COUNT
    threshold_modifier_indices = (1, 3, 5, 7)
    for node in network.nodes:
        if node.kind == NODE_REGULATOR:
            threshold = min(
                Q10_SCALE // 4,
                max(
                    0,
                    node.threshold_q10
                    + _trunc_div(
                        modifiers[
                            threshold_modifier_indices[node.channel_index]
                        ],
                        8,
                    ),
                ),
            )
            regulators[node.channel_index] = min(
                Q10_SCALE,
                max(0, regulator_accumulators[node.channel_index] - threshold),
            )

    regulated = list(epistatic_products)
    active_output_edges = 0
    for edge in network.edges:
        target = node_by_id[edge.target_node_id]
        if target.kind != NODE_PRODUCT_OUTPUT:
            continue
        source = node_by_id[edge.source_node_id]
        signal = regulators[source.channel_index]
        if signal == 0:
            continue
        contribution = _q10_product(signal, edge.weight_q10)
        if edge.effect == EDGE_INHIBIT:
            regulated[target.channel_index] -= abs(contribution)
        elif edge.effect == EDGE_ACTIVATE:
            regulated[target.channel_index] += contribution
        else:
            regulated[target.channel_index] += _q10_product(
                regulated[target.channel_index], contribution
            )
        regulated[target.channel_index] = _clamp_q10(
            regulated[target.channel_index]
        )
        active_output_edges += 1

    active_regulators = sum(value != 0 for value in regulators)
    active_edges = (
        active_epistatic_edges + active_input_edges + active_output_edges
    )
    signal_cost = sum(
        _q10_product(abs(value), network.signal_cost_q10) for value in regulators
    )
    regulatory_cost = (
        active_regulators * network.active_node_cost_q10
        + active_edges * network.active_edge_cost_q10
        + signal_cost
    )
    if regulatory_cost > network.regulatory_budget_q10:
        raise GenomeUpgradeV2RegulatoryError(
            "regulatory execution exceeds its fixed Q10 budget"
        )
    return RegulatoryExpressionV1(
        regulated_products_q10=tuple(regulated),
        regulator_nodes_q10=tuple(regulators),
        epistatic_delta_q10=tuple(
            _clamp_q10(value) for value in epistatic_delta
        ),
        regulatory_cost_q10=regulatory_cost,
        active_regulator_count=active_regulators,
        active_edge_count=active_edges,
        developmental_stage=stage,
    )


def express_regulatory_tensor_v1(
    products_q10: Any,
    context_q10: Any,
    modifiers_q10: Any,
) -> dict[str, Any]:
    """Apply the frozen regulator to matching int64 tensor batches."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise GenomeUpgradeV2RegulatoryError("torch is required") from exc
    tensors = (products_q10, context_q10, modifiers_q10)
    if any(not isinstance(value, torch.Tensor) for value in tensors):
        raise GenomeUpgradeV2RegulatoryError(
            "regulatory tensor execution requires torch tensors"
        )
    if (
        products_q10.shape[-1] != PRODUCT_COUNT
        or context_q10.shape[-1] != CONTEXT_COUNT
        or modifiers_q10.shape[-1] != PRODUCT_COUNT
        or products_q10.shape[:-1] != context_q10.shape[:-1]
        or products_q10.shape[:-1] != modifiers_q10.shape[:-1]
    ):
        raise GenomeUpgradeV2RegulatoryError(
            "regulatory tensor shapes differ"
        )
    if (
        products_q10.dtype != torch.int64
        or context_q10.dtype != torch.int64
        or modifiers_q10.dtype != torch.int64
        or products_q10.device != context_q10.device
        or products_q10.device != modifiers_q10.device
    ):
        raise GenomeUpgradeV2RegulatoryError(
            "regulatory tensors require matching int64 storage"
        )
    if bool(
        (
            (context_q10[..., CONTEXT_CHANNELS.index("juvenile")] > 0)
            & (context_q10[..., CONTEXT_CHANNELS.index("senescent")] > 0)
        ).any()
    ):
        raise GenomeUpgradeV2RegulatoryError(
            "juvenile and senescent context cannot both be active"
        )

    def q10(*values: Any) -> Any:
        result: Any = Q10_SCALE
        for value in values:
            result = torch.div(
                result * value, Q10_SCALE, rounding_mode="trunc"
            )
        return result

    prepared_products = products_q10.clone()
    stability_factor = torch.clamp(
        Q10_SCALE
        + torch.div(
            modifiers_q10[..., 2], 4, rounding_mode="trunc"
        ),
        MODIFIER_FACTOR_MIN_Q10,
        MODIFIER_FACTOR_MAX_Q10,
    )
    prepared_products[..., 2] = torch.clamp(
        q10(prepared_products[..., 2], stability_factor),
        -Q10_SCALE,
        Q10_SCALE,
    )
    saturation_limit = Q10_SCALE - torch.div(
        modifiers_q10[..., 11].abs(), 8, rounding_mode="trunc"
    )
    prepared_products[..., 7] = torch.minimum(
        saturation_limit,
        torch.maximum(
            -saturation_limit, prepared_products[..., 7]
        ),
    )
    products_q10 = prepared_products

    epistatic_delta = torch.zeros_like(products_q10)
    active_epistatic = torch.zeros(
        products_q10.shape[:-1],
        dtype=torch.int64,
        device=products_q10.device,
    )
    for edge in DEFAULT_FIXED_REGULATORY_NETWORK_V1.epistatic_edges:
        magnitude = q10(
            products_q10[..., edge.source_product_index],
            modifiers_q10[..., edge.modifier_index],
            edge.weight_q10,
        )
        delta = magnitude.abs()
        if edge.effect == EDGE_INHIBIT:
            delta = -delta
        epistatic_delta[..., edge.target_product_index] += delta
        active_epistatic += (delta != 0).to(dtype=torch.int64)
    epistatic_products = torch.clamp(
        products_q10 + epistatic_delta, -Q10_SCALE, Q10_SCALE
    )

    regulators = torch.zeros(
        (*products_q10.shape[:-1], REGULATOR_COUNT),
        dtype=torch.int64,
        device=products_q10.device,
    )
    active_input = torch.zeros_like(active_epistatic)
    nodes = {
        node.node_id: node
        for node in DEFAULT_FIXED_REGULATORY_NETWORK_V1.nodes
    }
    for edge in DEFAULT_FIXED_REGULATORY_NETWORK_V1.edges:
        target = nodes[edge.target_node_id]
        if target.kind != NODE_REGULATOR:
            continue
        source = nodes[edge.source_node_id]
        signal = (
            epistatic_products[..., source.channel_index]
            if source.kind == NODE_PRODUCT_INPUT
            else context_q10[..., source.channel_index]
        )
        contribution = q10(signal, edge.weight_q10)
        if (
            edge.source_node_id == "context-input-0"
            and edge.target_node_id == "regulator-0"
        ):
            contribution = q10(
                contribution,
                Q10_SCALE
                + torch.div(
                    modifiers_q10[..., 8],
                    4,
                    rounding_mode="trunc",
                ),
            )
        elif (
            edge.source_node_id == "context-input-5"
            and edge.target_node_id == "regulator-2"
        ):
            contribution = q10(
                contribution,
                Q10_SCALE
                + torch.div(
                    modifiers_q10[..., 6],
                    4,
                    rounding_mode="trunc",
                ),
            )
        elif (
            edge.source_node_id == "context-input-7"
            and edge.target_node_id == "regulator-3"
        ):
            contribution = q10(
                contribution,
                Q10_SCALE
                + torch.div(
                    modifiers_q10[..., 10],
                    4,
                    rounding_mode="trunc",
                ),
            )
        if edge.effect == EDGE_INHIBIT:
            contribution = -contribution.abs()
        regulators[..., target.channel_index] += contribution
        active_input += (contribution != 0).to(dtype=torch.int64)
    base_thresholds = torch.tensor(
        [
            node.threshold_q10
            for node in DEFAULT_FIXED_REGULATORY_NETWORK_V1.nodes
            if node.kind == NODE_REGULATOR
        ],
        dtype=torch.int64,
        device=products_q10.device,
    )
    threshold_indices = torch.tensor(
        (1, 3, 5, 7),
        dtype=torch.int64,
        device=products_q10.device,
    )
    thresholds = torch.clamp(
        base_thresholds
        + torch.div(
            modifiers_q10.index_select(-1, threshold_indices),
            8,
            rounding_mode="trunc",
        ),
        0,
        Q10_SCALE // 4,
    )
    regulators = torch.clamp(
        regulators - thresholds, 0, Q10_SCALE
    )

    regulated = epistatic_products.clone()
    active_output = torch.zeros_like(active_epistatic)
    for edge in DEFAULT_FIXED_REGULATORY_NETWORK_V1.edges:
        target = nodes[edge.target_node_id]
        if target.kind != NODE_PRODUCT_OUTPUT:
            continue
        source = nodes[edge.source_node_id]
        signal = regulators[..., source.channel_index]
        contribution = q10(signal, edge.weight_q10)
        if edge.effect == EDGE_INHIBIT:
            delta = -contribution.abs()
        elif edge.effect == EDGE_ACTIVATE:
            delta = contribution
        else:
            delta = q10(regulated[..., target.channel_index], contribution)
        regulated[..., target.channel_index] = torch.clamp(
            regulated[..., target.channel_index] + delta,
            -Q10_SCALE,
            Q10_SCALE,
        )
        active_output += (signal != 0).to(dtype=torch.int64)
    active_regulators = (regulators != 0).sum(dim=-1)
    active_edges = active_epistatic + active_input + active_output
    signal_cost = torch.div(
        regulators.abs()
        * DEFAULT_FIXED_REGULATORY_NETWORK_V1.signal_cost_q10,
        Q10_SCALE,
        rounding_mode="trunc",
    ).sum(dim=-1)
    regulatory_cost = (
        active_regulators
        * DEFAULT_FIXED_REGULATORY_NETWORK_V1.active_node_cost_q10
        + active_edges
        * DEFAULT_FIXED_REGULATORY_NETWORK_V1.active_edge_cost_q10
        + signal_cost
    )
    if bool(
        (
            regulatory_cost
            > DEFAULT_FIXED_REGULATORY_NETWORK_V1.regulatory_budget_q10
        ).any()
    ):
        raise GenomeUpgradeV2RegulatoryError(
            "regulatory execution exceeds its fixed Q10 budget"
        )
    return {
        "regulated_products_q10": regulated,
        "regulator_nodes_q10": regulators,
        "epistatic_delta_q10": torch.clamp(
            epistatic_delta, -Q10_SCALE, Q10_SCALE
        ),
        "regulatory_cost_q10": regulatory_cost,
        "active_regulator_count": active_regulators,
        "active_regulatory_edge_count": active_edges,
    }


def pleiotropic_influence_budget_v1(
    weights_q10: Sequence[Sequence[Any]] = POLYGENIC_WEIGHTS_Q10,
    *,
    budget_q10: int = PLEIOTROPIC_INFLUENCE_BUDGET_Q10,
) -> dict[str, Any]:
    if (
        isinstance(budget_q10, bool)
        or not isinstance(budget_q10, int)
        or budget_q10 <= 0
    ):
        raise GenomeUpgradeV2RegulatoryError(
            "pleiotropic influence budget must be positive"
        )
    if len(weights_q10) == 0:
        raise GenomeUpgradeV2RegulatoryError("pleiotropic mapping is empty")
    rows = [
        _validated_q10_vector(row, count=PRODUCT_COUNT, name=f"weights_q10[{index}]")
        for index, row in enumerate(weights_q10)
    ]
    influence = tuple(
        sum(abs(row[product_index]) for row in rows)
        for product_index in range(PRODUCT_COUNT)
    )
    contribution_counts = tuple(
        sum(row[product_index] != 0 for row in rows)
        for product_index in range(PRODUCT_COUNT)
    )
    if any(value > budget_q10 for value in influence):
        raise GenomeUpgradeV2RegulatoryError(
            "pleiotropic influence exceeds its per-product budget"
        )
    if any(value < 2 for value in contribution_counts):
        raise GenomeUpgradeV2RegulatoryError(
            "every product must influence at least two phenotype outputs"
        )
    tradeoff_products = tuple(
        product_index
        for product_index in range(PRODUCT_COUNT)
        if any(row[product_index] > 0 for row in rows)
        and any(row[product_index] < 0 for row in rows)
    )
    if not tradeoff_products:
        raise GenomeUpgradeV2RegulatoryError(
            "pleiotropic mapping has no signed trade-off"
        )
    return {
        "budget_q10": budget_q10,
        "influence_q10_by_product": influence,
        "contribution_count_by_product": contribution_counts,
        "tradeoff_product_indices": tradeoff_products,
    }


def regulatory_contract_payload_v1() -> dict[str, Any]:
    network = DEFAULT_FIXED_REGULATORY_NETWORK_V1
    influence = pleiotropic_influence_budget_v1()
    return {
        "schema_version": REGULATORY_SCHEMA_VERSION,
        "numeric_contract": "signed_q10_int64_truncation_toward_zero_v1",
        "product_channels": list(PRODUCT_CHANNELS),
        "context_channels": list(CONTEXT_CHANNELS),
        "regulator_count": REGULATOR_COUNT,
        "maximum_fan_in": network.maximum_fan_in,
        "update_schedule": network.update_schedule,
        "nodes": [
            {
                "node_id": node.node_id,
                "kind": node.kind,
                "channel_index": node.channel_index,
                "threshold_q10": node.threshold_q10,
            }
            for node in network.nodes
        ],
        "edges": [
            {
                "source_node_id": edge.source_node_id,
                "target_node_id": edge.target_node_id,
                "effect": edge.effect,
                "weight_q10": edge.weight_q10,
            }
            for edge in network.edges
        ],
        "epistatic_edges": [
            {
                "source_product_index": edge.source_product_index,
                "modifier_index": edge.modifier_index,
                "target_product_index": edge.target_product_index,
                "effect": edge.effect,
                "weight_q10": edge.weight_q10,
            }
            for edge in network.epistatic_edges
        ],
        "modifier_mechanisms": {
            "quantity": "existing_same_product_output_offset",
            "stability_degradation": "modifier_2_bounded_product_2_gain",
            "pathway_threshold": (
                "modifiers_1_3_5_7_shift_four_regulator_thresholds"
            ),
            "receptor_sensitivity": (
                "modifier_6_scales_hazard_input_to_regulator_2"
            ),
            "saturation_point": "modifier_11_bounds_product_7_output",
            "activation_gain": (
                "modifier_8_scales_energy_deficit_activation"
            ),
            "inhibition_gain": (
                "modifier_10_scales_senescent_inhibition"
            ),
            "free_dominance_scalar": False,
        },
        "costs_q10": {
            "active_node": network.active_node_cost_q10,
            "active_edge": network.active_edge_cost_q10,
            "signal": network.signal_cost_q10,
            "budget": network.regulatory_budget_q10,
        },
        "developmental_context": {
            "juvenile_index": CONTEXT_CHANNELS.index("juvenile"),
            "senescent_index": CONTEXT_CHANNELS.index("senescent"),
            "mutual_exclusion": True,
        },
        "pleiotropic_influence": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in influence.items()
        },
        "hidden_rng_draws": 0,
    }


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


REGULATORY_CONTRACT_SHA256: Final[str] = hashlib.sha256(
    _canonical_json_bytes(regulatory_contract_payload_v1())
).hexdigest()
