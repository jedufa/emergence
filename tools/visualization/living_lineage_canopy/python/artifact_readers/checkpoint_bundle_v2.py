from __future__ import annotations

import gzip
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DTYPES = {
    "bool": "<?",
    "int8": "<b",
    "uint8": "<B",
    "int16": "<h",
    "uint16": "<H",
    "int32": "<i",
    "uint32": "<I",
    "int64": "<q",
    "uint64": "<Q",
    "float32": "<f",
    "float64": "<d",
}


@dataclass(frozen=True)
class ProjectionSnapshot:
    path: Path
    generation: int
    absolute_tick: int | None
    ticks_per_generation: int | None
    event_boundary_tick: int
    alive_agent_ids: tuple[int, ...]
    lineage_by_agent_id: dict[int, int]
    depth_by_agent_id: dict[int, int]
    source_generation: int
    source_tick: int
    projection: dict[str, Any]


def load_projection(path: Path) -> dict[str, Any]:
    open_fn = gzip.open if path.name.endswith(".gz") else open
    with open_fn(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def tensor_values(projection: dict[str, Any], name: str) -> list[Any]:
    tensor = projection["projection"]["tensors"][name]
    dtype = tensor["dtype"]
    if dtype not in DTYPES:
        raise ValueError(f"unsupported tensor dtype {dtype!r} for {name}")
    raw = bytes.fromhex(tensor["data_hex"])
    fmt = DTYPES[dtype]
    size = struct.calcsize(fmt)
    return [struct.unpack(fmt, raw[index : index + size])[0] for index in range(0, len(raw), size)]


def bundle_generation(path: Path) -> int:
    match = re.search(r"bundle_gen(\d+)", str(path))
    if not match:
        raise ValueError(f"cannot derive bundle generation from {path}")
    return int(match.group(1))


def bundle_absolute_tick(path: Path) -> int | None:
    match = re.search(r"bundle_tick(\d+)", str(path))
    return int(match.group(1)) if match else None


def snapshot_boundary(snapshot: ProjectionSnapshot) -> int:
    if snapshot.absolute_tick is not None:
        return snapshot.absolute_tick
    return (
        snapshot.event_boundary_tick + 1
        if snapshot.event_boundary_tick
        else snapshot.generation
    )


def snapshot_order_key(snapshot: ProjectionSnapshot) -> tuple[int, int, int]:
    return (
        snapshot_boundary(snapshot),
        snapshot.source_generation,
        snapshot.source_tick,
    )


def projection_paths_for_cell(run_root: Path, cell_id: str | None = None) -> list[Path]:
    patterns = [
        "**/checkpoint_bundles_v2/*/bundle_gen*/native_runtime_checkpoint_projection.json.gz",
        "**/checkpoint_bundles_v2/*/bundle_gen*/native_runtime_checkpoint_projection.json",
        "**/checkpoint_bundles_v2/*/bundle_tick*/native_runtime_checkpoint_projection.json.gz",
        "**/checkpoint_bundles_v2/*/bundle_tick*/native_runtime_checkpoint_projection.json",
    ]
    all_paths = sorted(path for pattern in patterns for path in run_root.rglob(pattern))
    primary = [
        path
        for path in all_paths
        if "/comparisons/" not in str(path) and "/technical_gate/" not in str(path) and "/route_qualification/" not in str(path)
    ]
    paths = primary or all_paths
    if cell_id is not None:
        paths = [path for path in paths if f"checkpoint_bundles_v2/{cell_id}/" in str(path)]
    return paths


def choose_cell(run_root: Path) -> str:
    cells: dict[str, int] = {}
    for path in projection_paths_for_cell(run_root):
        parts = path.parts
        try:
            index = parts.index("checkpoint_bundles_v2")
        except ValueError:
            continue
        cell_id = parts[index + 1]
        coordinate = bundle_absolute_tick(path)
        if coordinate is None:
            coordinate = bundle_generation(path)
        cells[cell_id] = max(cells.get(cell_id, 0), coordinate)
    if not cells:
        raise ValueError(f"no checkpoint bundle v2 projections found under {run_root}")
    return sorted(cells.items(), key=lambda item: (-item[1], item[0]))[0][0]


def snapshot_from_projection(path: Path) -> ProjectionSnapshot:
    data = load_projection(path)
    slot_to_agent = tensor_values(data, "slot_to_agent_id")
    founders = tensor_values(data, "lineage_founder_id")
    depths = tensor_values(data, "lineage_depth")
    alive_ids = tuple(int(value) for value in tensor_values(data, "agent_ids_by_alive_order") if int(value) >= 0)
    lineage_by_agent_id: dict[int, int] = {}
    depth_by_agent_id: dict[int, int] = {}
    for slot, agent_id in enumerate(slot_to_agent):
        if int(agent_id) < 0:
            continue
        lineage_by_agent_id[int(agent_id)] = int(founders[slot])
        depth_by_agent_id[int(agent_id)] = int(depths[slot])
    absolute_tick = bundle_absolute_tick(path)
    if absolute_tick is None:
        generation = bundle_generation(path)
        ticks_per_generation = None
    else:
        generation = data.get("source_generation")
        source_tick = data.get("source_tick")
        metadata = data.get("projection", {}).get("metadata", {})
        ticks_per_generation = metadata.get("configured_steps_per_generation")
        if (
            type(generation) is not int
            or generation < 0
            or type(source_tick) is not int
            or type(ticks_per_generation) is not int
            or ticks_per_generation < 1
            or not 0 <= source_tick < ticks_per_generation
            or generation * ticks_per_generation + source_tick != absolute_tick
        ):
            raise ValueError(
                f"tick-native checkpoint cursor contradicts bundle name: {path}"
            )
    event_boundary_tick = 0
    if data["projection"]["tensors"].get("typed_lifecycle_event_sequence"):
        sequences = tensor_values(data, "typed_lifecycle_event_sequence")
        event_boundary_tick = (max(int(value) for value in sequences) + 1) if sequences else 0
    return ProjectionSnapshot(
        path=path,
        generation=generation,
        absolute_tick=absolute_tick,
        ticks_per_generation=ticks_per_generation,
        event_boundary_tick=event_boundary_tick,
        alive_agent_ids=alive_ids,
        lineage_by_agent_id=lineage_by_agent_id,
        depth_by_agent_id=depth_by_agent_id,
        source_generation=int(
            data["source_generation"]
            if data.get("source_generation") is not None
            else generation
        ),
        source_tick=int(
            data["source_tick"] if data.get("source_tick") is not None else 0
        ),
        projection=data,
    )


def typed_lifecycle_events(projection: dict[str, Any]) -> list[dict[str, int]]:
    tensors = projection["projection"]["tensors"]
    required = [
        "typed_lifecycle_event_event_type",
        "typed_lifecycle_event_success",
        "typed_lifecycle_event_sequence",
        "typed_lifecycle_event_generation",
        "typed_lifecycle_event_tick",
        "typed_lifecycle_event_parent_agent_id",
        "typed_lifecycle_event_child_agent_id",
        "typed_lifecycle_event_target_agent_id",
        "typed_lifecycle_event_lineage_founder_id",
        "typed_lifecycle_event_lineage_depth",
    ]
    if any(name not in tensors for name in required):
        return []
    columns = {name: tensor_values(projection, name) for name in required}
    rows = []
    for index in range(len(columns[required[0]])):
        rows.append(
            {
                "event_type": int(columns["typed_lifecycle_event_event_type"][index]),
                "success": int(columns["typed_lifecycle_event_success"][index]),
                "sequence": int(columns["typed_lifecycle_event_sequence"][index]),
                "generation": int(columns["typed_lifecycle_event_generation"][index]),
                "tick": int(columns["typed_lifecycle_event_tick"][index]),
                "parent_agent_id": int(columns["typed_lifecycle_event_parent_agent_id"][index]),
                "child_agent_id": int(columns["typed_lifecycle_event_child_agent_id"][index]),
                "target_agent_id": int(columns["typed_lifecycle_event_target_agent_id"][index]),
                "lineage_founder_id": int(columns["typed_lifecycle_event_lineage_founder_id"][index]),
                "lineage_depth": int(columns["typed_lifecycle_event_lineage_depth"][index]),
            }
        )
    return rows
