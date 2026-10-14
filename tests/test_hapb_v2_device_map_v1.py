from __future__ import annotations

import pytest

from engine.perf.hapb_v2.device_map_v1 import (
    HapbV2DeviceHotMapV1,
    HapbV2DeviceMapError,
)


def test_alias_clone_version_bump_invalidation_and_aba() -> None:
    device_map = HapbV2DeviceHotMapV1(cells=2, slots_per_cell=4)
    device_map.publish_binding_v1(
        cell=0, slot=0, content_id=11, content_version=3
    )
    generation = int(device_map.slot_generations[0, 0])
    device_map.alias_clone_v1(cell=0, parent_slot=0, child_slot=1)
    assert int(device_map.content_ids[0, 1]) == 11
    assert int(device_map.content_versions[0, 1]) == 3
    device_map.require_live_binding_v1(
        cell=0,
        slot=0,
        expected_content_id=11,
        expected_version=3,
        expected_slot_generation=generation,
    )
    with pytest.raises(HapbV2DeviceMapError, match="stale content version"):
        device_map.require_live_binding_v1(
            cell=0, slot=0, expected_content_id=11, expected_version=2
        )
    device_map.invalidate_slot_v1(0, 1)
    with pytest.raises(HapbV2DeviceMapError, match="invalid"):
        device_map.require_live_binding_v1(
            cell=0, slot=1, expected_content_id=11, expected_version=3
        )
    with pytest.raises(HapbV2DeviceMapError, match="ABA"):
        device_map.require_live_binding_v1(
            cell=0,
            slot=0,
            expected_content_id=11,
            expected_version=3,
            expected_slot_generation=generation - 1,
        )


def test_checkpoint_roundtrip_and_fresh_reconstruction() -> None:
    device_map = HapbV2DeviceHotMapV1(cells=1, slots_per_cell=3)
    device_map.publish_binding_v1(
        cell=0, slot=2, content_id=99, content_version=1
    )
    payload = device_map.checkpoint_metadata_v1()
    restored = HapbV2DeviceHotMapV1.restore_checkpoint_metadata_v1(payload)
    assert restored.checkpoint_metadata_v1()["sha256"] == payload["sha256"]
    rebuilt = HapbV2DeviceHotMapV1(cells=1, slots_per_cell=3)
    rebuilt.reconstruct_from_host_v1(
        [{"cell": 0, "slot": 2, "content_id": 99, "content_version": 1}],
        generation_epoch=4,
    )
    assert rebuilt.generation_epoch == 4
    assert int(rebuilt.content_ids[0, 2]) == 99
