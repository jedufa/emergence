"""Campaign integration surface for direct-resident ACS capture/settlement.

The physical producer and detached settlement worker share this boundary so
their persistence contract does not accumulate independent imports into the
performance implementation.  It contains no policy or state of its own.
"""

from __future__ import annotations

from engine.perf.cuda_architecture_b_boundary import EVENT_COLUMN_NAMES
from engine.perf.cuda_architecture_b_direct_resident_eed_v1 import (
    EED_STATE_FIELDS_V1,
)
from engine.perf.cuda_architecture_b_direct_resident_jungle_v4 import (
    JUNGLE_STATE_FIELDS_V4,
)
from engine.perf.cuda_architecture_b_direct_resident_patch_v1 import (
    DETACHED_CELL_PROJECTION_SCHEMA_VERSION_V1,
    CheckpointProjectionCursorV1,
    DetachedCellProjectionV1,
)
from engine.perf.cuda_architecture_b_direct_resident_runtime_v1 import (
    RESULT_SCHEMA_VERSION,
)
from engine.perf.cuda_architecture_b_direct_resident_wdl_jungle_v1 import (
    PreparedDirectResidentWdlJungleV1,
    project_direct_resident_wdl_jungle_multicell_settlement_v1,
)
from engine.perf.cuda_architecture_b_direct_resident_wdl_v2 import (
    PreparedDirectResidentWdlV2,
    WDL_STATE_FIELDS_V2,
)
from engine.perf.cuda_architecture_b_generation import CudaGenerationConfig
from engine.perf.cuda_architecture_b_generation_birth_state import (
    CudaGenerationBirthState,
    CudaGenerationBirthStateReceipt,
)


__all__ = [
    "CheckpointProjectionCursorV1",
    "CudaGenerationBirthState",
    "CudaGenerationBirthStateReceipt",
    "CudaGenerationConfig",
    "DETACHED_CELL_PROJECTION_SCHEMA_VERSION_V1",
    "DetachedCellProjectionV1",
    "EED_STATE_FIELDS_V1",
    "EVENT_COLUMN_NAMES",
    "JUNGLE_STATE_FIELDS_V4",
    "PreparedDirectResidentWdlJungleV1",
    "PreparedDirectResidentWdlV2",
    "RESULT_SCHEMA_VERSION",
    "WDL_STATE_FIELDS_V2",
    "project_direct_resident_wdl_jungle_multicell_settlement_v1",
]
