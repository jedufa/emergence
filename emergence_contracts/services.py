"""C-020 narrow Runtime Services infrastructure ports."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .artifacts import RunArtifactRefV1
from .capabilities import RuntimeCapabilityProfileV1
from .control import ControlPortV1
from .events import TelemetrySinkV1
from .ids import ComponentId, SchemaVersion
from .payloads import CanonicalPayloadV1
from .runtime import RunCursorV1
from .state import RNGAuthorityV1


CONTRACT_ID = "C-020"
CONTRACT_OWNER = "Contracts; provided by Runtime"
API_VERSION = SchemaVersion("c020_v1")


@runtime_checkable
class ArtifactOutputPortV1(Protocol):
    def publish(
        self, *, artifact_kind: str, payload: CanonicalPayloadV1
    ) -> RunArtifactRefV1: ...


@runtime_checkable
class ControlBoundaryRegistrarV1(Protocol):
    def register_boundary(
        self, *, owner_component_id: ComponentId, boundary_name: str
    ) -> None: ...


@runtime_checkable
class RuntimeServicesV1(Protocol):
    """No concrete runner, config bag, world, population, or backend state."""

    def cursor(self) -> RunCursorV1: ...

    def rng_authority(self) -> RNGAuthorityV1: ...

    def telemetry_sink(self) -> TelemetrySinkV1: ...

    def artifact_output(self) -> ArtifactOutputPortV1: ...

    def control_port(self) -> ControlPortV1: ...

    def control_boundaries(self) -> ControlBoundaryRegistrarV1: ...

    def capabilities(self) -> RuntimeCapabilityProfileV1: ...
