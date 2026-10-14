import pytest

from engine.campaigns.interesting_checkpoints import (
    DEFAULT,
    DisposableCheckpointDependencyError,
    reject_disposable_checkpoint_dependencies,
)


@pytest.mark.parametrize("value", [
    {"inputs": [{"checkpoint_path": str(DEFAULT / "tick-100")} ]},
    {"restore": DEFAULT},
    {"source": "file:///tmp/emergence-test/storage/emergence/interesting-ckpt/tick-100"},
    {"inputs": "file://localhost/tmp/emergence-test/storage/emergence/interesting-ckpt/tick-100"},
])
def test_disposable_inputs_rejected(value):
    with pytest.raises(DisposableCheckpointDependencyError, match="disposable"):
        reject_disposable_checkpoint_dependencies(value)


def test_resolved_alias_is_rejected(tmp_path):
    alias = tmp_path / "apparently-durable"
    alias.symlink_to(DEFAULT)
    with pytest.raises(DisposableCheckpointDependencyError):
        reject_disposable_checkpoint_dependencies({"input": str(alias / "tick-100")})


def test_relative_alias_resolves_from_request_directory(tmp_path):
    (tmp_path / "state").symlink_to(DEFAULT)
    with pytest.raises(DisposableCheckpointDependencyError):
        reject_disposable_checkpoint_dependencies({"checkpoint_path": "state/tick-100"}, base_dir=tmp_path)


def test_descriptions_and_unrelated_paths_allowed():
    reject_disposable_checkpoint_dependencies({
        "description": str(DEFAULT),
        "question": f"Can we save a copy under {DEFAULT}?",
        "source": "/tmp/emergence-test/storage/emergence/interesting-ckpt-other/checkpoint",
        "inputs": ["/tmp/emergence-test/runtime/runs/new/checkpoint"],
    })


def test_does_not_create_stash(tmp_path, monkeypatch):
    from engine.campaigns import interesting_checkpoints as module
    root = tmp_path / "disposable"
    monkeypatch.setattr(module, "DEFAULT", root)
    module.reject_disposable_checkpoint_dependencies({"input": str(tmp_path / "regular")})
    assert not root.exists()


def test_guard_still_rejects_after_optional_directory_is_removed(tmp_path, monkeypatch):
    from engine.campaigns import interesting_checkpoints as module
    root = tmp_path / "optional"
    root.mkdir()
    monkeypatch.setattr(module, "DEFAULT", root)
    for present in (True, False):
        if not present:
            root.rmdir()
        with pytest.raises(DisposableCheckpointDependencyError):
            module.reject_disposable_checkpoint_dependencies({"checkpoint_path": str(root / "tick-100")})
        module.reject_disposable_checkpoint_dependencies({"input": str(tmp_path / "fresh")})
