from __future__ import annotations

from pathlib import Path


H4 = (
    Path(__file__).resolve().parents[1]
    / "tools/scripts/eiel_b48_wdl_jungle_eed_hapb_h4_continuation_v1.py"
)


def test_alias_off_owner_is_exported_before_production_owner_is_restored() -> None:
    """G3-exact/G6-1-cell miss: export must see the alias-off owner.

    208/209 restored the production alias-on owner in the executor finally
    block, then exported the midpoint.  The envelope recorded
    content_alias_enabled=True and stale alias maps after an alias-off
    execute.  G3 projections still matched; one cell drifted on G3-G6
    resume (cell 44 then 43).
    """

    text = H4.read_text(encoding="utf-8")
    execute_at = text.index("def execute(")
    export_at = text.index(
        "envelope = export_direct_resident_wdl_jungle_hapb_checkpoint_v1",
        execute_at,
    )
    restore_at = text.index(
        'runtime_source.live_banks["hapb_policy_owner"] = original_owner',
        execute_at,
    )
    bind_at = text.index("invalidate_hot_prefix", execute_at)
    replace_at = text.index(
        'runtime_source.live_banks["hapb_policy_owner"] = replace(',
        execute_at,
    )
    assert "component_checkpoint is not None" in text
    assert "content_alias_enabled=False" in text[replace_at:export_at]
    assert bind_at < replace_at < export_at < restore_at
