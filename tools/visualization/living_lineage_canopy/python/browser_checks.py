from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _finite_snapshot(snapshot: dict[str, object]) -> bool:
    for group in ("particles", "lineageBodies", "branchBodies"):
        for row in snapshot[group]:
            for key in ("x", "y", "vx", "vy", "radius"):
                if key in row and not math.isfinite(row[key]):
                    return False
    return True


def _avg_neighbor_distance(snapshot: dict[str, object], lineage_id: str, center: dict[str, float]) -> float:
    values = []
    for body in snapshot["lineageBodies"]:
        if body["id"] != lineage_id and body["count"] > 0:
            values.append(math.hypot(body["x"] - center["x"], body["y"] - center["y"]))
    return sum(values) / len(values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run browser-executed Living Lineage Canopy diagnostics")
    parser.add_argument("--base-url", default="http://127.0.0.1:8769/index.html")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--external-playback", type=Path)
    parser.add_argument("--external-provenance", type=Path)
    parser.add_argument("--relay-manifest")
    args = parser.parse_args(argv)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit("Playwright is required: python -m pip install playwright && python -m playwright install chromium") from exc

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        page = browser.new_page(viewport={"width": 1440, "height": 980}, device_scale_factor=1)
        page.goto(f"{args.base_url}?dataset=balanced&tick=120", wait_until="networkidle")
        page.wait_for_timeout(600)
        first = page.evaluate("async () => { await window.__LLC_TEST_API.load('balanced', 120); return window.__LLC_TEST_API.stepFixed(80); }")
        page.reload(wait_until="networkidle")
        page.wait_for_timeout(600)
        second = page.evaluate("async () => { await window.__LLC_TEST_API.load('balanced', 120); return window.__LLC_TEST_API.stepFixed(80); }")
        deterministic = first == second
        finite = _finite_snapshot(first)
        page.close()

        p12b = browser.new_page(viewport={"width": 1440, "height": 980}, device_scale_factor=1)
        p12b.goto(f"{args.base_url}?dataset=p12b&tick=1200", wait_until="networkidle")
        p12b.wait_for_timeout(600)
        partial = p12b.evaluate(
            """
            () => ({
              metrics: window.__LLC_METRICS,
              livingLabel: livingMetricLabel.textContent,
              shareLabel: shareMetricLabel.textContent,
              warningHidden: semanticPanel.hidden,
              warningText: semanticNotice.textContent,
              dyingDisabled: dyingToggle.disabled
            })
            """
        )
        p12b.close()

        reclamation_page = browser.new_page(viewport={"width": 1440, "height": 980}, device_scale_factor=1)
        reclamation_page.goto(f"{args.base_url}?dataset=boom_collapse&tick=35", wait_until="networkidle")
        reclamation_page.wait_for_timeout(600)
        sequence = reclamation_page.evaluate(
            """
            async () => {
              await window.__LLC_TEST_API.load('boom_collapse', 35);
              window.__LLC_TEST_API.stepFixed(80);
              window.__LLC_TEST_API.advanceTicks(115, 4);
              const peak = window.__LLC_TEST_API.snapshot();
              const body = peak.lineageBodies.find((item) => item.id === 'lineage_2');
              window.__LLC_TEST_API.advanceTicks(95, 4);
              const late = window.__LLC_TEST_API.snapshot();
              return {peak, late, body};
            }
            """
        )
        collapsed = sequence["body"]
        late_body = next((body for body in sequence["late"]["lineageBodies"] if body["id"] == "lineage_2"), None)
        peak_distance = _avg_neighbor_distance(sequence["peak"], "lineage_2", collapsed)
        late_distance = _avg_neighbor_distance(sequence["late"], "lineage_2", collapsed)
        reclamation_page.close()

        stress = browser.new_page(viewport={"width": 1440, "height": 980}, device_scale_factor=1)
        stress.goto(f"{args.base_url}?dataset=stress&tick=190", wait_until="networkidle")
        stress.wait_for_timeout(1600)
        stress_metrics = stress.evaluate("window.__LLC_METRICS")
        stress_snapshot = stress.evaluate("window.__LLC_TEST_API.snapshot()")
        stress.close()

        external_result = None
        if args.external_playback:
            external_data = json.loads(args.external_playback.read_text())
            external_provenance = json.loads(args.external_provenance.read_text()) if args.external_provenance else None
            external = browser.new_page(viewport={"width": 1440, "height": 980}, device_scale_factor=1)
            external.goto(f"{args.base_url}?dataset=balanced&tick=0", wait_until="networkidle")
            external.wait_for_timeout(600)
            external.set_input_files("#playbackFileInput", str(args.external_playback))
            external.wait_for_function("() => window.__LLC_METRICS?.external_loaded === true")
            loaded_by_file = external.evaluate("window.__LLC_TEST_API.metrics()")
            without_provenance = external.evaluate("({metrics: window.__LLC_TEST_API.metrics(), warningText: semanticNotice.textContent})")
            verified_by_file = None
            if args.external_provenance:
                external.set_input_files("#provenanceFileInput", str(args.external_provenance))
                external.wait_for_function("() => window.__LLC_METRICS?.provenance_verified === true")
                verified_by_file = external.evaluate("({metrics: window.__LLC_TEST_API.metrics(), warningText: semanticNotice.textContent})")
            external_result = external.evaluate(
                """
                async ({data, provenance}) => {
                  const before = await window.__LLC_TEST_API.load('balanced', 20);
                  const loaded = await window.__LLC_TEST_API.loadData(data, 'real export');
                  const verified = provenance ? window.__LLC_TEST_API.loadProvenanceData(provenance) : null;
                  window.__LLC_TEST_API.stepFixed(80);
                  const afterLoad = window.__LLC_TEST_API.metrics();
                  let mismatched = null;
                  if (provenance) {
                    try {
                      window.__LLC_TEST_API.loadProvenanceData({...provenance, output_sha256: '0'.repeat(64)});
                      mismatched = {rejected: false};
                    } catch (error) {
                      mismatched = {rejected: true, message: error.message};
                    }
                  }
                  const invalid = window.__LLC_TEST_API.loadInvalidData({metadata: {schema_version: 'bad', semantics_mode: 'partial_agents'}, lineages: [], branches: [], agents: []});
                  return {
                    before,
                    loaded,
                    verified,
                    afterLoad,
                    mismatched,
                    invalid,
                    livingLabel: livingMetricLabel.textContent,
                    shareLabel: shareMetricLabel.textContent,
                    warningHidden: semanticPanel.hidden,
                    warningText: semanticNotice.textContent,
                    subtitle: datasetSubtitle.textContent,
                    returnHidden: returnGalleryButton.hidden,
                    datasetDisabled: datasetSelect.disabled
                  };
                }
                """,
                {"data": external_data, "provenance": external_provenance},
            )
            external_result["loadedByFile"] = loaded_by_file
            external_result["verifiedByFile"] = verified_by_file
            external_result["withoutProvenance"] = without_provenance
            external.close()

        relay_result = None
        if args.relay_manifest:
            relay_page = browser.new_page(viewport={"width": 1440, "height": 980}, device_scale_factor=1)
            relay_page.goto(f"{args.base_url}?dataset=balanced&tick=0", wait_until="networkidle")
            relay_page.wait_for_timeout(600)
            loaded = relay_page.evaluate("url => window.__LLC_TEST_API.connectRelay(url)", args.relay_manifest)
            relay_page.wait_for_timeout(600)
            status = relay_page.evaluate("window.__LLC_TEST_API.relayStatus()")
            relay_result = {
                "loaded": loaded,
                "status": status,
                "subtitle": relay_page.evaluate("datasetSubtitle.textContent"),
                "warningText": relay_page.evaluate("semanticNotice.textContent"),
            }
            relay_page.close()

        browser.close()

    checks = {
        "deterministic_fixed_step_repeatability": deterministic,
        "finite_snapshot_values": finite and _finite_snapshot(stress_snapshot),
        "partial_agent_labels": partial["livingLabel"] == "Recorded particles"
        and partial["shareLabel"] == "Recorded share"
        and not partial["warningHidden"]
        and partial["dyingDisabled"]
        and "not current living population" in partial["warningText"],
        "free_space_reclamation": late_body is not None
        and collapsed["radius"] > 70
        and late_body["count"] == 0
        and late_body["radius"] <= 10
        and late_distance < peak_distance,
        "stress_particle_count": stress_metrics["living"] >= 5000,
    }
    if external_result is not None:
        checks["external_file_load"] = (
            external_result["afterLoad"]["external_loaded"]
            and external_result["loadedByFile"]["external_loaded"]
            and external_result["verifiedByFile"]["metrics"]["provenance_verified"]
            and external_result["afterLoad"]["provenance_verified"]
            and external_result["afterLoad"]["source_classification"] in {"EXP-A", "EXP-B", "EXP-C"}
            and external_result["livingLabel"] in {"Recorded particles", "Projected active particles"}
            and external_result["shareLabel"] in {"Recorded share", "Projected active share"}
            and not external_result["warningHidden"]
            and "checkpoint-boundary" in external_result["warningText"]
            and "Initial population: complete selected-cell" in external_result["warningText"]
            and "checkpoint identities" in external_result["warningText"]
            and "Provenance verified: yes" in external_result["warningText"]
            and external_result["datasetDisabled"]
            and not external_result["returnHidden"]
        )
        checks["mismatched_external_provenance_rejected"] = (
            external_result["mismatched"]["rejected"]
            and external_result["afterLoad"]["provenance_verified"]
        )
        checks["playback_without_provenance_labeled_unverified"] = (
            external_result["withoutProvenance"]["metrics"]["external_loaded"]
            and not external_result["withoutProvenance"]["metrics"]["provenance_verified"]
            and "Provenance not loaded or not verified" in external_result["withoutProvenance"]["warningText"]
        )
        checks["invalid_external_rejected_without_state_replacement"] = (
            external_result["invalid"]["rejected"]
            and external_result["invalid"]["after"]["dataset_id"] == external_result["afterLoad"]["dataset_id"]
            and external_result["invalid"]["after"]["external_loaded"]
        )
    if relay_result is not None:
        checks["relay_manifest_load"] = (
            relay_result["loaded"]["relay_active"]
            and relay_result["loaded"]["relay_hash_verified"]
            and relay_result["loaded"]["relay_sequence"] >= 1
            and relay_result["loaded"]["app_mode"] == "relay"
            and relay_result["status"]["sourceState"] in {"source_interrupted", "source_complete", "watching"}
            and "Provenance verified: yes" in relay_result["warningText"]
        )
    result = {
        "ok": all(checks.values()),
        "checks": checks,
        "details": {
            "deterministic_snapshot_size": {
                "particles": len(first["particles"]),
                "lineage_bodies": len(first["lineageBodies"]),
                "branch_bodies": len(first["branchBodies"]),
            },
            "partial_agent_metrics": partial["metrics"],
            "free_space_reclamation": {
                "collapsed_lineage": "lineage_2",
                "peak_radius": collapsed["radius"],
                "late_count": late_body["count"] if late_body else None,
                "late_radius": late_body["radius"] if late_body else None,
                "peak_neighbor_distance": peak_distance,
                "late_neighbor_distance": late_distance,
            },
            "stress_metrics": stress_metrics,
            "external_result": external_result,
            "relay_result": relay_result,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.out)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
