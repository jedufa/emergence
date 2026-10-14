from __future__ import annotations

import argparse
import json
from pathlib import Path


SCENARIOS = [
    ("balanced_coexistence", "balanced", [25, 120, 190], (1440, 980)),
    ("dominant_sweep", "dominant", [40, 145, 250], (1440, 980)),
    ("weak_lineage_extinction", "weak_extinction", [30, 140, 220], (1440, 980)),
    ("late_bloomer", "late_bloomer", [80, 175, 265], (1440, 980)),
    ("boom_collapse", "boom_collapse", [35, 150, 245], (1440, 980)),
    ("branch_split", "branch_split", [40, 130, 225], (1440, 980)),
    ("sibling_branch_competition", "sibling_competition", [40, 130, 225], (1440, 980)),
    ("dominance_reversal", "dominance_reversal", [70, 165, 270], (1440, 980)),
    ("cyclic_lineage_fortunes", "cyclic_fortunes", [60, 180, 300], (1440, 980)),
    ("founder_bottleneck", "founder_bottleneck", [25, 115, 265], (1440, 980)),
    ("p12b_partial_real_data", "p12b", [837, 1200, 1814], (1440, 980)),
    ("stress_5000", "stress", [190], (1440, 980)),
    ("selected_lineage", "branch_split", [220], (1440, 980), "lineage=lineage_1"),
    ("isolated_lineage", "branch_split", [220], (1440, 980), "lineage=lineage_1&isolated=1"),
    ("narrow_layout", "balanced", [170], (390, 860)),
    ("wide_layout", "dominant", [240], (1920, 1080)),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture living lineage canopy browser smoke evidence")
    parser.add_argument("--base-url", default="http://127.0.0.1:8769/index.html")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit("Playwright is required for browser smoke capture: python -m pip install playwright && python -m playwright install chromium") from exc

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        for scenario in SCENARIOS:
            name, dataset, ticks, viewport = scenario[:4]
            extra = scenario[4] if len(scenario) > 4 else ""
            page = browser.new_page(viewport={"width": viewport[0], "height": viewport[1]}, device_scale_factor=1)
            scenario_results = []
            for tick_index, tick in enumerate(ticks):
                label = ("early", "middle", "late")[tick_index] if len(ticks) == 3 else "state"
                query = f"dataset={dataset}&tick={tick}" + (f"&{extra}" if extra else "")
                page.goto(f"{args.base_url}?{query}", wait_until="networkidle")
                page.wait_for_timeout(1800)
                metrics = page.evaluate("window.__LLC_METRICS")
                pixel = page.evaluate(
                    """
                    () => {
                      const canvas = document.getElementById('canopyCanvas');
                      const ctx = canvas.getContext('2d');
                      const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
                      let lit = 0;
                      for (let i = 0; i < data.length; i += 64) {
                        if (data[i] + data[i + 1] + data[i + 2] > 45) lit += 1;
                      }
                      return {sampled_lit_pixels: lit, canvas_width: canvas.width, canvas_height: canvas.height};
                    }
                    """
                )
                screenshot = args.out_dir / f"{len(results) + 1:02d}_{name}_{label}.png"
                page.screenshot(path=screenshot, full_page=True)
                scenario_results.append({"label": label, "tick": tick, "query": query, "screenshot": str(screenshot), "metrics": metrics, "pixel_check": pixel})
            results.append({"name": name, "dataset": dataset, "viewport": viewport, "states": scenario_results})
            page.close()

        video_dir = args.out_dir / "motion"
        video_dir.mkdir(exist_ok=True)
        context = browser.new_context(viewport={"width": 1440, "height": 980}, record_video_dir=video_dir)
        motion = context.new_page()
        motion.goto(f"{args.base_url}?dataset=dominant&tick=40", wait_until="networkidle")
        motion.wait_for_timeout(500)
        motion.click("#playButton")
        motion.wait_for_timeout(3600)
        motion.evaluate("async () => { await window.__LLC_TEST_API.load('sibling_competition', 40); }")
        motion.click("#playButton")
        motion.wait_for_timeout(3200)
        motion.evaluate("async () => { await window.__LLC_TEST_API.load('boom_collapse', 35); }")
        motion.click("#playButton")
        motion.wait_for_timeout(3600)
        motion.evaluate(
            """
            () => {
              const range = document.getElementById('timelineRange');
              range.value = 245;
              range.dispatchEvent(new Event('input', {bubbles: true}));
            }
            """
        )
        motion.wait_for_timeout(1200)
        video = motion.video
        motion.close()
        video_path = video.path()
        context.close()
        browser.close()
    summary = args.out_dir / "browser_smoke_summary.json"
    summary.write_text(json.dumps({"scenarios": results, "motion_artifact": video_path}, indent=2, sort_keys=True) + "\n")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
