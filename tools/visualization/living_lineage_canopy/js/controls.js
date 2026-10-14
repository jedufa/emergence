export function bindControls(elements, handlers) {
  elements.playButton.addEventListener("click", handlers.onPlayToggle);
  elements.resetButton.addEventListener("click", handlers.onReset);
  elements.endButton.addEventListener("click", handlers.onEnd);
  elements.scenariosModeButton.addEventListener("click", handlers.onScenariosMode);
  elements.savedModeButton.addEventListener("click", handlers.onSavedMode);
  elements.relayModeButton.addEventListener("click", handlers.onRelayMode);
  elements.relayConnectButton.addEventListener("click", handlers.onRelayConnect);
  elements.relayDisconnectButton.addEventListener("click", handlers.onRelayDisconnect);
  elements.followLiveToggle.addEventListener("change", () => handlers.onFollowLive(elements.followLiveToggle.checked));
  elements.showcaseButton.addEventListener("click", handlers.onShowcaseToggle);
  elements.skipScenarioButton.addEventListener("click", handlers.onShowcaseSkip);
  elements.datasetSelect.addEventListener("change", () => handlers.onDataset(elements.datasetSelect.value));
  elements.playbackFileInput.addEventListener("change", () => handlers.onPlaybackFile(elements.playbackFileInput.files?.[0] || null));
  elements.provenanceFileInput.addEventListener("change", () => handlers.onProvenanceFile(elements.provenanceFileInput.files?.[0] || null));
  elements.playbackSearch.addEventListener("input", () => handlers.onPlaybackSearch(elements.playbackSearch.value));
  elements.playbackLibrarySelect.addEventListener("change", () => handlers.onPlaybackLibrarySelect(elements.playbackLibrarySelect.value));
  elements.playbackLibraryPlay.addEventListener("click", handlers.onPlaybackLibraryPlay);
  elements.playbackLibraryRefresh.addEventListener("click", handlers.onPlaybackLibraryRefresh);
  elements.returnGalleryButton.addEventListener("click", handlers.onReturnGallery);
  elements.timelineRange.addEventListener("input", () => handlers.onScrub(Number(elements.timelineRange.value)));
  elements.speedRange.addEventListener("input", () => handlers.onSpeed(Number(elements.speedRange.value)));
  elements.particleSizeRange.addEventListener("input", () => handlers.onViewChange());
  elements.motionRange.addEventListener("input", () => handlers.onViewChange());
  elements.branchPullRange.addEventListener("input", () => handlers.onViewChange());
  elements.lineagePullRange.addEventListener("input", () => handlers.onViewChange());
  elements.birthPulseToggle.addEventListener("change", () => handlers.onViewChange());
  elements.dyingToggle.addEventListener("change", () => handlers.onViewChange());
  elements.lineageSelect.addEventListener("change", () => handlers.onLineage(elements.lineageSelect.value));
  elements.isolateButton.addEventListener("click", handlers.onIsolate);
  elements.showAllButton.addEventListener("click", handlers.onShowAll);
}

export function collectElements() {
  const ids = [
    "datasetSelect", "datasetSubtitle", "qualityBadge", "playButton", "resetButton", "endButton",
    "scenariosModeButton", "savedModeButton", "relayModeButton", "relayControls", "relayManifestInput",
    "relayConnectButton", "relayDisconnectButton", "followLiveToggle", "relayStatusPanel", "relayStateValue",
    "relayRunValue", "relayBoundaryValue", "relayLagValue", "relaySequenceValue", "relayVerifiedValue",
    "relayWarningValue", "playbackFileInput", "provenanceFileInput", "returnGalleryButton",
    "playbackSearch", "playbackLibrarySelect", "playbackLibraryPlay", "playbackLibraryRefresh", "playbackLibraryStatus",
    "showcaseButton", "skipScenarioButton", "loopShowcaseToggle", "timelineRange", "tickValue",
    "generationValue", "speedRange", "speedValue", "particleSizeRange", "motionRange",
    "branchPullRange", "lineagePullRange", "birthPulseToggle", "dyingToggle", "lineageSelect",
    "lineageLegend", "isolateButton", "showAllButton", "selectionDetails", "livingMetric", "lineageMetric",
    "branchMetric", "shareMetric", "livingMetricLabel", "shareMetricLabel", "fpsMetric",
    "physicsMetric", "renderMetric", "hoverHint", "semanticPanel", "semanticTitle", "semanticNotice",
    "scenarioGallery", "showcaseOverlay",
  ];
  return Object.fromEntries(ids.map((id) => [id, document.getElementById(id)]));
}
