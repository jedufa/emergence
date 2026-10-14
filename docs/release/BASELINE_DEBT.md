# Historical baseline limitations

The preserved `baseline-debt.json` and architecture policy record a historical
complete-tree comparison. They are not regenerated to make a curated public
snapshot pass. The live historical ratchet remains a failing diagnostic; the
current finding counts and causes are reported in [VALIDATION.md](VALIDATION.md).

The representative source-release gate explicitly excludes its old full-tree
sentinel and the environment-dependent frozen BundleV2 hash sentinel. Their test
assertions remain intact. Three native ACS expected failures retain their exact
exception boundary. Additional phase-C failures reproduced on the unmodified
baseline are also documented. None of these diagnostics grants Qualification.

Changing runtime location defaults, package membership, or a provenance-bearing
contract hash requires new source identity and targeted checks. It does not
preserve an old qualification merely because the numerical algorithms look
similar. Unavailable historical authority remains unavailable.
