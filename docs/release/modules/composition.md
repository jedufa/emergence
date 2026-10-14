# EIEL Composition

`eiel_composition` integrates existing contracts, Lab declarations, source-bound native artifacts, execution and Evidence.

Example: materialize `examples/release/paired_cpu_reproduction.py`, then use the canonical `eiel execute` entry point as described in the usage guide. `paired_evolution.json` only demonstrates declaration; its growth-base control remains unsupported.

Composition coordinates one production engine. It never creates a fallback simulator or second RNG/state owner. Source, configuration, device and artifact identities remain admission-bound.

Dependencies follow [the source map](../source-to-module.json) and the existing ten-module DAG. Shared legacy imports remain under the unchanged debt ratchet.

Important checks: Science Request compiler/front-door regressions, exact installed usage, modular DAG and tick authority.

Status: Supported orchestration interfaces; machine provisioning and historical artifact availability are explicit prerequisites.
