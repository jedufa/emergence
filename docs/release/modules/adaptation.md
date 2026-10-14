# Adaptation

Existing `engine.evolution` and native inheritance code own reproduction, mutation and selection.

Example: Declare mutation/recombination controls in a Science Request; see the paired release example.

Genome state remains with the existing authoritative engine. Changes to selection or draw schedules require explicit semantic impact review; contributor extensions cannot silently alter them.

Dependencies follow [the source map](../source-to-module.json) and the existing ten-module DAG. Shared legacy imports remain under the unchanged debt ratchet.

Important checks: Existing reproduction/genome tests plus source-bound real birth and inheritance evidence.

Status: Retained shared implementation; a config construction test is not evolutionary evidence.
