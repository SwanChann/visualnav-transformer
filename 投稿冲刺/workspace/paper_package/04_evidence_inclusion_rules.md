# Evidence inclusion rules

| Evidence | Allowed use | Prohibited inference |
|---|---|---|
| Offline unified CSV | source-scoped latency/proxy characterization | success, safety, cross-source global ranking |
| TTS proxy rows | descriptive candidate-selection behavior | independent improvement due to verifier overlap |
| Historical MuJoCo JSON | integration coverage and data-quality audit | controlled method/encoder superiority |
| Controlled MuJoCo v0.3 | deployment-scale-corrected descriptive policy-only and stabilizer-stratified simulation comparison | statistical superiority, real-robot transfer, pooled stabilizer claim; v0.2 is audit-only |
| U10 v5 independent offline metrics | held-out physical-unit ADE/FDE and sampler cost, with GO Stanford scale 0.12 m bound to the training data config | closed-loop success or cross-dataset generalization; U10 v2 is invalidated |
| Sampled timing CSV | sampled loop distribution for named log/config | continuous trace, deadline guarantee, trial count |
| Video/contact sheets | execution and visible motion | success without goal/termination protocol |
| Folder names | retrieval prior for manual review | ground-truth label |

Every quantitative statement must cite a row in `../claim_evidence_matrix.csv` or a
generated artifact. Planning documents and proposed experiments are never evidence.
Unknown fields remain unknown; no value is imputed from filenames unless explicitly
reported as filename metadata.

In the system-stabilizer stratum, all five methods produce byte-identical stored
trajectories for every scene/seed; only compute latency differs. These rows are one
system-controller behavior replicated across method calls, not five independent
navigation outcomes.
