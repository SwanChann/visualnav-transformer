# External execution pack

This pack is the boundary between locally completed preparation and experiments that
require storage, GPU training, simulation, or robot hardware. Execute stages in
order and return raw artifacts plus receipts; do not return screenshots alone.

1. `01_data_pilot.md`
2. `02_training_run_order.md`
3. `03_sim_robot_promotion.md`

All results must validate against `../benchmark/result_schema_v0.1.json`. A failed
gate is a research result and triggers the downgrade rules in
`../paper/go_no_go_v0.1.md`; it is not repaired by changing the claim after seeing
the test set.
