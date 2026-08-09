# Phase 3A-HW data contract

Do not place synthetic data here. Synthetic CI datasets are generated under `results/` and are software tests only.

The empirical eight-device pilot, when acquired, must be committed or supplied as `data/phase3a_hw/trials.csv` with one row per completed trial and these required columns:

`device_id, trial_id, trial_class, block_id, run_index, order, previous_trial_order, reset_elapsed_s, pre_voltage_v, pre_current_a, pre_current_fraction_full_scale, pre_voltage_slope_v_per_s, pre_temperature_delta_c, probe_work_j`

`trial_class` is either `main` or `long_washout`. Device/trial/order/block/run/previous-order fields must exactly match the frozen schedule generated from `config/phase3a_hw_schedule.json`.

Before collecting the first empirical main trial, `config/phase3a_hw_hardware_manifest.yaml` must be completed and changed to `FROZEN`. Do not alter the analysis, thresholds, schedule, primary outcome, or exclusion logic after inspecting pilot outcomes. A required change after unblinding creates a new protocol version and requires new data.
