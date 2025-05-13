MODELS=(
    "r2e-edits/qwen32B_swesmith_orig-32b_32k-bz8_epoch1_lr1en5-v1"
    "r2e-edits/qwen25coder-14b_swe-smith-trajectories-R2E-32k_bz16_epoch3_lr1en5-v2"
)

EXPERIMENT_NAMES=(
    "qwen32B_swesmith_orig-32b_32k-bz8_epoch1_lr1en5-v1"
    "qwen25coder-14b_swe-smith-trajectories-R2E-32k_bz16_epoch3_lr1en5-v2"
)

for i in "${!MODELS[@]}"; do
    bash run_model.sh "${MODELS[$i]}" "${EXPERIMENT_NAMES[$i]}"
done
