MODEL=$1
EXPERIMENT_NAME=$2

echo "Starting sglang server for $MODEL ..."

CONTAINER_ID=$(
sudo docker run -d --gpus all --shm-size 32g \
    -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v /home/ubuntu/LLaMA-Factory/saves:/models \
    --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server \
    --model-path "${MODEL}" \
    --host 0.0.0.0 \
    --port 8000 \
    --tp-size 8
)

echo "Container ID: $CONTAINER_ID"

echo "SGLang server for $MODEL started successfully."

echo "Waiting for the sglang server to fully initialize..."
sleep 300

echo "Running evaluation for $MODEL ..."
sudo bash src/docker_bash_utils/remove_containers.sh

bash src/docker_bash_utils/remove_containers.sh

uv run python src/r2egym/agenthub/run/edit.py \
    runagent_multiple \
    --traj_dir "./traj" \
    --max_workers 54 \
    --start_idx 0 \
    --k 500 \
    --dataset "R2E-Gym/SWE-Bench-Verified" \
    --split "test" \
    --llm_name "openai/${MODEL}" \
    --use_fn_calling False \
    --exp_name $2 \
    --temperature 0 \
    --max_steps 40 \
    --backend "docker" \
    --swesmith_wrapper False

echo "Shutting down sglang server (container $CONTAINER_ID) ..."
sudo docker stop "$CONTAINER_ID"
sudo docker rm "$CONTAINER_ID"
echo "Container $CONTAINER_ID has been stopped and removed."

echo "Sleeping to ensure the port is released..."
sleep 90
