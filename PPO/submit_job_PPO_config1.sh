#!/bin/bash

# Test submission for directing output to the 'output' directory

# Base parameters
ACCOUNT="liu32_1378"
PROJECT_DIR="$HOME/super-mario-RL/PPO"
OUTPUT_DIR="$HOME/output/PPO"
JOB_SCRIPT="$PROJECT_DIR/run_mario_PPO.job"

TEST_JOBS=1
CONFIG_VERSION=1
JOB_ID=1
TIMESTAMP=0

# Ensure the output directory exists
mkdir -p "${OUTPUT_DIR}"

# Function to submit a test job
submit_test_job() {
    local job_name=$1
    local mem=$2
    local gpus=$3
    local time_limit=$4
    local index=$5
    local job_id=$6
    local config_version=$7
    local output_file=$8
    local timestamp=$9
    local type="PPO"

    sbatch \
        --job-name="${job_name}_${index}_${type}" \
        --mem=${mem} \
        --gres=gpu:${gpus} \
        --time=${time_limit} \
        --account=${ACCOUNT} \
        --output="${output_file}" \
        --export=JOB_ID=${job_id},CONFIG_VERSION=${config_version},TIMESTAMP=${timestamp} \
        ${JOB_SCRIPT}
}

# Submit a single test job
JOB_NAME="mario_rl_PPO_conf1"
MEM="32G"
GPUS="2"
TIME_LIMIT="48:00:00"
INDEX=1
OUTPUT_FILE="${OUTPUT_DIR}/output_test_config_${CONFIG_VERSION}.log"
submit_test_job "${JOB_NAME}" "${MEM}" "${GPUS}" "${TIME_LIMIT}" "${INDEX}" "${JOB_ID}" "${CONFIG_VERSION}" "${OUTPUT_FILE}" "${TIMESTAMP}"

echo "Test job has been submitted."