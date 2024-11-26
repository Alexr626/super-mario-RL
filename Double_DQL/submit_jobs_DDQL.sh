#!/bin/bash

# Master script to submit 5*JOBS_PER_VARIATION SLURM jobs with varying resources

# Base parameters
ACCOUNT="liu32_1378"
PROJECT_DIR="$HOME/super-mario-RL/Double_DQL"
OUTPUT_DIR="$HOME/output/DDQL"
JOB_SCRIPT="$PROJECT_DIR/run_mario_DDQL.job"

# Number of jobs per variation
JOBS_PER_VARIATION=2

JOB_ID=1

# Ensure the output directory exists
mkdir -p "${OUTPUT_DIR}"

# Function to submit jobs
submit_job() {
    local type="DDQL"
    local job_name=$1
    local mem=$2
    local gpus=$3
    local time_limit=$4
    local index=$5
    local job_id=$6
    local output_file="${OUTPUT_DIR}/${type}_output_${job_name}_${job_id}.log"

    sbatch \
        --job-name="${job_name}_${index}" \
        --mem=${mem} \
        --gres=gpu:${gpus} \
        --time=${time_limit} \
        --account=${ACCOUNT} \
        --output="${output_file}" \
        --export=JOB_ID=${job_id} \
        ${JOB_SCRIPT}
}

# 1. Submit 10 jobs with 16G memory instead of 32G
for i in $(seq 1 ${JOBS_PER_VARIATION}); do
    JOB_NAME="mario_rl_mem16"
    MEM="16G"
    GPUS="1"
    TIME_LIMIT="48:00:00"
    submit_job "${JOB_NAME}" "${MEM}" "${GPUS}" "${TIME_LIMIT}" "${i}" "${JOB_ID}"
    JOB_ID=$((JOB_ID + 1))
done

# 2. Submit 10 jobs requesting 2 GPUs instead of 1
for i in $(seq 1 ${JOBS_PER_VARIATION}); do
    JOB_NAME="mario_rl_gpu2"
    MEM="32G"
    GPUS="2"
    TIME_LIMIT="24:00:00"
    submit_job "${JOB_NAME}" "${MEM}" "${GPUS}" "${TIME_LIMIT}" "${i}" "${JOB_ID}"
    JOB_ID=$((JOB_ID + 1))
done

# 3. Submit 10 jobs with a 3-hour time limit instead of 6
for i in $(seq 1 ${JOBS_PER_VARIATION}); do
    JOB_NAME="mario_rl_time3h"
    MEM="32G"
    GPUS="1"
    TIME_LIMIT="24:00:00"
    submit_job "${JOB_NAME}" "${MEM}" "${GPUS}" "${TIME_LIMIT}" "${i}" "${JOB_ID}"
    JOB_ID=$((JOB_ID + 1))
done

# 4. Submit 10 jobs with a 12-hour time limit instead of 6
for i in $(seq 1 ${JOBS_PER_VARIATION}); do
    JOB_NAME="mario_rl_time12h"
    MEM="32G"
    GPUS="1"
    TIME_LIMIT="24:00:00"
    submit_job "${JOB_NAME}" "${MEM}" "${GPUS}" "${TIME_LIMIT}" "${i}" "${JOB_ID}"
    JOB_ID=$((JOB_ID + 1))
done

# 5. Submit 10 standard jobs (32G, 1 GPU, 6-hour time limit)
for i in $(seq 1 ${JOBS_PER_VARIATION}); do
    JOB_NAME="mario_rl_standard"
    MEM="32G"
    GPUS="1"
    TIME_LIMIT="48:00:00"
    submit_job "${JOB_NAME}" "${MEM}" "${GPUS}" "${TIME_LIMIT}" "${i}" "${JOB_ID}"
    JOB_ID=$((JOB_ID + 1))
done

echo "All 50 jobs have been submitted."