#!/bin/bash

srun \
	--mem=16g \
	--nodes=1 \
	--ntasks-per-node=1 \
	--cpus-per-task=4 \
	--partition=gpuA100x4 \
	--gpus-per-node=1 \
	--account=bfpt-delta-gpu \
	--time=04:00:00 \
	--constraint="scratch" \
	--job-name=interact \
	--pty /bin/bash

