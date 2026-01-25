#!/bin/bash

srun \
	--mem=32g \
	--nodes=1 \
	--ntasks-per-node=1 \
	--cpus-per-task=8 \
	--partition=gpuA100x4 \
	--gpus-per-node=1 \
	--account=bgak-delta-gpu \
	--time=04:00:00 \
	--constraint="scratch" \
	--job-name=interact \
	--pty /bin/bash

