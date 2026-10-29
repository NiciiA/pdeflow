#!/usr/bin/env bash
set -eu

# Run inside the desired project root. Creates directories only; never overwrites files.
mkdir -p \
  docs references prompts scripts notebooks \
  configs/{data,models,training,inference,experiments,runtime} \
  src/pdeflow/{core,equations,numerics,data,observations,conditioning,models,generative,sampling,steering,refinement,evaluation,runtime} \
  tests/{unit,numerics,integration}

printf '%s\n' 'PDEFlow directory structure is ready. No existing files were overwritten.'
