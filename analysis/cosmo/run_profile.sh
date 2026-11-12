#!/usr/bin/env bash
# Run the beta profile likelihood at low priority on a limited number of cores.
#   bash run_profile.sh                 # all grid points, 2 cores
#   CORES=1 bash run_profile.sh         # gentler
#   bash run_profile.sh --only 0.0      # one point (extra args go to profile_beta.py)
# Stop any time with Ctrl+C (or `pkill -f profile_beta.py`); re-running skips
# finished points. Output and log: ~/estg/chains/profile/
set -euo pipefail
CORES="${CORES:-2}"
source "$HOME/estg/miniforge/etc/profile.d/conda.sh" && conda activate estg
export COBAYA_PACKAGES_PATH="$HOME/estg/packages"
export OMP_NUM_THREADS="$CORES" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT="$HOME/estg/chains/profile"
mkdir -p "$OUT"
cd "$(dirname "${BASH_SOURCE[0]}")"
exec nice -n 19 python -u profile_beta.py --out "$OUT" "$@" 2>&1 | tee -a "$OUT/profile.log"
