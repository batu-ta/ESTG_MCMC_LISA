#!/usr/bin/env bash
# One-time setup inside WSL2 (Ubuntu). Run from the repo's analysis/ folder:
#     bash setup_wsl.sh
# Installs: build tools + MPI, Miniforge, a Python 3.11 env "estg" with Cobaya,
# hi_class (built from source) and the native Planck 2018 likelihood data.
# Everything heavy (Miniforge, hi_class, Planck data, chains) goes under
# $ESTG_WORK (default ~/estg), i.e. inside the Ubuntu virtual disk. When Ubuntu
# is installed with --location D:\WSL\Ubuntu, that disk lives on D:.
set -euo pipefail

# Refuse to run if Ubuntu's virtual disk is not on D: (user requirement).
if [ -n "${WSL_DISTRO_NAME:-}" ] && command -v powershell.exe >/dev/null 2>&1; then
  PS_CMD="Get-ChildItem 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss' | ForEach-Object { Get-ItemProperty \$_.PSPath } | Where-Object { \$_.DistributionName -eq '${WSL_DISTRO_NAME}' } | ForEach-Object { \$_.BasePath }"
  BASEPATH="$(powershell.exe -NoProfile -Command "$PS_CMD" 2>/dev/null | tr -d '\r' | head -n1)"
  echo ">> Ubuntu virtual disk folder: ${BASEPATH:-unknown}"
  # Accept "D:\..." or "\\?\D:\..." (strip leading backslashes and '?')
  DRIVE="$(printf '%s' "$BASEPATH" | sed 's/^[\\?]*//' | cut -c1-2 | tr 'd' 'D')"
  if [ "$DRIVE" != "D:" ] && [ "${ALLOW_NON_D:-0}" != "1" ]; then
    echo "!! Ubuntu is not installed on D:. See analysis/README.md step 0, or set ALLOW_NON_D=1."
    exit 1
  fi
fi

ESTG_WORK="${ESTG_WORK:-$HOME/estg}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$ESTG_WORK"

echo ">> system packages"
sudo apt-get update
sudo apt-get install -y build-essential gfortran git wget libopenmpi-dev openmpi-bin

echo ">> Miniforge"
if [ ! -d "$ESTG_WORK/miniforge" ]; then
  wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh
  bash /tmp/miniforge.sh -b -p "$ESTG_WORK/miniforge"
fi
# shellcheck disable=SC1091
source "$ESTG_WORK/miniforge/etc/profile.d/conda.sh"
conda env list | grep -q '^estg ' || conda create -y -n estg python=3.11
conda activate estg

echo ">> Python packages"
pip install "numpy>=2" scipy matplotlib "cython>=3" pyyaml getdist mpi4py cobaya

echo ">> hi_class"
if [ ! -d "$ESTG_WORK/hi_class" ]; then
  git clone --depth 1 https://github.com/miguelzuma/hi_class_public.git "$ESTG_WORK/hi_class"
fi
# hi_class (CLASS 3.4 based) installs as the Python package 'hiclassy';
# pip builds the C core and the wrapper. estg_theory.py imports it directly.
( cd "$ESTG_WORK/hi_class" && pip install . )
python -c "import hiclassy; print('hiclassy', hiclassy.__version__)"
( cd "$ESTG_WORK/hi_class" && git rev-parse HEAD > "$HERE/HICLASS_COMMIT.txt" )

echo ">> Planck 2018 likelihood data (native Python versions)"
export COBAYA_PACKAGES_PATH="$ESTG_WORK/packages"
cobaya-install planck_2018_lowl.TT planck_2018_lowl.EE \
  planck_2018_highl_plik.TTTEEE_lite_native planck_2018_lensing.native \
  -p "$COBAYA_PACKAGES_PATH"

echo ">> Cobaya configs"
( cd "$HERE/cosmo" && python make_configs.py --chains-dir "$ESTG_WORK/chains" )

cat <<EOF

Setup done. In every new shell:
    source "$ESTG_WORK/miniforge/etc/profile.d/conda.sh" && conda activate estg
    export COBAYA_PACKAGES_PATH="$ESTG_WORK/packages"
Then follow analysis/README.md, step 3.
EOF
