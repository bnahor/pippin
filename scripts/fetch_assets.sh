#!/usr/bin/env bash
# Fetch third-party robot models (not vendored in the repo).
#   scripts/fetch_assets.sh            # Franka Panda from MuJoCo Menagerie (Apache-2.0)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party
if [ ! -d third_party/mujoco_menagerie ]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
fi
git -C third_party/mujoco_menagerie sparse-checkout set franka_emika_panda
echo "fetched: third_party/mujoco_menagerie/franka_emika_panda"
