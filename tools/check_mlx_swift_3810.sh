#!/bin/zsh
# Does an mlx-swift tag vendor the NAX split-K GEMM fix (mlx PR #3810, commit a8c3e9c)?
#
# The bug: mlx-swift ≤0.31.6 JIT-compiles `steel_gemm_splitk_axpby_nax` with the
# wrong dtype template parameter (ml-explore/mlx#3797) — bf16 inputs read as fp32 →
# garbage/NaN in the dispatch window (half precision, M·N ≥ 2048², K ≥ 10240,
# K ≥ 3·max(M,N)). Worked around by row-chunked down-projections in
# boogu-image-swift (LuminaFeedForward.downProjected), mage-flow-swift
# (MageFeedForward.downProjected), and qwen3vl-mlx-swift (MLP.downProjected).
#
# Usage: tools/check_mlx_swift_3810.sh [tag]   (default: latest release tag)
#
# On FIXED: bump the mlx-swift pin, run `swift run BooguGate --nax-probe` in a real
# Metal context (strict: cos > 0.999 AND max_abs < 100 — cos ≥0.99 is NOT enough
# near the boundary), and on PASS remove all three row-chunks. A locally patched
# checkout for A/B validation lives at mlxengine-image/WIP/mlx-swift-3810
# (`swift package edit mlx-swift --path ...` / `unedit`).
set -euo pipefail

FIX_COMMIT=a8c3e9c66821f1ee7f377afe4999885abe9a2d05  # mlx PR #3810 merge
TAG=${1:-$(gh api repos/ml-explore/mlx-swift/releases/latest --jq .tag_name)}

echo "mlx-swift tag: $TAG"
SUBMODULE_SHA=$(gh api "repos/ml-explore/mlx-swift/contents/Source/Cmlx/mlx?ref=$TAG" --jq .sha)
echo "vendored mlx submodule: $SUBMODULE_SHA"

# Is the fix an ancestor of the vendored commit? (compare in ml-explore/mlx)
STATUS=$(gh api "repos/ml-explore/mlx/compare/$FIX_COMMIT...$SUBMODULE_SHA" --jq .status 2>/dev/null || echo unknown)
case "$STATUS" in
  ahead|identical)
    echo "FIXED — $TAG vendors mlx ≥ $FIX_COMMIT ($STATUS)."
    echo "Next: bump the pin, run \`swift run BooguGate --nax-probe\`; on PASS remove the row-chunks"
    echo "in boogu-image-swift, mage-flow-swift, and qwen3vl-mlx-swift." ;;
  behind|diverged)
    echo "NOT FIXED — vendored mlx is $STATUS relative to the fix. Keep the row-chunks." ;;
  *)
    echo "UNKNOWN — could not compare ($STATUS). Check manually:"
    echo "  gh api repos/ml-explore/mlx/compare/$FIX_COMMIT...$SUBMODULE_SHA --jq .status"
    exit 2 ;;
esac
