#!/usr/bin/env bash
# Ubuntu 22.04 system toolchain; deliberately isolated from Pixi shared libraries.
set -euo pipefail
adapter_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(realpath "$adapter_dir/../../OKVIS1")
revision=33ac70b4d95df98cd662086dc897b90a6f35d5a9
if [[ ! -d "$source_dir/.git" ]]; then
    git clone --no-checkout https://github.com/ethz-asl/okvis.git "$source_dir"
    git -C "$source_dir" checkout --detach "$revision"
fi
if [[ $(git -C "$source_dir" rev-parse HEAD) != "$revision" ]]; then
    echo "Unexpected OKVIS 1 revision; checkout preserved: $source_dir" >&2
    exit 1
fi
if git -C "$source_dir" apply --check "$adapter_dir/modernise.patch" 2>/dev/null; then
    git -C "$source_dir" apply "$adapter_dir/modernise.patch"
else
    git -C "$source_dir" apply --reverse --check "$adapter_dir/modernise.patch"
fi
export PATH=/usr/bin:/bin
unset CONDA_PREFIX CMAKE_PREFIX_PATH CMAKE_ARGS LD_LIBRARY_PATH LD_PRELOAD PYTHONPATH
unset CC CXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS
cmake -S "$source_dir" -B "$source_dir/build" \
    -DUSE_SYSTEM_CERES=ON -DN_CORES=1 -DDO_TIMING=OFF \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE='-O2 -DNDEBUG'
cmake --build "$source_dir/build" --target vslamlab_okvis1_stereo_vi -j1
