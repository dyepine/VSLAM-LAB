#!/usr/bin/env bash
set -euo pipefail
adapter_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(realpath "$adapter_dir/../../SqrtVINS")
revision=30fafc89d89cfb896d06fdaa68fbda8545e02e9f
if [[ ! -d "$source_dir/.git" ]]; then
    git clone --filter=blob:none --no-checkout https://github.com/rpng/sqrtVINS.git "$source_dir"
    git -C "$source_dir" sparse-checkout init --cone
    git -C "$source_dir" sparse-checkout set ov_core ov_srvins config
    git -C "$source_dir" checkout --detach "$revision"
fi
[[ $(git -C "$source_dir" rev-parse HEAD) == "$revision" ]] || { echo 'Unexpected sqrtVINS revision; checkout preserved.' >&2; exit 1; }
export PATH=/usr/bin:/bin
unset CONDA_PREFIX CMAKE_PREFIX_PATH CMAKE_ARGS LD_LIBRARY_PATH LD_PRELOAD PYTHONPATH
unset CC CXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS
cmake -S "$adapter_dir" -B "$source_dir/build" -DSQRTVINS_SOURCE="$source_dir" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE='-O1 -DNDEBUG'
cmake --build "$source_dir/build" -j1
