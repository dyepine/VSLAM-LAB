#!/usr/bin/env bash
set -euo pipefail
adapter_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(realpath "$adapter_dir/../../Voxel-SVIO")
revision=eac6b87d0f925b96ddd5c5c6b0dcda3d36c7ab5f
if [[ ! -d "$source_dir/.git" ]]; then
    git clone --filter=blob:none --no-checkout https://github.com/ZikangYuan/voxel_svio.git "$source_dir"
    git -C "$source_dir" sparse-checkout init --cone
    git -C "$source_dir" sparse-checkout set src include config thirdLibrary/tessil-src/include
    git -C "$source_dir" checkout --detach "$revision"
fi
[[ $(git -C "$source_dir" rev-parse HEAD) == "$revision" ]] || { echo 'Unexpected Voxel-SVIO revision; checkout preserved.' >&2; exit 1; }
for patch in offline compatibility memory_safety; do
    if git -C "$source_dir" apply --check "$adapter_dir/$patch.patch" 2>/dev/null; then
        git -C "$source_dir" apply "$adapter_dir/$patch.patch"
    else
        git -C "$source_dir" apply --reverse --check "$adapter_dir/$patch.patch"
    fi
done
export PATH=/usr/bin:/bin
unset CONDA_PREFIX CMAKE_PREFIX_PATH CMAKE_ARGS LD_LIBRARY_PATH LD_PRELOAD PYTHONPATH
unset CC CXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS
deps_dir="$source_dir/build/deps"
mkdir -p "$deps_dir"
if [[ ! -f "$deps_dir/pcl/usr/include/pcl-1.12/pcl/point_cloud.h" ]]; then
    curl --fail --location --retry 5 --connect-timeout 30 \
        https://archive.ubuntu.com/ubuntu/pool/universe/p/pcl/libpcl-dev_1.12.1+dfsg-3build1_amd64.deb \
        -o "$deps_dir/pcl.deb"
    printf '%s  %s\n' 0a199482588ae37d81db83bdbe2523ba3ad4e57714733e8b126182b76e770f68 "$deps_dir/pcl.deb" | sha256sum -c -
    dpkg-deb -x "$deps_dir/pcl.deb" "$deps_dir/pcl"
fi
cmake -S "$adapter_dir" -B "$source_dir/build" -DVOXEL_SOURCE="$source_dir" \
    -DPCL_HEADERS="$deps_dir/pcl/usr/include/pcl-1.12" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE='-O1 -DNDEBUG'
cmake --build "$source_dir/build" -j1
