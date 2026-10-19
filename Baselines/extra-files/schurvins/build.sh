#!/usr/bin/env bash
set -euo pipefail
adapter_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(realpath "$adapter_dir/../../SchurVINS")
revision=d8ab6dff20860694d8d879d716d7fd6cb2c2530e
if [[ ! -d "$source_dir/.git" ]]; then
 git clone --filter=blob:none --no-checkout https://github.com/bytedance/SchurVINS.git "$source_dir"
 git -C "$source_dir" sparse-checkout init --cone
 git -C "$source_dir" sparse-checkout set svo svo_common svo_direct svo_img_align svo_tracker svo_vio_common svo_ros vikit rpg_common
 git -C "$source_dir" checkout --detach "$revision"
fi
[[ $(git -C "$source_dir" rev-parse HEAD) == "$revision" ]] || { echo 'Unexpected SchurVINS revision; checkout preserved.' >&2; exit 1; }
for patch in "$adapter_dir"/*.patch; do
 if git -C "$source_dir" apply --check "$patch" 2>/dev/null; then git -C "$source_dir" apply "$patch"
 else git -C "$source_dir" apply --reverse --check "$patch"; fi
done
export PATH=/usr/bin:/bin
unset CONDA_PREFIX CMAKE_PREFIX_PATH CMAKE_ARGS LD_LIBRARY_PATH LD_PRELOAD PYTHONPATH CC CXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS
mkdir -p "$source_dir/build/deps"
clone_pinned() {
 local repo_url=$1 repo_name=$2 repo_revision=$3
 if [[ ! -d "$source_dir/build/deps/$repo_name/.git" ]]; then
  git clone --filter=blob:none --no-checkout "$repo_url" "$source_dir/build/deps/$repo_name"
  git -C "$source_dir/build/deps/$repo_name" checkout --detach "$repo_revision"
 fi
 [[ $(git -C "$source_dir/build/deps/$repo_name" rev-parse HEAD) == "$repo_revision" ]]
}
clone_pinned https://github.com/ethz-asl/minkindr.git minkindr 564f12639a8447d4d3e5e7707851424302941056
clone_pinned https://github.com/uzh-rpg/fast_neon.git fast_neon 4914ed963aba8d2bb476551e186e4b20a206f03d
# Reuse the existing pinned OpenGV build, or build that dependency locally.
opengv_root=${SCHURVINS_OPENGV_ROOT:-$(realpath -m "$source_dir/../OKVIS1/build")}
if [[ ! -f "$opengv_root/lib/libopengv.a" || ! -d "$opengv_root/include/opengv" ]]; then
 opengv_source="$source_dir/build/deps/opengv"
 if [[ ! -d "$opengv_source/.git" ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/laurentkneip/opengv.git "$opengv_source"
  git -C "$opengv_source" checkout --detach 306a54e6c6b94e2048f820cdf77ef5281d4b48ad
 fi
 [[ $(git -C "$opengv_source" rev-parse HEAD) == 306a54e6c6b94e2048f820cdf77ef5281d4b48ad ]]
 opengv_root="$source_dir/build/deps/opengv_install"
 cmake -S "$adapter_dir/opengv" -B "$source_dir/build/deps/opengv_build" -DOPENGV_SOURCE="$opengv_source" -DCMAKE_INSTALL_PREFIX="$opengv_root" -DCMAKE_BUILD_TYPE=Release
 cmake --build "$source_dir/build/deps/opengv_build" -j1
 cmake --install "$source_dir/build/deps/opengv_build"
fi
python3 "$adapter_dir/prepare_sources.py" "$source_dir"
cmake -S "$adapter_dir" -B "$source_dir/build" -DSCHURVINS_SOURCE="$source_dir" -DOPENGV_ROOT="$opengv_root" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE='-O1 -DNDEBUG'
cmake --build "$source_dir/build" -j1
