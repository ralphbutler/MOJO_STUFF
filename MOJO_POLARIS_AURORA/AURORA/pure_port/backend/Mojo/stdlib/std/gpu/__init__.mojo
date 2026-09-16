# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Mojo 1.0 compatibility: the `std.gpu` import path.

AURORA PATCH (G5): Mojo 1.0.0 exported the GPU thread-indexing primitives from
`std.gpu`; later versions moved them to the private `std._gpu` package, with
`max.gpu` as the public home. This package re-exports the 1.0.0 names so Mojo 1.0
GPU programs (e.g. `from std.gpu import global_idx`) compile unchanged.
"""

from std._gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_dim,
    block_id_in_cluster,
    block_idx,
    cluster_dim,
    cluster_idx,
    global_idx,
    grid_dim,
    lane_id,
    sm_id,
    thread_idx,
    warp_id,
)
