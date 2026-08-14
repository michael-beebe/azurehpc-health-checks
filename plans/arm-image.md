# PLAN: aarch64 NVIDIA Az NHC Image

**Branch (implementation):** `feature/aznhc-nv-aarch64`  
**Depends on:** nothing (land before GB300/GB200 SKU PRs)  
**Consumers:** ND GB300 v6, ND GB200 v6, any future Grace+NVIDIA SKUs  
**Fork plans branch only** — do not merge this doc to upstream `main`

---

## Goal

Ship an **aarch64 NVIDIA** container that can run the existing NHC entrypoint and custom tests on Grace hosts.

**Done when** (on a Grace+NVIDIA node):

- [ ] Image builds natively on aarch64
- [ ] `run-health-checks.sh` selects it automatically when `uname -m == aarch64` and NVIDIA is present
- [ ] Inside container: `nvidia-smi`, `nhc`, `mpirun`, `all_reduce_perf`, `ib_write_bw`, `nvbandwidth` all run
- [ ] x86 path unchanged (`aznhc-nv` still used on x86_64)
- [ ] Docs cover build/pull for arm

**Out of scope**

- GB300/GB200 conf files and thresholds → [`gb300.md`](./gb300.md)
- ROCm arm image
- Multi-arch manifest on `latest` (optional follow-up; start with explicit arm tag)
- MCR publish mechanics owned by maintainers (PR should still name the tag)

---

## Current x86 image (baseline to port)

| Piece | Today (`dockerfile/azure-nvrt-nhc.dockerfile`) |
|-------|-----------------------------------------------|
| Base | `nvcr.io/nvidia/cuda:12.4.1-*-ubuntu22.04` |
| Stages | builder (devel) → runtime |
| OFED | `MLNX_OFED_*_ubuntu22.04-x86_64.tgz` **both stages** |
| Builds | OpenMPI 5.0.5, nccl-tests 2.13.8, NHC 1.4.3, perftest GDR+non-GDR, nvbandwidth 0.4, STREAM via **AOCC amd64 deb** |
| nvbandwidth arch | `70;80;90` only |
| Topofiles | clone `Azure/azhpc-images` → `${AZ_NHC_ROOT}/topofiles` |
| Runtime pkgs | numactl, hwloc, bc, pciutils, libnccl2, bats, … |
| Entry | `aznhc-entrypoint.sh`; confs in `/azure-nhc/default/conf`; tests in `/etc/nhc/scripts/` |
| Tag | `mcr.microsoft.com/aznhc/aznhc-nv` |

**Hard x86 assumptions to eliminate:** OFED tarball arch, CUDA image arch, AOCC `.deb`, nvbandwidth arch list, any `linux-x86_64` downloads.

---

## Recommended design

### Image naming

| Arch | Image |
|------|--------|
| x86_64 | `mcr.microsoft.com/aznhc/aznhc-nv` (unchanged) |
| aarch64 | `mcr.microsoft.com/aznhc/aznhc-nv:aarch64` **or** `.../aznhc-nv-arm:latest` |

Pick one tag scheme in the PR and use it consistently in build/pull/runner/docs.  
Prefer **`aznhc-nv:aarch64`** if MCR allows tags on existing repo name.

### Dockerfile strategy

**Preferred:** new file `dockerfile/azure-nvrt-nhc-aarch64.dockerfile`  
Copy structure from x86 file; do **not** fork logic into one mega-Dockerfile with unreadable conditionals (unless a thin shared pattern stays obvious).

Keep multi-stage: **builder → runtime**.

### Version pins (resolve during implementation; record finals here)

| Component | x86 today | aarch64 action |
|-----------|-----------|----------------|
| CUDA base | 12.4.1 U22.04 | **Confirmed:** `nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04` (arm64) — 12.6.3 lacks Blackwell (`compute_100/103`); 13.0.0 has it |
| Ubuntu in image | 22.04 | **24.04** (matches CUDA 13 arm64 image; host is also 24.04) |
| MOFED | 23.07 x86_64 | **No MOFED tarball for arm.** Use DOCA-Host public apt repo (`linux.mellanox.com/public/repo/doca/3.3.0/ubuntu24.04/arm64-sbsa/`), install specific `rdma-core`/`perftest`/`ucx` packages (NOT `doca-all`/`doca-networking-runtime` meta-packages — too heavy). Versions match host exactly (`rdma-core` 2601.0.7-1). |
| OpenMPI | 5.0.5 | source build (same); drop/fix mellanox platform flag if configure fails on arm — **not yet validated** |
| NCCL | image `libnccl2` | CUDA13 sbsa apt repo only has NCCL up to `2.27.7-1+cuda12.9` (no cuda13 build yet) — build from source against container CUDA/NCCL like x86 does, don't rely on apt package |
| nccl-tests | 2.13.8 | source build against container CUDA/NCCL — not yet validated on arm |
| perftest | 23.10.0 (source build) | **Likely no source build needed** — DOCA repo ships prebuilt `perftest 26.01.5-1` with `--use_cuda`/`--use_cuda_dmabuf`/`--use_data_direct` GDR flags already present. Still need an end-to-end GDR run to confirm (only `--help` checked so far) |
| nvbandwidth | 0.4 | bump if needed for Blackwell; arches **`100;103`** confirmed via `nvcc --list-gpu-arch` on CUDA 13.0.0 arm64 (host GPU compute_cap is `10.3`) |
| NHC | 1.4.3 | source build (perl/bash; should be arch-ok) |
| STREAM | AOCC amd64 | **no AOCC on arm** — build with `gcc`/`clang -fopenmp`, or ship STREAM only on x86 image — not yet validated |

STREAM is only required for HB/HX CPU confs. Arm image can:

1. Build STREAM with distro GCC OpenMP, or  
2. Omit STREAM binary and document CPU STREAM unsupported on arm until needed.

Prefer (1) if cheap so the arm image stays a superset for future CPU Grace SKUs.

---

## Implementation checklist

### 1. Research / pins (half-day on node)

- [x] `uname -m`, driver version, `nvidia-smi` — done on a real ND128isr_GB300_v6 node (see findings below)
- [x] Confirm docker + NVIDIA container toolkit on Grace — confirmed working
- [x] Choose CUDA base image digest/tag that pulls on aarch64 and sees GPUs with `--gpus all` — `nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04`
- [x] Locate MOFED aarch64 user-space tarball matching distro — **no tarball; use DOCA-Host apt repo instead (see below)**
- [x] Note NCCL package version in that CUDA image — see below (may need source build for CUDA 13)
- [x] Confirm Blackwell GPU arch id for nvcc (`sm_100` / `sm_103` / etc.) — **`compute_103`/`sm_103`** (host reports compute_cap `10.3`)

#### Findings (from a live ND128isr_GB300_v6 node, 2026-08-14)

| Item | Value |
|------|-------|
| `uname -m` | `aarch64` |
| OS | Ubuntu 24.04.4 LTS (noble) |
| Kernel | `6.17.0-1005-azure-nvidia` |
| NVIDIA driver | `580.126.20` |
| GPU | 4× `NVIDIA GB300`, compute_cap `10.3` → nvcc arch `compute_103`/`sm_103` |
| Host CUDA toolkit | `13.0.2` (`cuda-toolkit-13-0`) |
| Docker | 29.2.1, `nvidia` runtime installed (`libnvidia-container` 1.18.2), default runtime still `runc` — must pass `--gpus all` |
| CPU | 2× socket, Neoverse-V2, 128 vCPU total, NUMA reported as **34 nodes** (not 2 — verify NHC `check_hw_cpuinfo`/topology parsing handles this) |
| Memory | 862 GiB total |
| NVMe | 4 data disks (3.84 TB Micrsoft NVMe Direct Disk v2 each) + 1 small 68GB "MSFT NVMe Accelerator" boot/temp disk (5 nvme devices total, not 4 — confirm which are health-checked) |
| IB | 4× `mlx5_ib0..3`, **800 Gb/sec (4X XDR)**, link_layer InfiniBand |
| Net | `eth0`/`eth1` (bond members), no visible `mlx5_an0` accelerated-network device |

**CUDA base image arch support (critical pin):**
- `nvcr.io/nvidia/cuda:12.6.3-devel-ubuntu24.04` (arm64) pulls fine and sees GPUs, but `nvcc --list-gpu-arch` tops out at `compute_90` — **no Blackwell support**.
- `nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04` (arm64) pulls fine, sees GPUs, and lists `compute_100/103/110/120/121` — **this is the correct base**, confirming the plan's suspicion that x86's CUDA 12.4 must be bumped for the arm image.
- **Decision: pin arm image to CUDA 13.0.x, not 12.4.1.**

**MOFED replacement — DOCA-Host (major finding):**
The host does **not** use a classic `MLNX_OFED_*.tgz` tarball. It uses NVIDIA **DOCA-Host** (`doca-host` apt package, installed version `3.3.0-088000-26.01-ubuntu2404`), which lives in a *local* repo (`/usr/share/doca-host-.../repo`) set up by a downloaded installer. However, DOCA also publishes a **public, unauthenticated network apt repo** that mirrors host package versions exactly:

```bash
wget -qO - https://linux.mellanox.com/public/repo/doca/GPG-KEY-Mellanox.pub | gpg --dearmor -o /usr/share/keyrings/GPG-KEY-Mellanox.pub
echo "deb [signed-by=/usr/share/keyrings/GPG-KEY-Mellanox.pub] https://linux.mellanox.com/public/repo/doca/3.3.0/ubuntu24.04/arm64-sbsa/ ./" \
  > /etc/apt/sources.list.d/doca.list
apt-get update
```
Verified reachable (HTTP 200) and verified in a throwaway `ubuntu:24.04` arm64 container:
- `rdma-core` candidate `2601.0.7-1` — **matches host exactly**.
- Installing `rdma-core ibverbs-providers libibverbs-dev librdmacm-dev libibumad-dev infiniband-diags` works cleanly (only harmless udev "Read-only file system" warnings, expected in containers).
- Avoid the `doca-networking-runtime`/`doca-all` meta-packages — they drag in OVS, DPDK, telemetry, etc. (way too heavy for a compute-node health-check image). Install the **specific rdma-core/perftest/ucx packages directly** instead.
- **`perftest` is available prebuilt from this repo at `26.01.5-1`** and its `ib_write_bw --help` already lists `--use_cuda`, `--use_cuda_dmabuf`, `--use_data_direct` (GPUDirect RDMA flags) — **we likely do NOT need to build perftest from source on arm** (unlike the x86 image, which builds perftest 23.10.0 from source). Big Dockerfile simplification if this holds up under a real GDR run.
- `ucx` also available prebuilt at `1.20.0-1.20260211...` matching host.

**Decision: replace the x86 "download MLNX_OFED tarball + run mlnxofedinstall" step with "add the DOCA public apt repo + apt-get install specific rdma-core/perftest/ucx packages" for the arm Dockerfile.** This is simpler than the x86 approach, not just a port.

**NCCL package version:** CUDA 13.0 sbsa apt repo only publishes NCCL up to `2.27.7-1+cuda12.9` (no `+cuda13.x` build yet as of this check). Since nccl-tests/NCCL need to be source-built anyway per the existing x86 process, this isn't blocking, but note NCCL apt packages lag the CUDA 13 host toolkit — building against source or the bundled NCCL in the CUDA devel image is the safer path (as x86 already does).

**Update 2026-08-14 — Dockerfile written and built successfully end-to-end** (`dockerfile/azure-nvrt-nhc-aarch64.dockerfile` on branch `feature/aznhc-nv-aarch64`):

- [x] Full multi-stage build (49 steps) completed on the live GB300 node with no manual intervention beyond one fix (below).
- [x] `nvidia-smi -L` inside container lists all 4 GB300 GPUs (`--gpus all --privileged --net=host`).
- [x] **nccl-tests build fix required:** its default `NVCC_GENCODE` targets legacy archs (`compute_35/50/60/61/70/80`) that CUDA 13's `nvcc` rejects outright (`nvcc fatal: Unsupported gpu architecture 'compute_60'`). Fixed by passing `NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_100,code=sm_100 -gencode=arch=compute_103,code=sm_103 -gencode=arch=compute_103,code=compute_103"` explicitly to `make`. Same class of issue would hit any other CUDA-13-era source build with hardcoded old gencode flags — watch for it.
- [x] OpenMPI 5.0.5 built fine from source on arm64 using `--with-verbs --with-rdmacm` (dropped the x86 `--with-platform=contrib/platform/mellanox/optimized` flag entirely rather than trying to fix it — simpler and worked first try).
- [x] nvbandwidth v0.10.0 built with `CMAKE_CUDA_ARCHITECTURES="90;100;103"` and **ran successfully** inside the container: `device_to_host_memcpy_ce` measured ~193.5 GB/s per GPU (774 GB/s summed across 4 GPUs), CoV 0.00.
- [x] NCCL allreduce (`all_reduce_perf` from nccl-tests, single-node, 4 ranks, 8MB, `--map-by ppr:4:node -bind-to numa`) **ran successfully**: ~290 GB/s avg bus bandwidth over NVLink — matches the existing `azure_nccl_allreduce.nhc` invocation pattern.
- [x] `ib_write_bw` (DOCA-packaged, GDR loopback via `--use_cuda=0` against own hostname) **ran successfully after loading `nvidia_peermem`**: measured 420 Gb/sec. **Important operational note:** the `nvidia_peermem` kernel module was NOT loaded on this host by default (`lsmod` showed nothing); GPUDirect RDMA memory registration fails with `Couldn't allocate MR with error=14` until it's loaded (`sudo modprobe nvidia-peermem`, module present at `/lib/modules/$(uname -r)/updates/dkms/nvidia-peermem.ko.zst`). This is a **host/image-provisioning prerequisite**, not something the container can fix — needs a callout in docs/developer_guide (and possibly a pre-flight check in `run-health-checks.sh` or the GDR test itself) so a missing module doesn't get misread as a hardware fault.
- [x] STREAM omitted from the arm image as planned (documented in the Dockerfile header comment); not needed for GB300 GPU conf.

**Still open / not yet validated:**
- [ ] 34 NUMA nodes reported by `lscpu` (not 2 sockets as naively expected) — check how this affects `check_hw_cpuinfo`, `check_hw_topology`, and any NUMA-aware binding logic (`numactl -N $numa_node` in `azure_ib_write_bw_gdr.nhc`) before assuming x86-style 1-socket-per-NUMA-node logic holds.
- [ ] 5 nvme devices visible (1 small boot-ish disk + 4 large data disks) vs. plan's "4 disks" — confirm which the SKU doc/health check should count.
- [ ] Real (non-loopback) multi-HCA IB bandwidth across all 4 `mlx5_ib0..3` devices simultaneously (only device 0 loopback tested so far).
- [ ] `run-health-checks.sh` arch-based image selection, `build_image.sh`/`pull-image-mcr.sh` `cuda-arm` target, and `test/unit-tests/run_tests.sh` image name — not yet wired up (Dockerfile-only so far, per commit plan step 1).

### 2. Dockerfile

- [ ] Add `dockerfile/azure-nvrt-nhc-aarch64.dockerfile`
- [ ] Builder: arm CUDA devel base, apt build deps (same set as x86)
- [ ] OFED install with **aarch64** tarball (builder + runtime)
- [ ] OpenMPI configure/make/install → `/opt/openmpi`
- [ ] nccl-tests with `MPI=1`
- [ ] NHC install to same paths as x86 (`/usr/sbin/nhc`, `/etc/nhc`, …)
- [ ] perftest → `${AZ_NHC_ROOT}/bin/ib_write_bw` + `ib_write_bw_nongdr`
- [ ] nvbandwidth with Blackwell-inclusive `CMAKE_CUDA_ARCHITECTURES`
- [ ] STREAM via GCC/Clang OpenMP (or skip with comment)
- [ ] Topofiles copy from azhpc-images (same as x86; missing GB300 topo is OK for this PR)
- [ ] Runtime stage: slim packages + OFED + COPY --from=builder (mirror x86 file list)
- [ ] COPY conf, customTests, entrypoint, LICENSE (same as x86)

### 3. Build / pull scripts

`dockerfile/build_image.sh`:

```text
cuda      → existing x86 dockerfile + aznhc-nv
cuda-arm  → aarch64 dockerfile + aznhc-nv:aarch64 (or aznhc-nv-arm)
rocm      → unchanged
```

Optional: if `build_type=cuda` and host is aarch64, default to arm dockerfile (convenient; document it).

`dockerfile/pull-image-mcr.sh`: same `cuda` / `cuda-arm` / `rocm` switch.

### 4. Runner selection

In `run-health-checks.sh` image pick (today: NVIDIA → `aznhc-nv`, AMD → rocm, else nv):

```text
if NVIDIA and aarch64 → DOCK_IMG_NAME_NV_ARM
if NVIDIA and x86_64  → DOCK_IMG_NAME_NV
if AMD                → DOCK_IMG_NAME_AMD
else                  → arch-appropriate NV image (CPU checks)
```

Also update:

- [ ] `test/unit-tests/run_tests.sh` hard-coded `aznhc-nv`
- [ ] Any docs examples using a single image name

Do **not** change conf/AN/SKU logic in this PR.

### 5. Docs

- [ ] `dockerfile/README.MD` — pull/build arm, tag name, GPU runtime flags on Grace
- [ ] `developer_guide.md` — one short note: arm image required for Grace SKUs
- [ ] Root README only if you mention multi-arch images (keep light)

### 6. Validate on Grace node

```bash
# build
sudo dockerfile/build_image.sh cuda-arm

# GPU visibility
sudo docker run --rm --gpus all --privileged --net=host \
  mcr.microsoft.com/aznhc/aznhc-nv:aarch64 nvidia-smi

# tool smoke (interactive or docker run bash -c)
which nhc mpirun
ls /opt/nccl-tests/build/all_reduce_perf
$AZ_NHC_ROOT/bin/ib_write_bw --help
$AZ_NHC_ROOT/bin/nvbandwidth --help   # or -h

# single-node nccl-tests (4 or N local GPUs)
mpirun -np $(nvidia-smi -L | wc -l) --allow-run-as-root \
  /opt/nccl-tests/build/all_reduce_perf -b 8M -e 8M -f 2 -g 1

# runner still fails SKU detect without conf — OK; force a known conf or expect "not supported"
sudo ./run-health-checks.sh -v -t 120   # should at least start correct image
```

Acceptance:

| Check | Pass criteria |
|-------|----------------|
| Build | exit 0 on aarch64 |
| GPU in container | `nvidia-smi` lists GPUs |
| NCCL tests binary | runs, nonzero BW line |
| ib_write_bw | starts (loopback GDR may need devices) |
| Runner arch pick | docker uses arm tag on Grace; x86 host still x86 tag |
| Regression | no changes to rocm dockerfile behavior |

---

## Commit plan (this PR)

1. `feat(docker): add aarch64 NVIDIA Dockerfile`  
   - new dockerfile only
2. `feat(docker): build/pull cuda-arm image target`  
   - `build_image.sh`, `pull-image-mcr.sh`
3. `feat(runner): select aarch64 aznhc image on Grace hosts`  
   - `run-health-checks.sh`, test runner image name
4. `docs: document aarch64 NVIDIA image build and pull`  
   - dockerfile README + developer_guide blurb

---

## Files touched (expected)

```
dockerfile/azure-nvrt-nhc-aarch64.dockerfile   # new
dockerfile/build_image.sh
dockerfile/pull-image-mcr.sh
dockerfile/README.MD
run-health-checks.sh
test/unit-tests/run_tests.sh
developer_guide.md
README.md                                      # optional one-liner
```

Unchanged on purpose: `conf/*`, `customTests/*` (except if STREAM makefile needs an arm recipe), ROCm dockerfile.

---

## Risks / decisions

| Risk | Mitigation |
|------|------------|
| CUDA 12.4 too old for B300 | **Resolved:** pin to CUDA 13.0.x arm64 (confirmed has `compute_100/103`) |
| No MOFED aarch64 for chosen Ubuntu | **Resolved:** no tarball; use DOCA-Host public apt repo (`linux.mellanox.com/public/repo/doca/3.3.0/ubuntu24.04/arm64-sbsa/`), install specific rdma-core/perftest/ucx packages, skip heavy `doca-all`/networking-runtime meta-packages |
| OpenMPI mellanox platform flag fails | Configure without `--with-platform=...` on arm — still to be validated |
| nvbandwidth won’t build for Blackwell | Set `CMAKE_CUDA_ARCHITECTURES=100;103` (confirmed via CUDA 13.0.0 `nvcc --list-gpu-arch`) |
| AOCC unavailable | GCC OpenMP STREAM or omit — still to be validated |
| Host OFED vs container OFED mismatch | **Resolved via DOCA apt repo** — package versions (`rdma-core` 2601.0.7-1) matched host exactly in testing |
| Image large / slow build | Keep multi-stage; no change to shrink goals beyond x86 parity |
| 34 NUMA nodes reported (not 2 sockets) | New risk found during inventory — verify `check_hw_cpuinfo`/topology/`numactl -N` binding logic in `azure_ib_write_bw_gdr.nhc` handles this before assuming x86-style numbering |
| 5 nvme devices vs 4 expected | New risk found during inventory — 1 is a small "MSFT NVMe Accelerator" boot-adjacent disk; confirm which count the GB300 conf should check |

### Open decisions (resolve in PR description)

- [ ] Final image name/tag
- [x] Final CUDA version — **13.0.x arm64** (`nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04` or later 13.0.z patch)
- [x] MOFED replacement — **DOCA-Host public apt repo**, not a tarball
- [ ] Final NCCL version (source build against CUDA 13, apt package lags at +cuda12.9)
- [ ] STREAM: build with GCC vs omit on arm
- [ ] Whether `cuda` build_type auto-picks arm on aarch64 hosts
- [ ] Whether to source-build perftest at all, or trust the DOCA apt package (26.01.5-1) after an end-to-end GDR validation run

---

## Handoff to SKU PRs

After merge + image available locally (or on MCR):

1. [`gb300.md`](./gb300.md) — conf, topology, thresholds, AN160, tests  
2. Future `gb200.md` — same image, different conf  

SKU PRs must not re-litigate Dockerfile contents unless a missing binary/library blocks a check.
