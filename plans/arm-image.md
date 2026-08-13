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
| CUDA base | 12.4.1 U22.04 | Use NGC **arm64** tag that supports host driver + Blackwell; likely **newer than 12.4** |
| Ubuntu in image | 22.04 | Match CUDA image distro (22.04 or 24.04) |
| MOFED | 23.07 x86_64 | aarch64 package for same/compatible distro; confirm Grace/IB stack |
| OpenMPI | 5.0.5 | source build (same); drop/fix mellanox platform flag if configure fails on arm |
| NCCL | image `libnccl2` | must support Blackwell; may need newer package/CUDA |
| nccl-tests | 2.13.8 | source build against container CUDA/NCCL |
| perftest | 23.10.0 | source build; GDR needs `CUDA_H_PATH` |
| nvbandwidth | 0.4 | bump if needed for Blackwell; arches e.g. `90;100` (**confirm B300 sm**) |
| NHC | 1.4.3 | source build (perl/bash; should be arch-ok) |
| STREAM | AOCC amd64 | **no AOCC on arm** — build with `gcc`/`clang -fopenmp`, or ship STREAM only on x86 image |

STREAM is only required for HB/HX CPU confs. Arm image can:

1. Build STREAM with distro GCC OpenMP, or  
2. Omit STREAM binary and document CPU STREAM unsupported on arm until needed.

Prefer (1) if cheap so the arm image stays a superset for future CPU Grace SKUs.

---

## Implementation checklist

### 1. Research / pins (half-day on node)

- [ ] `uname -m`, driver version, `nvidia-smi`
- [ ] Confirm docker + NVIDIA container toolkit on Grace
- [ ] Choose CUDA base image digest/tag that pulls on aarch64 and sees GPUs with `--gpus all`
- [ ] Locate MOFED aarch64 user-space tarball matching distro
- [ ] Note NCCL package version in that CUDA image
- [ ] Confirm Blackwell GPU arch id for nvcc (`sm_100` / `sm_103` / etc.)

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
| CUDA 12.4 too old for B300 | Newer NGC arm base; pin tested driver↔CUDA combo |
| No MOFED aarch64 for chosen Ubuntu | Align Ubuntu to available MOFED; or use DOCA/host OFED libs (last resort) |
| OpenMPI mellanox platform flag fails | Configure without `--with-platform=...` on arm |
| nvbandwidth won’t build for Blackwell | Upgrade nvbandwidth; set correct `CMAKE_CUDA_ARCHITECTURES` |
| AOCC unavailable | GCC OpenMP STREAM or omit |
| Host OFED vs container OFED mismatch | user-space-only install; match major version to host if IB fails |
| Image large / slow build | Keep multi-stage; no change to shrink goals beyond x86 parity |

### Open decisions (resolve in PR description)

- [ ] Final image name/tag
- [ ] Final CUDA / MOFED / NCCL versions
- [ ] STREAM: build with GCC vs omit on arm
- [ ] Whether `cuda` build_type auto-picks arm on aarch64 hosts

---

## Handoff to SKU PRs

After merge + image available locally (or on MCR):

1. [`gb300.md`](./gb300.md) — conf, topology, thresholds, AN160, tests  
2. Future `gb200.md` — same image, different conf  

SKU PRs must not re-litigate Dockerfile contents unless a missing binary/library blocks a check.
