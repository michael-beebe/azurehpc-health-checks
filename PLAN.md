# PLAN: Standard_ND128isr_GB300_v6 Support

**Branch:** `feature/gb300-v6-support`  
**SKU:** `Standard_ND128isr_GB300_v6`  
**Conf name (IMDS-normalized):** `nd128isr_gb300_v6`  
**Out of scope for this PR:** `Standard_ND*_GB200_v6` (separate PR), multi-node NCCL fabric tests  
**Related follow-up SKU:** `Standard_ND128isrG5_GB300_v6` (same series; add conf once primary works)

---

## 1. Goals

Bring Az NHC to functional parity (single-node health + perf checks) on **ND GB300 v6**:

1. Auto-detect SKU via IMDS and select `conf/nd128isr_gb300_v6.conf`.
2. Run inside a working **NVIDIA + aarch64 (Grace)** container image.
3. Validate hardware presence: CPU, memory, NVMe, eth, IB, GPU count/health.
4. Validate performance floors: GPU BW, NVLink/NCCL allreduce, IB GDR BW.
5. Wire topology, clock boost, AN handling, tests, and docs.

Success criteria:

- [ ] `sudo ./run-health-checks.sh` on a healthy GB300 node exits 0 with default conf.
- [ ] `sudo ./run-health-checks.sh -a -v` completes without NHCNA env/tooling faults.
- [ ] Unit/integration tests updated for GB300 where hardware-gated.
- [ ] README lists GB300 and documents expected thresholds (once calibrated).

Non-goals:

- Multi-node / rack-scale NCCL (Az NHC remains per-node; distributed_nhc only fans out).
- Full GB200 support (separate PR; may share Arm image work).
- Publishing MCR images (coordinate with repo owners after local build verify).

---

## 2. Target hardware (from Microsoft docs)

| Component | ND128isr_GB300_v6 |
|-----------|-------------------|
| CPU | 2× NVIDIA Grace, **128 vCPUs**, **aarch64** |
| Memory | **864 GiB** LPDDR5X |
| GPUs | **4×** NVIDIA Blackwell Ultra B300 (288 GB HBM3E each) |
| GPU interconnect | NVLink 5th gen (~4× 1.8 TB/s aggregate per VM) |
| IB | **800 Gbps per GPU** (expect ~4 IB endpoints; **verify names/rates on node**) |
| Ethernet | **160 Gbps** (1 NIC) — **not** covered by current AN40/AN100 lists |
| Local NVMe | **4 disks**, 16 TiB total |

Docs: [ND GB300-v6 size series](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nd-gb300-v6-series)

Closest existing conf templates: `conf/nd96isr_h100_v5.conf`, `conf/nd96isr_h200_v5.conf`  
**Do not copy thresholds blindly** — GPU count, IB rate, arch, and BW floors all differ.

---

## 3. Repo audit — current state vs GB300 gaps

### 3.1 Architecture overview (unchanged)

```
run-health-checks.sh
  → detect SKU (IMDS)
  → conf/<sku>.conf
  → docker run mcr.microsoft.com/aznhc/aznhc-nv|rocm
       → aznhc-entrypoint.sh
            → copy customTests/*.nhc → /etc/nhc/scripts/
            → nhc CONFFILE=... LOGFILE=... TIMEOUT=...
```

SKU support today = **conf file exists** + **image can run checks** + **SKU-specific code branches**.

### 3.2 Images

| Image | Role | Arch today |
|-------|------|------------|
| `mcr.microsoft.com/aznhc/aznhc-nv` | NVIDIA/CUDA + CPU fallback | **x86_64 only** |
| `mcr.microsoft.com/aznhc/aznhc-rocm` | AMD/ROCm | **x86_64 only** |

Evidence:

- `dockerfile/azure-nvrt-nhc.dockerfile` pulls `ubuntu22.04-x86_64` OFED, CUDA 12.4 x86 base.
- `nvbandwidth` built with `-DCMAKE_CUDA_ARCHITECTURES="70;80;90"` (no Blackwell).
- `run-health-checks.sh` / `build_image.sh` / `pull-image-mcr.sh` only choose `cuda` vs `rocm`.

**Gap:** No aarch64 NVIDIA image. GB300 cannot use current `aznhc-nv` as-is.

### 3.3 NVIDIA + IB SKUs already supported (conf present)

| Conf | GPUs | IB rails in conf | IB perf |
|------|-----:|------------------|---------|
| `nd40rs_v2` | 8 | 1×100 | non-GDR |
| `nd96asr_v4` | 8 | 8×200 | GDR 180 |
| `nd96amsr_a100_v4` | 8 | 8×200 | GDR 180 |
| `nd96isr_h100_v5` | 8 | 8×400 | GDR 380 |
| `nd96isr_h200_v5` | 8 | 8×400 (+AN) | GDR 360 |
| `nd96isrf_h200_v5` | 8 | 8×400 (+AN) | GDR 360 |

Partial / non-IB NVIDIA: `nd96is_h100_v5`, `nd96is_h200_v5`, NC A100/H100 family.  
**No GB200/GB300 confs.**

### 3.4 File touchpoints for GB300

| Path | Why it matters | GB300 action |
|------|----------------|--------------|
| `conf/nd128isr_gb300_v6.conf` | **Missing** — required for auto SKU detect | Add |
| `run-health-checks.sh` | Image select; AN40/AN100 only (40/100 Gbps) | Arm image select; AN160 (or eth-only) handling |
| `dockerfile/azure-nvrt-nhc.dockerfile` | x86 CUDA 12.4, OFED x86, sm70/80/90 | New aarch64 Dockerfile or multi-arch |
| `dockerfile/build_image.sh` | `cuda`/`rocm` only | Add `cuda-arm` / arch flag |
| `dockerfile/pull-image-mcr.sh` | Same | Pull arm tag when applicable |
| `dockerfile/aznhc-entrypoint.sh` | SKU conf fallback under `/azure-nhc/default/conf` | Works if conf baked/mounted |
| `dockerfile/README.MD` | Run/build docs | Document arm image |
| `customTests/azure_hw_topology_check.nhc` | Only ndv4 / ncv4 / `nd96isr*_v5` | Add GB300 topo mapping |
| `customTests/azure_common.nhc` | `boost_gpu_clock` only A100/H100 | Add GB300 clocks if needed |
| `customTests/azure_nccl_allreduce.nhc` | Local GPUs via `nvidia-smi` (OK for 4 GPU) | Verify env/topo; calibrate BW |
| `customTests/azure_ib_write_bw_gdr.nhc` | Ports sized for 8; `dev_idx`↔CUDA index assumed | Verify 4-rail GB300 mapping |
| `customTests/azure_gpu_bandwidth.nhc` | Needs working `nvbandwidth` on Blackwell | Image rebuild + thresholds |
| `customTests/azure_nvme_count.nhc` | Conf-driven count | Conf: 4 |
| Topofiles | Loaded from azhpc-images at **image build** | Need `ndv6-gb300-topo.xml` (name TBD) in image/repo |
| `test/unit-tests/nhc-test-common.sh` | sad-path conf / topo / ib_type SKU switches | Add GB300 |
| `test/bad_test_confs/` | No GB300 | Add optional sad conf |
| `test/unit-tests/run_tests.sh` | Hardcodes `aznhc-nv` | Arm image aware |
| `README.md` / `developer_guide.md` | Supported SKUs, AN lists, tables | Document GB300 |
| `distributed_nhc/` | Uses same per-node runner/conf | Works once conf+image exist |
| CI `azure-pipelines.yml` | Triggers external ADO matrix | Coordinate GB300 job with owners |

### 3.5 Checks likely in GB300 conf (draft)

Presence / health:

- [ ] `check_hw_cpuinfo` — sockets/cores (docs: 128 vCPU; **measure** actual NHC cpuinfo shape on Grace)
- [ ] `check_hw_physmem` — ~864 GiB (**measure** exact kB/MB NHC sees)
- [ ] `check_hw_swap`
- [ ] `check_hw_ib <rate> <dev:port>` per IB device (**800** nominal; confirm `ibstat` names)
- [ ] `check_hw_eth` for real ifaces (lo, eth0, ib*, docker0 as applicable)
- [ ] `check_nvme_count 4`
- [ ] `check_hw_topology` (after topo file + code support)
- [ ] `check_gpu_count 4`
- [ ] `check_nvsmi_healthmon`
- [ ] `check_gpu_xid`
- [ ] `check_gpu_ecc`
- [ ] `check_gpu_clock_throttling`
- [ ] `check_nvlink_status`

Performance (calibrate on healthy node):

- [ ] `check_gpu_bw <h2d/d2h> <p2p>`
- [ ] `check_nccl_allreduce <bus_bw> <repeats> <msg> [topo]`
- [ ] `check_ib_bw_gdr <gbps>`
- [ ] `check_ib_link_flapping`

Explicitly **defer** unless inventory shows need:

- `check_nccl_allreduce_ib_loopback` (used on A100 ND; not on H100 conf)
- CPU STREAM (HB/HX family)

### 3.6 Known implementation risks

1. **aarch64 image** is blocking; conf-only PR will still fail default docker path on GB300.
2. **CUDA/driver/NCCL** versions must support Blackwell B300 on Grace.
3. **IB device naming / GPU↔IB index** may not match H100 (`mlx5_ib0..7`, cuda dev == ib index).
4. **Ethernet 160 Gbps** — runner only auto-adds AN at 40 or 100 via `mlx5_an0`.
5. **`boost_gpu_clock`** no-ops on unknown SKUs (may be OK; confirm BW tests still stable).
6. **Topo files** currently copied from `Azure/azhpc-images` at docker build — may not include GB300 yet; may need to vendor a generated topo.
7. **This workspace host is already `aarch64` / Ubuntu 24.04** — good for inventory and arm image bring-up; confirm it is actually `Standard_ND128isr_GB300_v6` before locking conf numbers.

---

## 4. Workstreams checklist

Use this as the execution tracker. Mark items `[x]` as done.

### WS0 — Inventory (on a real GB300 node)

- [ ] Confirm IMDS SKU string exactly (`Standard_ND128isr_GB300_v6` → conf `nd128isr_gb300_v6`)
- [ ] `uname -m`, distro, kernel, NVIDIA driver, CUDA user-mode
- [ ] `nvidia-smi -L` / `nvidia-smi topo -m` / nvlink status
- [ ] `ibstat -l`, `ibstatus`, rates, link layer, pkeys
- [ ] `ip -br link`, AN device present? name? rate?
- [ ] `lsblk`, `nvme list`, mount layout
- [ ] `lscpu`, `free -h`, `lstopo-no-graphics` (save artifact)
- [ ] Record healthy baseline numbers for NCCL allreduce, nvbandwidth, ib_write_bw GDR
- [ ] Save inventory under session artifacts or attach to PR (not necessarily committed secrets/hostnames)

**Exit criteria:** filled “Measured values” table in §6.

### WS1 — aarch64 NVIDIA container image (blocking)

- [ ] Choose image strategy:
  - **A (recommended):** new Dockerfile `dockerfile/azure-nvrt-nhc-aarch64.dockerfile` + tag `aznhc-nv:aarch64` or `aznhc-nv-arm`
  - **B:** multi-arch manifest on `aznhc-nv:latest` (harder ops)
- [ ] Select CUDA base image with Grace+Blackwell support (newer than 12.4 if required)
- [ ] OFED/MOFED **aarch64** user-space install path
- [ ] Build OpenMPI, nccl-tests, perftest (GDR + non-GDR), nvbandwidth, NHC for arm64
- [ ] nvbandwidth `CMAKE_CUDA_ARCHITECTURES` includes Blackwell (e.g. `90;100` — **confirm arch IDs** for B300)
- [ ] Install runtime deps: numactl, hwloc, bc, pciutils, libnccl, etc.
- [ ] Copy/topofiles path remains `${AZ_NHC_ROOT}/topofiles`
- [ ] Update `build_image.sh` (`cuda-arm` or auto `uname -m`)
- [ ] Update `pull-image-mcr.sh`
- [ ] Update `run-health-checks.sh` image selection:
  - NVIDIA + `aarch64` → arm image
  - NVIDIA + `x86_64` → existing `aznhc-nv`
- [ ] Smoke: container starts, `nvidia-smi`, `mpirun`, `all_reduce_perf`, `ib_write_bw --help`

**Exit criteria:** manual docker run on GB300 can execute nccl-tests and ib_write_bw against local GPUs/IB.

### WS2 — Conf file + runner wiring

- [ ] Add `conf/nd128isr_gb300_v6.conf` from H100/H200 template
- [ ] Set GPU count **4**, NVMe **4**, mem/cpu from inventory
- [ ] IB lines: one per real device at measured rate (nominal 800)
- [ ] Eth lines: only interfaces that exist
- [ ] Omit AN lines from conf if runner injects them; **extend runner** for 160G if AN device exists
- [ ] Add SKU to appropriate AN list **or** new `AN160=(nd128isr_gb300_v6)` / eth rate check design
- [ ] Placeholder perf thresholds initially low enough to pass, then tighten after calibration
- [ ] Default timeout: keep 500s or raise if NCCL/IB slower to stabilize

**Exit criteria:** runner finds conf by SKU without `-c`.

### WS3 — Custom test code updates

- [ ] `azure_hw_topology_check.nhc`: map `nd128isr_gb300_v6` (and maybe `*gb300*`) → new topo file
- [ ] Produce topo file (`ndv6-gb300-topo.xml` or azhpc-images name) and ensure docker build installs it
- [ ] `boost_gpu_clock` / `remove_clock_boost`: add GB300 lock frequencies if required for stable BW
- [ ] `check_ib_bw_gdr`: validate device discovery with 4 IB HCAs; fix CUDA index mapping if not 1:1
- [ ] `check_nccl_allreduce`: confirm 4-rank launch; set `NCCL_*` env if GB300 needs topo/graph
- [ ] `azure_nccl_allreduce.nhc`: currently sets `NCCL_TOPO_FILE=` empty in ENV string — decide if GB300 should pass topo like conf arg intends
- [ ] ECC / xid / throttling / nvsmi healthmon: run as-is; fix query flags if Blackwell nvidia-smi differs
- [ ] Link flap: confirm syslog path still mounted (`/var/log/syslog` or `messages`)

**Exit criteria:** each enabled check returns pass or a real hardware fault (not tool/SKU-unsupported).

### WS4 — Threshold calibration

On a known-good node, verbose run and capture:

- [ ] GPU H2D / D2H / P2P (nvbandwidth)
- [ ] NCCL allreduce bus BW (16G msg, 4 GPUs)
- [ ] IB GDR write BW per device
- [ ] Set conf floors ~5–10% below healthy minimum (match existing H100/H200 style)
- [ ] Re-run 3× to confirm no flaky fails
- [ ] Update README expected-values table column for GB300

**Exit criteria:** three consecutive clean default runs on healthy node.

### WS5 — Tests

- [ ] `test/unit-tests/nhc-test-common.sh`:
  - [ ] `get_sad_path_conf` → gb300
  - [ ] `get_topofile` → gb300 topo
  - [ ] `get_ib_type` → `gdr` for gb300
- [ ] Add `test/bad_test_confs/nd128isr_gb300_v6.conf` (inflated thresholds / wrong gpu count)
- [ ] `run_tests.sh` / docker image name respects arch
- [ ] Run agnostic tests still green on non-HPC if applicable
- [ ] On GB300: `./test/unit-tests/run_tests.sh` GPU + IB + hardware suites

### WS6 — Docs + developer UX

- [ ] README: supported SKU bullet + optional badge later
- [ ] README health-check table: GB300 expected column
- [ ] developer_guide: “Adding a new SKU” note for Arm images + AN160
- [ ] dockerfile/README.MD: pull/build arm instructions
- [ ] Comment in conf header: generation date, source inventory, known limits

### WS7 — PR / release coordination

- [ ] Logical commits (§5)
- [ ] PR description links PLAN.md + inventory summary
- [ ] Call out MCR publish dependency for arm image
- [ ] Ask owners about ADO matrix entry for GB300
- [ ] Do **not** mix GB200 changes into this PR

---

## 5. Suggested PR commit plan

Keep commits reviewable and bisectable. Adjust SHAs/messages as needed; order matters where noted.

### Commit 1 — Plan + inventory scaffold
**Message:** `docs: add GB300 v6 integration plan`

- Add `PLAN.md` (this file)
- Optional: `docs/gb300/` inventory template (only if we choose to keep inventory in-tree)

### Commit 2 — aarch64 NVIDIA Dockerfile + build/pull hooks
**Message:** `feat(docker): add aarch64 NVIDIA runtime image for Grace/Blackwell`

- New Dockerfile (or multi-stage arch-conditional)
- `build_image.sh` / `pull-image-mcr.sh` support
- dockerfile README notes  
**Depends on:** WS1 research (CUDA/OFED versions)

### Commit 3 — Runner selects arm image + AN160 handling
**Message:** `feat(runner): select aarch64 aznhc-nv and support GB300 networking rates`

- `run-health-checks.sh` arch-aware image
- Extend AN rate lists / logic for 160G if required  
**Depends on:** Commit 2 image name/tag decision

### Commit 4 — GB300 conf (presence checks first)
**Message:** `feat(conf): add nd128isr_gb300_v6 health conf (presence checks)`

- `conf/nd128isr_gb300_v6.conf` with hardware/GPU presence checks
- Perf checks omitted or very low placeholders commented with TODO  
**Can parallelize** with Commit 2 once SKU inventory exists

### Commit 5 — Topology + common SKU hooks
**Message:** `feat(checks): teach topology and clock boost about GB300`

- `azure_hw_topology_check.nhc` branch
- topo xml addition (repo and/or docker copy path)
- `boost_gpu_clock` GB300 case if needed

### Commit 6 — Perf check fixes for 4-GPU / 4-IB GB300
**Message:** `fix(checks): harden NCCL and IB GDR paths for GB300 device layout`

- IB port/index mapping fixes
- NCCL env/topo handling cleanup
- Any nvbandwidth invocation fixes

### Commit 7 — Enable and calibrate perf thresholds
**Message:** `feat(conf): enable calibrated GB300 GPU/IB/NCCL thresholds`

- Final numbers in conf
- README table update for expected BW

### Commit 8 — Unit/integration test coverage
**Message:** `test: add GB300 sad-path conf and SKU helpers`

- `nhc-test-common.sh`, bad conf, run_tests image selection

### Commit 9 — Docs polish
**Message:** `docs: document ND128isr_GB300_v6 support`

- README supported list, developer_guide, docker README final pass
- Mark PLAN.md WS items complete / add “Implementation status” section

> **Single-PR guidance:** Commits 2–8 are one PR if arm image can be built and validated in-branch.  
> If MCR publish is blocked, split:
> - **PR-A:** plan + conf + code hooks + tests (with `-c` and local image tag instructions)
> - **PR-B:** default MCR arm tag cutover once published  
> Prefer one PR if possible so default path works on merge.

---

## 6. Measured values (fill during WS0/WS4)

| Metric | Docs / nominal | Measured on node | Conf threshold |
|--------|----------------|------------------|----------------|
| IMDS vmSize | Standard_ND128isr_GB300_v6 | _TBD_ | n/a |
| arch | aarch64 | _TBD_ | n/a |
| CPU sockets / cores (NHC view) | 2 / 128 vCPU | _TBD_ | `check_hw_cpuinfo …` |
| Phys mem | 864 GiB | _TBD_ | `check_hw_physmem …` |
| GPU count / name | 4× B300 | _TBD_ | `check_gpu_count 4` |
| NVMe count | 4 | _TBD_ | `check_nvme_count 4` |
| IB devices (names) | ~4 × 800 Gbps | _TBD_ | `check_hw_ib …` |
| IB GDR BW (Gbps) | high / ~800 class | _TBD_ | `check_ib_bw_gdr …` |
| Eth / AN devices | 160 Gbps eth | _TBD_ | runner AN / eth checks |
| GPU H2D / D2H (GB/s) | _unknown_ | _TBD_ | `check_gpu_bw …` |
| GPU P2P (GB/s) | NVLink5 class | _TBD_ | `check_gpu_bw …` |
| NCCL allreduce bus BW (GB/s) | _unknown_ | _TBD_ | `check_nccl_allreduce …` |
| App clock lock (MHz) | _unknown_ | _TBD_ | `boost_gpu_clock` |
| Topo file source | n/a | _TBD_ | `check_hw_topology` |

---

## 7. Conf draft skeleton (not final thresholds)

```text
# conf/nd128isr_gb300_v6.conf
# Standard_ND128isr_GB300_v6 — values MUST be replaced from inventory/calibration

#######################################################################
### Hardware checks
#######################################################################
 * || check_hw_cpuinfo REPLACE_SOCKETS REPLACE_CORES REPLACE_CORES
 * || check_hw_physmem REPLACE_MEM REPLACE_MEM 5%
 * || check_hw_swap 0kB 0kB 3%
 # One line per IB HCA from ibstat; rate from device (nominal 800)
 * || check_hw_ib 800 mlx5_ib0:1
 # ...
 * || check_hw_eth lo
 * || check_hw_eth eth0
 # add ib* eth aliases only if present on node
 # NOTE: leave mlx5_an0 out if run-health-checks.sh injects AN
 * || check_hw_topology
 * || check_nvme_count 4

#######################################################################
### GPU checks
#######################################################################
 * || check_gpu_count 4
 * || check_nvsmi_healthmon
 * || check_gpu_xid
 * || check_gpu_bw REPLACE_H2D_D2H REPLACE_P2P
 * || check_gpu_ecc 20000000 10000
 * || check_gpu_clock_throttling
 * || check_nccl_allreduce REPLACE_NCCL_BW 1 16G $AZ_NHC_ROOT/topofiles/REPLACE_TOPO.xml
 * || check_nvlink_status

#######################################################################
### Additional IB checks
#######################################################################
 * || check_ib_bw_gdr REPLACE_IB_GDR_BW
 * || check_ib_link_flapping
```

---

## 8. Validation plan

### Local / node

```bash
# 1) Image
sudo dockerfile/build_image.sh cuda-arm   # final flag TBD
sudo docker run --rm --gpus all --privileged --net=host \
  mcr.microsoft.com/aznhc/aznhc-nv:aarch64 nvidia-smi

# 2) Presence-only conf iteration
sudo ./run-health-checks.sh -c conf/nd128isr_gb300_v6.conf -v -a -t 900

# 3) Default path (SKU detect)
sudo ./run-health-checks.sh -v -a -t 900

# 4) Tests on GB300
./test/unit-tests/run_tests.sh
```

### Acceptance matrix

| Case | Expected |
|------|----------|
| Healthy GB300 default run | exit 0 |
| Wrong gpu count conf | fail `check_gpu_count` |
| Inflated NCCL threshold | fail `check_nccl_allreduce` |
| x86 NVIDIA host still works | still pulls/runs x86 `aznhc-nv` |
| Unsupported SKU | existing message, no crash |

### Regression

- Spot-check that H100 conf path unchanged on x86 (no forced arm image).
- Ensure ROCm path untouched.

---

## 9. Open questions

- [ ] Exact IB device names and whether GDR `cuda_idx == ib enumeration order` holds on GB300.
- [ ] Is frontend network exposed as `mlx5_an0` IB link layer or pure Ethernet at 160G?
- [ ] Minimum CUDA / driver / NCCL versions certified on Azure GB300 images.
- [ ] Does `Azure/azhpc-images` already ship a GB300 topo file name we should reuse?
- [ ] MCR tag naming: `aznhc-nv:aarch64` vs `aznhc-nv-arm:latest` vs multi-arch `latest`.
- [ ] Should `Standard_ND128isrG5_GB300_v6` ship in the same PR as a second conf (likely identical checks)?
- [ ] ADO/CI capacity for Grace-Blackwell validation jobs.
- [ ] Clock lock frequencies for B300 app clocks (`nvidia-smi -lgc`).

---

## 10. Implementation status

| WS | Status | Notes |
|----|--------|-------|
| WS0 Inventory | Not started | Host arch here is aarch64; confirm SKU |
| WS1 Arm image | Not started | Blocking |
| WS2 Conf + runner | Not started | |
| WS3 Check code | Not started | |
| WS4 Calibration | Blocked on WS1–3 | |
| WS5 Tests | Not started | |
| WS6 Docs | In progress | This plan |
| WS7 PR | Branch created | `feature/gb300-v6-support` |

---

## 11. Quick reference — key code locations

```
run-health-checks.sh                 # host launcher, SKU, AN lists, image pick
dockerfile/azure-nvrt-nhc.dockerfile # current x86 NVIDIA image
dockerfile/aznhc-entrypoint.sh       # in-container nhc launch
conf/*.conf                          # per-SKU check lists
customTests/azure_common.nhc         # get_sku, boost clocks, IB helpers
customTests/azure_hw_topology_check.nhc
customTests/azure_nccl_allreduce.nhc # single-node, np=gpu_count
customTests/azure_ib_write_bw_gdr.nhc
test/unit-tests/nhc-test-common.sh
developer_guide.md                   # official "add a SKU" steps (x86-era)
```

---

## 12. Next immediate actions

1. Run WS0 inventory on this GB300 (or confirm SKU) and fill §6.
2. Prototype arm Dockerfile (WS1) — longest pole.
3. In parallel, draft `conf/nd128isr_gb300_v6.conf` presence checks (Commit 4).
4. Only after image smoke passes, enable perf checks and calibrate.
