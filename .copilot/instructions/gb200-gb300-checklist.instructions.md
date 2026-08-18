# GB200/GB300 Support — Live Cross-Repo Checklist

**Location note:** lives at `.copilot/instructions/` on the fork-local
`plans` branch of **both** `azhpc-images` and `azurehpc-health-checks`
(identical copy in each). Update this file's checkboxes/notes as work
progresses in *either* repo, and sync the copy in the other repo's `plans`
branch so both stay accurate — the two tracks depend on each other (see
"Cross-repo dependency" below). Last updated: 2026-08-18.

## Cross-repo dependency
`azurehpc-health-checks` conf/tests do **not** strictly block on
`azhpc-images` landing — GB300 no longer needs a generated topofile, so the
health-checks conf doesn't need anything from the image repo to work
standalone. The one remaining link: if `azhpc-images` ends up setting
`NCCL_NET_GDR_LEVEL` for GB300 as a system default, that could change what
env vars `azurehpc-health-checks`'s NCCL checks need to set/override
themselves — resolve the GDR-level ambiguity (see context file) before
finalizing both `ndv6-gb300.sh` **and** any GDR-level env vars in
`customTests/azure_nccl_allreduce*.nhc`.

## azhpc-images (branch `michaelbeebe/gb-family-topology`, pushed to fork)
- [X] Split `setup_sku_customizations.sh`: `gb-family` -> `gb300-family`/`gb200-family`
- [X] Confirm real SKU strings: `standard_nd128isr_gb300_v6`, `standard_nd128isr_ndr_gb200_v6`
- [X] Rename `ndv6.sh` -> `ndv6-gb300.sh`; add cleanup entry to `remove_sku_customizations.sh`
- [X] Create `ndv6-gb200.sh` (no-op placeholder, preserves current behavior)
- [X] Rewrite `ndv6-gb300.sh` to drop all topofile logic (no topofile, no
      `/etc/nccl.conf`, no GDR-level env var) — done in working tree, **still
      NOT committed/pushed** (last pushed commit `dc1917a` is the stale
      topofile-based version)
- [ ] Commit + push the no-topofile `ndv6-gb300.sh` rewrite + `topology/ndv6-topo.xml` deletion
- [ ] Resolve GDR-level ambiguity (see context file), then finalize
      `ndv6-gb200.sh` (likely needs `NCCL_NET_GDR_LEVEL=PHB`, unconfirmed)
- [ ] Mark `plans/gb-family-topology.md` superseded/rewritten (premise is now wrong)
- [ ] (Optional) Maintainer check-in before opening PR
- [ ] Open PR to `Azure/azhpc-images`

## azurehpc-health-checks (branch `feature/gb300-v6-support`, pushed)
- [X] `conf/nd128isr_gb300_v6.conf` created (aarch64 GB300 hardware/GPU/IB checks)
- [X] Fix `check_ib_bw_gdr`: 380 -> 640 Gb/s (was a stale H100 copy-paste;
      ConnectX-8 is 2x ConnectX-7 bandwidth) — committed `8c9a589`, pushed
- [X] Add `check_nccl_allreduce_ib_loopback 80.0 1 16G` (real IB-isolated NCCL
      test via `NCCL_SHM_DISABLE`/`NCCL_P2P_DISABLE` — was completely missing
      for GB300 before; the existing `check_nccl_allreduce` call there
      measures NVLink-dominated bandwidth, not IB) — committed `8c9a589`, pushed
- [ ] Resolve/remove stray "TODO: this comment cannot be here" note above `check_nccl_allreduce`
- [ ] Test conf against real GB300 node(s)
- [ ] Merge/land aarch64 support branch (dependency, currently only in this branch's history)
- [ ] Open/update PR for GB300 conf support

## IB/NCCL baseline analysis (supporting work, local-only `~/ib-baselines-csv/`)
- [X] Catalogue existing NHC conf baselines across all SKUs -> `ib-baselines.csv`
- [X] Capture manager-authoritative recommended baseline table (6 tests x
      A100/H100+H200+GB200/GB300) -> `ib-perf-baselines-recommended.csv`
- [X] Compare `manifold_scripts` (`run_ibw_write_bw.py`, `run_nccl_tests.py`)
      baselines against recommended table -> `manifold-vs-recommended.csv`
- [X] Confirmed manifold's GB300 IB GDR (`ib_write_bw`) and NCCL IB baselines
      are under-scaled for ConnectX-8 (should be ~640 Gb/s / ~80 GB/s,
      currently ~400/50) — confirmed genuine GDR (real `--use_cuda`), and
      unidirectional (no `-b`) in both scripts, matching the recommended
      table's methodology exactly
- [ ] Get `manifold_scripts` GB300 baselines actually fixed (repo/owner TBD — not either of these two repos)
- [ ] Deprecate/reconcile older `ib-baseline-vs-theoretical.csv` (known
      labeling bug + contested "theoretical peak" methodology for NCCL busbw)

## Next up
- Awaiting user direction on what to tackle next.
