# Plans (fork-local)

Working docs for upcoming work. Live on the **`plans`** branch in this fork only — not for upstream merge.

| Plan | PR order | Status |
|------|----------|--------|
| [arm-image.md](./arm-image.md) | **1st** — aarch64 NVIDIA container | Not started |
| [gb300.md](./gb300.md) | **2nd** — `Standard_ND128isr_GB300_v6` | Not started (blocked on arm image) |

## How to use

```bash
git checkout plans
# edit plans/*.md
git add plans && git commit && git push origin plans
```

Implementation branches off `main` (or stacked on the arm-image branch), not off `plans`.
