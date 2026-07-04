# Efficiency Adoption Brief — `boogu-image-swift` (Boogu-Image-0.1, `textToImage`)

> **For a session-specific agent.** Adopt engine 1.14 efficiency (engine 0.17.0+). Load the
> `mlx-swift-integration` skill; read references/package-efficiency.md (four levers + ALL of **"Measurement
> findings"**: in-app phys vs smoke, post-load floor, encoder-evict) + references/memory-harness.md. Closest
> template: the **Lens** / **Qwen-Image-Edit** encoder-evict adoptions. **This is the in-house PRODUCT model —
> parity-locked bit-exact. Preserve parity.** Audited 2026-06-30.

## Package at a glance
- Wrapper `MLXBoogu` (`BooguImagePackage: ModelPackage`) over core `BooguImage`. Capability **`textToImage`**
  (also an Edit mode w/ ref-latent). Engine pinned `from: "0.3.0"`; depends on `qwen3vl-mlx-swift` 0.1.1.
- **Components:** `encoder: BooguPromptEncoder?` = the **Qwen3-VL-8B conditioner** (produces the
  `last_hidden_state` conditioning) + `generator: BooguImageGenerator?` = OmniGen2-lineage **DiT** + **FLUX VAE**.
- **Footprints today (FLAT residentBytes only, NO transient):** int4 **24** · int8 **29** · bf16 **36 GB**,
  declared as ESTIMATES. The comment says it straight: "Resident = DiT + **Qwen3-VL-8B (bf16)** + FLUX VAE …
  int4/int8 quantize only the DiT; **the conditioner stays bf16**." → the ~16 GB Qwen3-VL conditioner is
  **baked into residency on every quant**, held through the whole denoise.
- **Already good — don't regress:** the denoise loop already does per-step `MLX.GPU.clearCache()`
  (`Pipeline.swift:69`). The absolute `snapshotPath`/`qwenPath` already resolves cleanly under the store
  (Xcode-confirmed) — don't break it.
- **Known in-app measure (PRE-evict):** MLXEngineImage ran Boogu Base bf16 → phys **floor 67.4 / peak 70.5 GB**
  (engine charged the 36 GB estimate; real phys is ~1.85× higher — the gap). This is the baseline to beat.

## Audit vs. the four levers
| Lever | State | Finding | Priority |
|---|---|---|---|
| Engine dep | 🟡 | from 0.3.0 → 0.17.0 | **P0** |
| 2. **Encoder-evict** | ❌ | **the headline.** Qwen3-VL-8B encodes the prompt(s) ONCE upfront (`encodeText` → `last_hidden_state`), then the DiT denoise loop runs WITHOUT it — but it's held resident the whole time. ~16 GB bf16 evictable on EVERY quant. | **P2 (headline, huge)** |
| 1. Split footprint | ❌ | flat 24/29/36 GB; transient baked in | **P1** |
| 3. mmap/lazy | 🟡 verify | confirm DiT/VAE/conditioner load lazily (no eager full copy) | note |
| 4. BudgetAware | 🟡 maybe | DiT quant is the lever; conditioner fixed bf16 — defer unless trivial | defer |

## Plan (mirror Lens; PRESERVE BIT-EXACT PARITY)
- **P0:** `swift package update` → 0.17.0 (also re-resolve qwen3vl if needed); build + fix drift.
- **P2 (HEADLINE — encoder-evict):** refactor the pipeline so the Qwen3-VL conditioner is **loaded → used to
  encode pos/neg conditioning → `eval`'d/retained → EVICTED** (`encoder = nil` + `Memory.clearCache()`) BEFORE
  the DiT denoise loop. The conditioning tensors (`last_hidden_state`) are what the DiT consumes — **`eval`
  them before dropping the encoder** so parity is bit-identical (you change only WHEN the encoder frees, never
  the math). Swift 6 `#isolation` gotcha if the staged path goes async (`isolated (any Actor)? = #isolation` —
  recurred on Lens/ERNIE/Qwen-Image-Edit). This moves ~16 GB resident→transient on every quant.
- **P1:** split per quant. `residentBytes` (POST-evict) = DiT(quant) + FLUX VAE (resident through denoise);
  `peakActivationBytes` = `max(`Qwen3-VL encode transient`, `DiT denoise working set`)`. `QuantConfigured`
  (int4/int8/bf16; quant affects the DiT only).
- **`unload()` must `MLX.Memory.clearCache()`** (verify; add if missing).
- **Parity gate:** Boogu has `BooguGate` (+ the qwen3vl `Qwen3VLGate`) parity harnesses — run the relevant
  component gate after the refactor to confirm DiT/VAE/encode math is unchanged.

## Measurement — IMPORTANT (heavy; re-baseline post-evict)
The known 67.4/70.5 GB is the PRE-evict baseline. Post-evict, the resident floor should drop ~16 GB. Declare
`residentBytes` from the measured post-evict weight floor (DiT+VAE; solid), and `peakActivationBytes`
**FLAGGED** pending an in-app phys RE-BASELINE in MLXEngineImage (the autorun re-measures Boogu post-evict —
the existing declaration is way off, so flag clearly). Don't run a full 70 GB pipeline headless if it risks
the watchdog; the parity gate + weight-floor measure + flag is sufficient for the subagent.

## Definition of done
- [ ] engine 0.17.0; `QuantConfigured`; **P2 encoder-evict** (Qwen3-VL freed before denoise, parity-preserved);
      P1 split per quant; `unload()` clearCache; per-step clearCache + absolute-path resolution NOT regressed.
- [ ] residentBytes = post-evict DiT+VAE floor; peakActivationBytes FLAGGED for in-app phys re-baseline.
- [ ] Parity gate green (DiT/VAE/encode unchanged); split recorded.
- [ ] Registry: boogu row Eff ⬜→✅ (note "encoder-evict; activation phys re-baseline pending"), Eng→0.17.0.

## Report back
the encoder-evict effect (resident drop per quant from moving Qwen3-VL out), flat→split, parity-gate result,
drift since 0.3.0, effort, commit SHAs. STAY IN SCOPE — four-lever adoption + this brief + registry row only;
**preserve bit-exact parity**; no testing-app/xcodeproj changes; stop-and-report if the refactor needs to
touch the DiT/VAE math or anything bigger.
