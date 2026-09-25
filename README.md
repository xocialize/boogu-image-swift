# boogu-image-swift

Swift/MLX port of **Boogu-Image-0.1** (Apache-2.0) — a Qwen3-VL-8B-conditioned,
OmniGen2-lineage DiT (8 double-stream + 32 single-stream blocks) + FLUX.1
`AutoencoderKL` (16-channel) + a FlowMatchEuler static-v1 time-shift scheduler.
It ships as one MLXEngine `ModelPackage` exposing two surfaces:

- **`textToImage`** — Base (30-step CFG 3.5) and Turbo (4-step distilled, int8) tiers.
- **`imageEdit`** — the Edit variant: Qwen3-VL vision+text conditioning over the input
  image + a VAE ref-latent branch, structure-preserving edits at true CFG.

Reference = the parity-locked Python-MLX port
[`boogu-image-mlx`](https://github.com/xocialize/boogu-image-mlx). The Qwen3-VL
conditioner is reused from the
[`qwen3vl-mlx-swift`](https://github.com/xocialize/qwen3vl-mlx-swift) backbone
(`Qwen3VL.lastHiddenState`).

## Products

- **`BooguImage`** — the model core (DiT + VAE + scheduler + prompt encoder + generator).
- **`MLXBoogu`** — the `BooguImagePackage` MLXEngine wrapper (`textToImage` + `imageEdit`).

```swift
.package(url: "https://github.com/xocialize/boogu-image-swift", from: "0.1.0")
```

## Parity status

Locked against the Python-MLX port:

- DiT bit-exact (T2I + Edit), VAE decode/encode bit-exact, scheduler ~6e-8.
- Qwen3-VL text conditioning cos 1.0000; image-edit cos 0.998 fp32 / 0.967 bf16;
  preprocessing bit-exact.
- int4 DiT cos 0.996 (Turbo ships int8 — distilled few-step is quant-sensitive).
- Both surfaces render coherent, prompt-accurate / structure-preserving images end-to-end.

> **fp32-DiT note:** the bf16 DiT NaNs at large seqLen (≥~512²); set the package's
> `useFP32DiT` for production resolutions (same fp32-DiT lesson as the Wan/TI2V ports).

## Gates

CLI gates run in a real Metal context via `swift run` (the SPM test product's metallib
is unreliable). Parity goldens (`fixtures/goldens/`) are gitignored — regenerate them
from the Python-MLX oracle with `tools/dump_goldens.py`.

```
swift run BooguGate --nax-probe                              # mlx-swift NAX split-K GEMM health (no weights)
swift run BooguGate --s0-keys   <baseDir> fixtures           # structural key contract (no weights)
swift run BooguGate --s1-vae    <baseDir> fixtures/goldens   # VAE decode/encode parity
swift run BooguGate --s1-sched  <baseDir> fixtures/goldens   # scheduler parity
swift run BooguGate --s2-dit    <baseDir> fixtures/goldens   # DiT t2i + edit parity
swift run BooguGate --s6-quant  <cfgDir> <quantDir> <goldenDir>   # quantized DiT cosine
swift run BooguGate --e2e-golden <baseDir> <goldenDir> out.png [steps] [size] [fp32]
swift run BooguGate --e2e       <baseDir> <qwenDir> "<prompt>" out.png [steps] [guidance] [size]
swift run BooguGate --e2e-edit  <editDir> <qwenDir> in.png "<instruction>" out.png [steps] [size] [dtype]
```

Weights: `mlx-community/Boogu-Image-0.1-{Base,Turbo,Edit}` (transformer / vae / scheduler)
+ the stock `mlx-community/Qwen3-VL-8B-Instruct` conditioner. This repo contains no weights.

### mlx-swift NAX split-K workaround (temporary)

mlx-swift ≤0.31.6 JIT-compiles `steel_gemm_splitk_axpby_nax` with the wrong dtype
template parameter (ml-explore/mlx#3797, fixed upstream by mlx#3810 on 2026-07-07 —
no mlx-swift release ships it yet). The only Boogu DiT GEMM in the dispatch window is
the FFN down-projection (K=13568, N=3360), corrupting at M ∈ [1249, 4522] tokens; it
is row-chunked at ≤896 rows in `LuminaFeedForward.downProjected` (mathematically
exact), so the DiT default is bf16. Env switches: `BOOGU_FP32` forces the fp32 DiT,
`BOOGU_NO_CHUNK` disables the chunk (only for validating a fixed mlx-swift).
If a bf16 render ever looks suspect (in-app or CLI), set `BOOGU_FP32=1` in the run
environment for an instant A/B against the fp32 path — same seed, same request; if
the artifact survives fp32 it is not this bug.
**On every mlx-swift bump:** run `swift run BooguGate --nax-probe`; on PASS delete
`downProjected` (and its siblings in mage-flow-swift + qwen3vl-mlx-swift).

### VAE: mlx's lossy Winograd conv2d window (2026-09-24)

mlx's Metal `conv2d` takes a Winograd F(6×6,3×3) path when the conv is 3×3, stride 1, dilation 1,
groups 1, C % 32 == 0, O % 32 == 0, C + O ≥ 256 and N·H·W ≥ 4096. On M5 that path loses about
6.4e-3 relL2 per conv in fp32, because its inner GEMM runs TF32, and about 5.8e-2 in bf16.

The FLUX.1 AE hits it in nearly every 3×3 conv: 31 decoder convs at 1024² and 21 encoder convs per
edit reference. `--s1-vae` runs on the CPU lane, so this was never visible there.

Every stride-1 3×3 conv is now a `WinogradFreeConv2d` with a route (`BooguVAEConvRoute`). Shapes
outside the window take plain conv2d.

**Defaults: encoder `.conv3d`, decoder `.winograd`.** The choice follows the fleet audit decision
of 2026-09-24:

- Route where the loss is material. The encoder loss is: 2.2e-2 in the edit-reference latent.
- Keep mlx's path where the fp32 loss is below 8-bit visibility and the route is expensive. That is
  the decoder.

Measurements: DIV2K photo, against the CPU lane. Identical to z-image-swift, which uses the same
FLUX.1-dev AE.

| | Raw conv2d (Winograd) | conv3d route |
|---|---|---|
| Encode latent, 512² / 1024² | 1.7e-2 / 2.2e-2 | 6.5e-5 / 3.7e-4 (see note) |
| Decode, fp32, 1024² | 1.3e-3 · 70.0 dB · max 1.35e-2 | 2.0e-5 · 106.5 dB |
| Decode, bf16, 1024² | 1.2e-2 · 50.8 dB | 3.8e-3 · 60.8 dB |
| Time at 1024², decode / encode (fp32) | ~597 ms / ~330 ms | +471 ms / +212 ms |

Note on the 1024² encode figure: most of the 3.7e-4 is error in the CPU reference. MLX's CPU
GroupNorm drifts with group size, to 8e-5 per full-resolution norm at 1024² against float64.

Controls:

- Parity lanes: set `vae.decoderConvRoute = .conv3d`, or run with `MLX_ENABLE_TF32=0`, which makes
  fp32 Winograd exact at full speed.
- Environment override: `BOOGU_VAE_CONV_ROUTE=winograd|conv3d|fp32Winograd`.

Tests:

- `swift test --filter WinogradProbeTests` is weight-free. It is the removal signal on an mlx-swift
  bump.
- `BOOGU_PARITY=1 BOOGU_SNAPSHOT=<Boogu-Image-0.1-Edit> swift test -c release -Xswiftc
  -enable-testing --filter VAEGPULaneTests` compares the GPU and CPU lanes.
