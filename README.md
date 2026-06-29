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
