"""Dump Python-MLX goldens for the Swift parity gates.

Run with the Python port's venv from ~/Development/mlxengine-image:
    .venv/bin/python boogu-image-swift/tools/dump_goldens.py

Saves fp32 input+output fixtures as .safetensors (loadable directly by MLX-Swift's
MLX.loadArrays) under boogu-image-swift/fixtures/goldens/. All forwards run on the CPU
stream in fp32 for clean Python-MLX <-> Swift-MLX parity.
"""
import os, sys, json
import numpy as np
import mlx.core as mx

ROOT = os.path.expanduser("~/Development/mlxengine-image")
sys.path.insert(0, os.path.join(ROOT, "boogu-image-mlx"))
from boogu_image_mlx.models.transformer import BooguImageTransformer2DModel
from boogu_image_mlx.models.vae import AutoencoderKL
from boogu_image_mlx.scheduler import FlowMatchEulerDiscreteScheduler
from boogu_image_mlx.utils.weights import (read_safetensors_dir, read_safetensors_np,
                                           load_named_into_mlx, load_diffusers_into_mlx)

BASE = os.path.join(ROOT, "weights", "Boogu-Image-0.1-Base")
OUT = os.path.join(ROOT, "boogu-image-swift", "fixtures", "goldens")
os.makedirs(OUT, exist_ok=True)
mx.set_default_device(mx.cpu)

def save(name, d):
    mx.save_safetensors(os.path.join(OUT, name), {k: v.astype(mx.float32) for k, v in d.items()})
    print("wrote", name, {k: tuple(v.shape) for k, v in d.items()})

# ---- VAE (fp32, CPU) -------------------------------------------------------
vcfg = json.load(open(os.path.join(BASE, "vae", "config.json")))
vae = AutoencoderKL.from_config(vcfg)
load_diffusers_into_mlx(vae, read_safetensors_np(
    os.path.join(BASE, "vae", "diffusion_pytorch_model.safetensors")))

rng = np.random.default_rng(0)
z = mx.array(rng.standard_normal((1, 16, 32, 32)).astype(np.float32))
dec = vae.decode(z); mx.eval(dec)
img = mx.array(rng.standard_normal((1, 3, 256, 256)).astype(np.float32))
mom = vae.encode_moments(img); mx.eval(mom)
save("vae_golden.safetensors", {"z": z, "decode_out": dec, "img": img, "encode_out": mom})

# ---- Scheduler (static v1 seq_len, + dynamic v1 token path) -----------------
scfg = json.load(open(os.path.join(BASE, "scheduler", "scheduler_config.json")))
sch = FlowMatchEulerDiscreteScheduler.from_config(scfg)
ts_static = sch.set_timesteps(28).copy()
ts_dyn = sch.set_timesteps(28, num_tokens=32 * 32).copy()  # ignored unless dynamic
save("scheduler_golden.safetensors", {
    "ts_static_28": mx.array(ts_static),
    "ts_static_4": mx.array(sch.set_timesteps(4)),
})

# ---- DiT (fp32, CPU): Base T2I (no ref) and Edit (ref) ----------------------
tcfg = json.load(open(os.path.join(BASE, "transformer", "config.json")))
dit = BooguImageTransformer2DModel.from_config(tcfg)
load_named_into_mlx(dit, read_safetensors_dir(
    os.path.join(BASE, "transformer"), dtype=mx.float32))

H = W = 256
hl, wl = H // 8, W // 8           # 32x32 latent
latent = mx.array(rng.standard_normal((1, 16, hl, wl)).astype(np.float32))
timestep = mx.array(np.array([0.7], dtype=np.float32))
instr = mx.array(rng.standard_normal((1, 48, 4096)).astype(np.float32) * 0.1)

out_t2i = dit(latent, timestep, instr); mx.eval(out_t2i)
ref = mx.array(rng.standard_normal((1, 16, hl, wl)).astype(np.float32))
out_edit = dit(latent, timestep, instr, ref_latent=ref); mx.eval(out_edit)
save("dit_golden.safetensors", {
    "latent": latent, "timestep": timestep, "instruction": instr,
    "out_t2i": out_t2i, "ref_latent": ref, "out_edit": out_edit,
})
print("done")
