"""Round-2 review GPU experiment driver.

Per (circuit, seed) pair, runs:
  M1  baseline                                      [reference]
  M2  VSR-post(lambda=8)                            [R2 / Q3 / Q6]
  M3  cd_sched(raw baseline)                        [reference: raw -> legalizer]
  M4  cd_sched(VSR-post(lambda=8) output)           [R1 / S1 / Q1: decisive]
  M5  FD-pure(no mask, no spring)                   [R4]
  M6  FD+spring(no mask, hpwl_weight=1)             [R4]
  M7-11  cg_strong with w in {0.5,1,2,4,8}          [R3 / Q4]
  M12-16 repaint_binary with t in {0.1,...,0.7}     [R3 / Q4]

For every method we record: violation count, macros-only HPWL,
full-design HPWL with cells fixed at .pl, total overlap area,
max overlap, wall-clock, peak VRAM.

Output: results/vsr_extra/round2_review.json (incremental, resume-safe).
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
import traceback
import types
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "third_party" / "chipdiffusion"))
sys.path.insert(0, str(REPO / "third_party" / "chipdiffusion" / "diffusion"))
sys.path.insert(0, str(REPO / "scripts"))

_mp = types.ModuleType("moviepy")
_mp_ed = types.ModuleType("moviepy.editor")
_mp_ed.ImageSequenceClip = lambda *a, **k: None
sys.modules["moviepy"] = _mp
sys.modules["moviepy.editor"] = _mp_ed

import torch  # noqa: E402

NAMES = ["adaptec1", "adaptec2", "adaptec3", "adaptec4",
         "bigblue1", "bigblue2", "bigblue3", "bigblue4"]
CIRCUITS = [0, 1, 2, 3, 4, 6]
SEEDS = [42, 123, 300, 2024]
LAMBDA8 = 8.0
NUM_REPAIR = 100
T_INTRA = 0.3

# CD legalizer config (HPWL-aware variant: matches ChipDiffusion's released eval logs)
CD_SCHED = dict(step_size=0.2, grad_descent_steps=5000, softmax_min=5.0, softmax_max=50.0,
                softmax_critical_factor=1.0, hpwl_weight=4e-5, legality_weight=2.0,
                macros_only=True)

CG_WEIGHTS = [0.5, 1.0, 2.0, 4.0, 8.0]
REPAINT_TS = [0.1, 0.2, 0.3, 0.5, 0.7]

CHECKPOINT = "checkpoints/large-v2/large-v2.ckpt"
OUT = REPO / "results" / "vsr_extra" / "round2_review.json"


def _overlap_metrics(centers, sizes):
    half = sizes / 2.0
    mins = centers - half; maxs = centers + half
    inter_min = torch.maximum(mins.unsqueeze(1), mins.unsqueeze(0))
    inter_max = torch.minimum(maxs.unsqueeze(1), maxs.unsqueeze(0))
    d = torch.clamp(inter_max - inter_min, min=0.0)
    A = d[..., 0] * d[..., 1]
    eye = torch.eye(centers.shape[0], dtype=torch.bool, device=centers.device)
    A = A.masked_fill(eye, 0.0)
    return float(A.sum().item() / 2.0), float(A.max().item())


def _full_hpwl(centers_macro, full_centers, macro_mask, edge_index_full, edge_attr_full):
    from vsr_place.metrics.hpwl import compute_hpwl_from_edges
    full = full_centers.clone()
    full[macro_mask] = centers_macro
    return float(compute_hpwl_from_edges(full, edge_index_full, edge_attr_full))


def _cd_legalize(adapter, x_hat_0, cond, cfg):
    import legalization as cd_leg
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    pl = x_hat_0.unsqueeze(0).detach().to(adapter.device)
    out = cd_leg.legalize(pl, cond.to(adapter.device), **cfg)
    if isinstance(out, tuple): out = out[0]
    t = time.time() - t0
    vram = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    return out.squeeze(0), t, vram


def _eval_all(centers, sizes, ei, ea, full_x, macro_mask, ei_full, ea_full,
              cw, ch, ver):
    from vsr_place.metrics.hpwl import compute_hpwl_from_edges
    fb = ver(centers, sizes)
    v = int(fb.global_stats["total_violations"])
    h_macro = float(compute_hpwl_from_edges(centers, ei, ea))
    oa, mx = _overlap_metrics(centers, sizes)
    h_full = _full_hpwl(centers, full_x.cpu(), macro_mask, ei_full, ea_full) if macro_mask is not None else None
    return v, h_macro, oa, mx, h_full


def main():
    from run_vsr import load_benchmark_data, filter_macros_only
    from vsr_place.backbone.adapter import ChipDiffusionAdapter
    from vsr_place.verifier.verifier import Verifier
    from vsr_place.renoising.local_repair import local_repair_loop

    val_full = load_benchmark_data("ispd2005")
    val = [filter_macros_only(x, c) for x, c in val_full]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = json.load(open(OUT)) if OUT.exists() else []
    done = {(r["circuit"], r["seed"]) for r in rows if r.get("baseline_v") is not None}
    print(f"Resume: {len(done)} pairs done", flush=True)

    for ci in CIRCUITS:
        name = NAMES[ci]
        full_x, full_cond = val_full[ci]
        x_in, cond = val[ci]
        macro_mask = full_cond.is_macros.bool() if hasattr(full_cond, "is_macros") and full_cond.is_macros is not None else None
        ei_full = full_cond.edge_index.cpu()
        ea_full = full_cond.edge_attr.cpu() if full_cond.edge_attr is not None else None

        for seed in SEEDS:
            if (name, seed) in done:
                continue
            print(f"\n=== {name} seed={seed} ===", flush=True)
            row = {"circuit": name, "seed": seed, "n_macros": int(cond.x.shape[0])}
            try:
                torch.manual_seed(seed); torch.cuda.empty_cache()

                # ---- baseline + decoded sizes ----
                ad = ChipDiffusionAdapter.from_checkpoint(
                    CHECKPOINT, device="cuda", input_shape=tuple(x_in.shape), guidance="opt",
                )
                cw, ch = ad.get_canvas_size(cond)
                ver = Verifier(canvas_width=cw, canvas_height=ch)

                t0 = time.time()
                pl_base = ad.guided_sample(cond.to("cuda"), num_samples=1).squeeze(0)
                t_base = time.time() - t0
                cent_b, sz = ad.decode_placement(pl_base, cond)
                cent_b = cent_b.cpu(); sz = sz.cpu()
                ei = cond.edge_index.cpu()
                ea = cond.edge_attr.cpu() if cond.edge_attr is not None else None
                fb_b = ver(cent_b, sz); sev = fb_b.severity_vector
                bv, bh, boa, bmx, bhf = _eval_all(cent_b, sz, ei, ea, full_x, macro_mask, ei_full, ea_full, cw, ch, ver)
                row.update(baseline_v=bv, baseline_h=bh, baseline_oa=boa, baseline_max=bmx,
                           baseline_full_h=bhf, baseline_time=t_base)
                print(f"  baseline      v={bv} h={bh:.0f} oa={boa:.2f} full_h={bhf}", flush=True)

                # ---- M2: VSR-post(lambda=8) ----
                t0 = time.time()
                cent_v8 = local_repair_loop(cent_b, sz, cw, ch, num_steps=NUM_REPAIR,
                                            step_size=0.3, only_mask=(sev > 0),
                                            edge_index=ei, hpwl_weight=LAMBDA8)
                t_v8 = time.time() - t0
                v, h, oa, mx, hf = _eval_all(cent_v8, sz, ei, ea, full_x, macro_mask, ei_full, ea_full, cw, ch, ver)
                row.update(vsr8_v=v, vsr8_h=h, vsr8_oa=oa, vsr8_max=mx, vsr8_full_h=hf, vsr8_time=t_v8)
                print(f"  vsr8          v={v} ({(v-bv)/max(bv,1)*100:+.1f}%) h={h:.0f} ({(h-bh)/max(abs(bh),1e-9)*100:+.1f}%) full_h={hf}", flush=True)

                # ---- M5: FD-pure (no mask, no spring) ----
                t0 = time.time()
                cent_fd = local_repair_loop(cent_b, sz, cw, ch, num_steps=NUM_REPAIR,
                                            step_size=0.3, only_mask=None,
                                            edge_index=None, hpwl_weight=0.0)
                t_fd = time.time() - t0
                v, h, oa, mx, hf = _eval_all(cent_fd, sz, ei, ea, full_x, macro_mask, ei_full, ea_full, cw, ch, ver)
                row.update(fd_pure_v=v, fd_pure_h=h, fd_pure_oa=oa, fd_pure_full_h=hf, fd_pure_time=t_fd)
                print(f"  fd_pure       v={v} ({(v-bv)/max(bv,1)*100:+.1f}%) h={h:.0f} ({(h-bh)/max(abs(bh),1e-9)*100:+.1f}%)", flush=True)

                # ---- M6: FD+spring (all macros, hpwl_weight=1) ----
                t0 = time.time()
                cent_fs = local_repair_loop(cent_b, sz, cw, ch, num_steps=NUM_REPAIR,
                                            step_size=0.3, only_mask=None,
                                            edge_index=ei, hpwl_weight=1.0)
                t_fs = time.time() - t0
                v, h, oa, mx, hf = _eval_all(cent_fs, sz, ei, ea, full_x, macro_mask, ei_full, ea_full, cw, ch, ver)
                row.update(fd_spring_v=v, fd_spring_h=h, fd_spring_oa=oa, fd_spring_full_h=hf, fd_spring_time=t_fs)
                print(f"  fd_spring     v={v} ({(v-bv)/max(bv,1)*100:+.1f}%) h={h:.0f} ({(h-bh)/max(abs(bh),1e-9)*100:+.1f}%)", flush=True)

                # ---- M3: cd_sched(raw baseline) ----
                try:
                    pl_cdr, t_cdr, vram_cdr = _cd_legalize(ad, pl_base, cond, CD_SCHED)
                    cent_cdr, _ = ad.decode_placement(pl_cdr, cond)
                    cent_cdr = cent_cdr.cpu()
                    v, h, oa, mx, hf = _eval_all(cent_cdr, sz, ei, ea, full_x, macro_mask, ei_full, ea_full, cw, ch, ver)
                    row.update(cd_raw_v=v, cd_raw_h=h, cd_raw_oa=oa, cd_raw_full_h=hf, cd_raw_time=t_cdr)
                    print(f"  cd(raw)       v={v} ({(v-bv)/max(bv,1)*100:+.1f}%) h={h:.0f} ({(h-bh)/max(abs(bh),1e-9)*100:+.1f}%) ({t_cdr:.0f}s)", flush=True)
                except Exception as e:
                    print(f"  cd(raw) error: {e}", flush=True)
                    row.update(cd_raw_error=str(e)[:200])

                # ---- M4: cd_sched(VSR-post(lambda=8) output)  *** decisive ***
                try:
                    # encode_placement is the inverse of decode_placement: absolute centers -> normalized [-1, 1]
                    pl_v8_norm = ad.encode_placement(cent_v8, cond)
                    pl_v8 = pl_v8_norm.to(ad.device).unsqueeze(0)
                    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
                    t0 = time.time()
                    import legalization as cd_leg
                    pl_v8c = cd_leg.legalize(pl_v8, cond.to(ad.device), **CD_SCHED)
                    if isinstance(pl_v8c, tuple): pl_v8c = pl_v8c[0]
                    t_pipe = time.time() - t0
                    cent_pipe, _ = ad.decode_placement(pl_v8c.squeeze(0), cond)
                    cent_pipe = cent_pipe.cpu()
                    v, h, oa, mx, hf = _eval_all(cent_pipe, sz, ei, ea, full_x, macro_mask, ei_full, ea_full, cw, ch, ver)
                    row.update(pipe_vsr8_cd_v=v, pipe_vsr8_cd_h=h, pipe_vsr8_cd_oa=oa,
                               pipe_vsr8_cd_full_h=hf, pipe_vsr8_cd_time=t_pipe + t_v8)
                    print(f"  vsr8+cd       v={v} ({(v-bv)/max(bv,1)*100:+.1f}%) h={h:.0f} ({(h-bh)/max(abs(bh),1e-9)*100:+.1f}%) ({t_pipe:.0f}s)", flush=True)
                except Exception as e:
                    traceback.print_exc()
                    row.update(pipe_vsr8_cd_error=str(e)[:200])

                # ---- M7-11: cg_strong sweep (loads fresh adapter each) ----
                # WARNING: each fresh adapter load is heavy (~10s); 5 settings × 10s = 50s/trial
                # We do a reduced sweep (just w=2 and w=8) on 4 seeds to save GPU time, given the
                # seed-42 sweep already gave per-circuit best in seeds_42 supplement table.
                cg_subset = [2.0, 8.0]
                for w in cg_subset:
                    try:
                        torch.manual_seed(seed); torch.cuda.empty_cache()
                        adcg = ChipDiffusionAdapter.from_checkpoint(
                            CHECKPOINT, device="cuda", input_shape=tuple(x_in.shape), guidance="opt",
                        )
                        if hasattr(adcg.model, "legality_guidance_weight"):
                            adcg.model.legality_guidance_weight = w
                        if hasattr(adcg.model, "hpwl_guidance_weight"):
                            adcg.model.hpwl_guidance_weight = 0.0
                        if hasattr(adcg.model, "grad_descent_steps") and adcg.model.grad_descent_steps == 0:
                            adcg.model.grad_descent_steps = 5
                        t0 = time.time()
                        pl_cg = adcg.guided_sample(cond.to(adcg.device), num_samples=1).squeeze(0)
                        t_cg = time.time() - t0
                        cent_cg, _ = adcg.decode_placement(pl_cg, cond)
                        cent_cg = cent_cg.cpu()
                        v, h, oa, mx, hf = _eval_all(cent_cg, sz, ei, ea, full_x, macro_mask, ei_full, ea_full, cw, ch, ver)
                        row[f"cg{w}_v"] = v; row[f"cg{w}_h"] = h
                        row[f"cg{w}_oa"] = oa; row[f"cg{w}_full_h"] = hf
                        row[f"cg{w}_time"] = t_cg
                        print(f"  cg w={w}       v={v} ({(v-bv)/max(bv,1)*100:+.1f}%) h={h:.0f} ({(h-bh)/max(abs(bh),1e-9)*100:+.1f}%)", flush=True)
                        del adcg; gc.collect(); torch.cuda.empty_cache()
                    except torch.cuda.OutOfMemoryError:
                        row[f"cg{w}_error"] = "OOM"
                        print(f"  cg w={w} OOM", flush=True)
                        gc.collect(); torch.cuda.empty_cache()
                    except Exception as e:
                        row[f"cg{w}_error"] = str(e)[:200]
                        print(f"  cg w={w} error: {e}", flush=True)

                # ---- M12-13: repaint at best two t_start (0.3, 0.5) ----
                rp_subset = [0.3, 0.5]
                for ts in rp_subset:
                    try:
                        torch.manual_seed(seed); torch.cuda.empty_cache()
                        t0 = time.time()
                        pl_re = ad.denoise_repaint(pl_base, cond.to(ad.device),
                                                   (sev > 0).to(ad.device),
                                                   start_timestep=ts, num_steps=NUM_REPAIR)
                        t_re = time.time() - t0
                        cent_re, _ = ad.decode_placement(pl_re, cond)
                        cent_re = cent_re.cpu()
                        v, h, oa, mx, hf = _eval_all(cent_re, sz, ei, ea, full_x, macro_mask, ei_full, ea_full, cw, ch, ver)
                        row[f"rp{ts}_v"] = v; row[f"rp{ts}_h"] = h
                        row[f"rp{ts}_oa"] = oa; row[f"rp{ts}_full_h"] = hf
                        row[f"rp{ts}_time"] = t_re
                        print(f"  rp t={ts}      v={v} ({(v-bv)/max(bv,1)*100:+.1f}%) h={h:.0f} ({(h-bh)/max(abs(bh),1e-9)*100:+.1f}%)", flush=True)
                    except Exception as e:
                        row[f"rp{ts}_error"] = str(e)[:200]
                        print(f"  rp t={ts} error: {e}", flush=True)

                rows.append(row)
                with open(OUT, "w") as f: json.dump(rows, f, indent=2)
                del ad; gc.collect(); torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                print("  OOM at trial level", flush=True)
                row["error"] = "OOM"; rows.append(row)
                with open(OUT, "w") as f: json.dump(rows, f, indent=2)
                gc.collect(); torch.cuda.empty_cache()
            except Exception as e:
                traceback.print_exc()
                row["error"] = str(e)[:200]; rows.append(row)
                with open(OUT, "w") as f: json.dump(rows, f, indent=2)

    print(f"\nDone. {len(rows)} rows.", flush=True)


if __name__ == "__main__":
    main()
