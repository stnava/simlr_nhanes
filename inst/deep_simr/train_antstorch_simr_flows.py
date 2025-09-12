#!/usr/bin/env python3
"""
Train SIMR normalizing-flow whiteners using ANTsTorch and (optionally) export latents/reconstructions.

Exports (optional via flags):
  --save-z          : raw flow latents z (one CSV per view)
  --save-whitened   : PCA-projected / standardized latents (eps) (one CSV per view)
  --save-recon      : inverse-transformed reconstructions in observed scale (one CSV per view)

Added stability knobs (forwarded to ANTsTorch whitener if supported):
  --scale-cap, --spectral-norm-scales, --additive-first-n, --actnorm-every, --mask-mode
  --base-min-log, --base-max-log, --base-sigma
  --scale-penalty-weight

These are passed only if the underlying whitener function accepts them (signature-checked).

# Sample call
python3 train_antstorch_simr_flows.py \
  --views ./InputData/nh_list_2.csv ./InputData/nh_list_3.csv ./InputData/nh_list_4.csv ./InputData/nh_list_5.csv \
  --output-prefix ./runs/try1 \
  --base-distribution GaussianPCA --pca-latent-dimension 6 --base-sigma 0.1 \
  --K 8 --scale-cap 2.0 --spectral-norm-scales \
  --additive-first-n 2 --actnorm-every 1 --mask-mode rolling \
  --jitter-alpha 0.05 --jitter-alpha-end 0.005 --jitter-alpha-mode cosine \
  --best-selection-metric smooth_total \
  --max-iter 2000 --val-interval 10 \
  --save-z --save-whitened pca --verbose

"""

import argparse
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import time
import inspect
import warnings

from antstorch import normalizing_simr_flows_whitener, apply_normalizing_simr_flows_whitener


def load_views(view_paths):
    views = []
    for p in view_paths:
        df = pd.read_csv(p)
        # Force numeric; non-numeric will become NaN → trainer imputes with train means
        df = df.apply(pd.to_numeric, errors="coerce")
        views.append(df)
    # Basic shape check
    nset = {len(df) for df in views}
    if len(nset) != 1:
        raise ValueError(f"All views must have equal row counts; got {nset}")
    return views


def save_views(dfs, base_prefix: Path, tag: str):
    paths = []
    for i, df in enumerate(dfs):
        out = Path(f"{base_prefix}_{tag}_view{i}.csv")
        df.to_csv(out, index=False)
        paths.append(str(out))
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", nargs="+", required=True, help="CSV paths for one or more views")
    ap.add_argument("--output-prefix", required=True, help="Prefix for saving models/metrics/config")

    # Model/base
    ap.add_argument("--base-distribution", default="GaussianPCA", choices=["GaussianPCA", "DiagGaussian"])
    ap.add_argument("--pca-latent-dimension", type=int, default=4)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--leaky-relu-negative-slope", type=float, default=0.2)

    # === New stability knobs (flow/builder) ===
    ap.add_argument("--scale-cap", type=float, default=3.0, help="Bound for log-scale s via tanh; exp(s) in [e^-cap, e^cap]")
    ap.add_argument("--spectral-norm-scales", action="store_true", default=False, help="Apply spectral norm in scale MLP (if supported)")
    ap.add_argument("--additive-first-n", type=int, default=0, help="Use additive (no-scaling) couplings for first N layers")
    ap.add_argument("--actnorm-every", type=int, default=1, help="Insert ActNorm after every N couplings (1=after each)")
    ap.add_argument("--mask-mode", type=str, default="alternating", choices=["alternating", "rolling"], help="Mask alternation strategy")

    # === New stability knobs (base distribution) ===
    ap.add_argument("--base-min-log", type=float, default=-5.0, help="Lower clamp for base log-scales (if supported)")
    ap.add_argument("--base-max-log", type=float, default=5.0, help="Upper clamp for base log-scales (if supported)")
    ap.add_argument("--base-sigma", type=float, default=0.1, help="Noise for GaussianPCA (if used)")

    # Optional jitter knob (honored if your antstorch build supports it)
    ap.add_argument("--jitter-alpha", type=float, default=0.0)

    # Jitter annealing (temperature schedule)
    ap.add_argument("--jitter-alpha-end", type=float, default=0.0)
    ap.add_argument("--jitter-alpha-mode", type=str, default="cosine", choices=["cosine", "linear", "exp"])
    ap.add_argument("--jitter-alpha-total-steps", type=int, default=20000)

    # Optimization
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--max-iter", type=int, default=1200)
    ap.add_argument("--cuda-device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)

    # Penalty/tradeoff
    ap.add_argument("--tradeoff-mode", default="uncertainty", choices=["ema", "uncertainty", "fixed"])
    ap.add_argument("--target-ratio", type=float, default=9.0)
    ap.add_argument("--lambda-penalty", type=float, default=1.0)
    ap.add_argument("--ema-beta", type=float, default=0.98)
    ap.add_argument("--penalty-type", default="barlow_twins_align",
                   choices=["decorrelate", "correlate", "barlow_twins_align"])
    ap.add_argument("--bt-lambda-diag", type=float, default=1.0)
    ap.add_argument("--bt-lambda-offdiag", type=float, default=5e-3)
    ap.add_argument("--bt-eps", type=float, default=1e-6)
    ap.add_argument("--penalty-warmup-iters", type=int, default=400)

    # === New: scale regularizer weight (if supported by whitener) ===
    ap.add_argument("--scale-penalty-weight", type=float, default=None,
                    help="Weight for mean|s| regularizer; passed only if whitener supports it")

    # Validation / early stopping
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--val-interval", type=int, default=200)
    ap.add_argument("--val-batch-size", type=int, default=2048)
    ap.add_argument("--early-stop-enabled", action="store_true", default=False)
    ap.add_argument("--early-stop-patience", type=int, default=300)
    ap.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    ap.add_argument("--early-stop-min-iters", type=int, default=600)
    ap.add_argument("--early-stop-beta", type=float, default=0.98)

    # Misc
    ap.add_argument("--best-selection-metric", default="val_bpd", choices=["val_bpd","smooth_total"])

    # Checkpointing / resume
    ap.add_argument("--resume-checkpoint", type=str, default=None)
    ap.add_argument("--save-checkpoint-dir", type=str, default=None)
    ap.add_argument("--checkpoint-interval", type=int, default=None, help="Defaults to --val-interval if omitted")
    ap.add_argument("--restore-best-for-final-eval", action="store_true", default=True)
    ap.add_argument("--verbose", action="store_true", default=False)

    # Optional exports
    ap.add_argument("--save-z", action="store_true", help="Export raw flow latents z_*_view{i}.csv")
    ap.add_argument("--save-whitened", default="pca", choices=["pca","full"])
    ap.add_argument("--save-recon", action="store_true", help="Export inverse reconstructions recon_view{i}.csv (observed scale)")

    args = ap.parse_args()
    verbose = args.verbose

    # Determinism
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load CSV views
    views = load_views(args.views)

    # === Prepare kwargs and filter by whitener signature for backward compatibility ===
    base_kwargs = dict(
        base_distribution=args.base_distribution,
        pca_latent_dimension=args.pca_latent_dimension,
        base_min_log=args.base_min_log,
        base_max_log=args.base_max_log,
        base_sigma=args.base_sigma,
    )

    flow_kwargs = dict(
        K=args.K,
        leaky_relu_negative_slope=args.leaky_relu_negative_slope,
        scale_cap=args.scale_cap,
        spectral_norm_scales=args.spectral_norm_scales,
        additive_first_n=args.additive_first_n,
        actnorm_every=args.actnorm_every,
        mask_mode=args.mask_mode,
    )

    train_kwargs = dict(
        jitter_alpha=args.jitter_alpha,
        jitter_alpha_end=args.jitter_alpha_end,
        jitter_alpha_mode=args.jitter_alpha_mode,
        jitter_alpha_total_steps=args.jitter_alpha_total_steps,

        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        max_iter=args.max_iter,
        cuda_device=args.cuda_device,
        seed=args.seed,

        tradeoff_mode=args.tradeoff_mode,
        target_ratio=args.target_ratio,
        lambda_penalty=args.lambda_penalty,
        ema_beta=args.ema_beta,

        penalty_type=args.penalty_type,
        bt_lambda_diag=args.bt_lambda_diag,
        bt_lambda_offdiag=args.bt_lambda_offdiag,
        bt_eps=args.bt_eps,
        penalty_warmup_iters=args.penalty_warmup_iters,

        val_fraction=args.val_fraction,
        val_interval=args.val_interval,
        val_batch_size=args.val_batch_size,

        early_stop_enabled=args.early_stop_enabled,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        early_stop_min_iters=args.early_stop_min_iters,
        early_stop_beta=args.early_stop_beta,

        best_selection_metric=args.best_selection_metric,
        restore_best_for_final_eval=args.restore_best_for_final_eval,
        resume_checkpoint=args.resume_checkpoint,
        save_checkpoint_dir=args.save_checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        verbose=verbose,
    )

    # Optional: scale penalty weight
    if args.scale_penalty_weight is not None:
        train_kwargs["scale_penalty_weight"] = args.scale_penalty_weight

    # Merge kwargs and filter by signature
    call_kwargs = dict(views=views)
    call_kwargs.update(base_kwargs)
    call_kwargs.update(flow_kwargs)
    call_kwargs.update(train_kwargs)

    sig = inspect.signature(normalizing_simr_flows_whitener)
    filtered_kwargs = {k: v for k, v in call_kwargs.items() if k in sig.parameters}

    # Train
    if verbose:
        print("\n=== Train ===")
        missing = sorted(set(call_kwargs) - set(filtered_kwargs))
        if missing:
            print("Note: the following knobs are not supported by your installed whitener and were omitted:")
            for k in missing:
                print("  -", k)

    start_time = time.perf_counter()
    result = normalizing_simr_flows_whitener(**filtered_kwargs)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print(f"Time elapsed: {elapsed_time:.4f} seconds")

    # Print metrics
    if verbose:
        print("\n=== Metrics ===")
        for k, v in result.get("metrics", {}).items():
            print(f"{k}: {v}")

    # Optional exports using the apply helper
    base_prefix = Path(args.output_prefix)
    if args.save_z or args.save_whitened or args.save_recon:

        # Save
        if verbose:
            print("\n=== Save outputs ===")

        # Forward transforms use the trainer dict (so it has embedded standardizers)
        if args.save_z:
            z_views = apply_normalizing_simr_flows_whitener(
                trainer_output=result,
                data=views,
                direction="forward",
                output_space="z",
                use_training_standardization=True,
                batch_size=args.val_batch_size,
                device=args.cuda_device
            )
            z_paths = save_views(z_views, base_prefix, "z")
            if verbose:
                print("  z latents:")
                for p in z_paths:
                    print("  ", p)

        if args.save_whitened == "pca":
            if args.base_distribution == "GaussianPCA":
                wh_views = apply_normalizing_simr_flows_whitener(
                    trainer_output=result,
                    data=views,
                    direction="forward",
                    output_space="whitened",
                    use_training_standardization=True,
                    batch_size=args.val_batch_size,
                    device=args.cuda_device
                )
                wh_paths = save_views(wh_views, base_prefix, "whitened")
                if verbose:
                    print("  whitened pca latents:")
                    for p in wh_paths:
                        print("  ", p)

        elif args.save_whitened == "full":
            if args.base_distribution == "GaussianPCA":
                wh_views = apply_normalizing_simr_flows_whitener(
                    trainer_output=result,
                    data=views,
                    direction="forward",
                    output_space="whitened_full",
                    use_training_standardization=True,
                    batch_size=args.val_batch_size,
                    device=args.cuda_device
                )
                wh_paths = save_views(wh_views, base_prefix, "whitened_full")
                if verbose:
                    print("  whitened full:")
                    for p in wh_paths:
                        print("  ", p)
            else:
                warnings.warn("Requested --save-whitened-full with base_distribution=DiagGaussian. "
                              "Whitened is undefined.")                           

        if args.save_recon:
            # Choose input space to match what we saved most recently; default to whitened if requested, else z
            if args.save_whitened == "pca" and args.base_distribution == "GaussianPCA":
                inv_input = "whitened"
                inv_data = wh_views
                warnings.warn("Requested reconstruction won't be accurate.")                           
            elif args.save_whitened == "full" and args.base_distribution == "GaussianPCA":
                inv_input = "whitened_full"
                inv_data = wh_views
            else:
                inv_input = "z"
                # If we didn't compute z already, do it now (cheap)
                if not args.save_z:
                    z_views = apply_normalizing_simr_flows_whitener(
                        trainer_output=result,
                        data=views,
                        direction="forward",
                        output_space="z",
                        use_training_standardization=True,
                        batch_size=args.val_batch_size,
                        device=args.cuda_device
                    )
                inv_data = z_views

            recon_views = apply_normalizing_simr_flows_whitener(
                trainer_output=result["models"],   # can pass models directly
                data=inv_data,
                direction="inverse",
                input_space=inv_input,
                use_training_standardization=True,
                custom_standardizers=result.get("standardizers", None),
                batch_size=args.val_batch_size,
                device=args.cuda_device
            )
            recon_paths = save_views(recon_views, base_prefix, "recon")
            if verbose:
                print("  reconstructions:")
                for p in recon_paths:
                    print("  ", p)


if __name__ == "__main__":
    main()
