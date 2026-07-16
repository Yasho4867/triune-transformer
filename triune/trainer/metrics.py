import math
import torch

target_depth_dist = torch.tensor(config["target_depth_dist"], device=device)
depth_usage_ema = target_depth_dist.clone()
best_eval_loss = float("inf")

# ─── Resume ────────────────────────────────────────────────────
start_step = 0
resume_path = None
if not args.fresh:
    if args.resume_latest and os.path.exists(args.resume_latest):
        resume_path = args.resume_latest
    elif args.resume_best and os.path.exists(os.path.join(config["checkpoint_dir"], "best.pt")):
        resume_path = os.path.join(config["checkpoint_dir"], "best.pt")
    elif os.path.exists(os.path.join(config["checkpoint_dir"], "latest.pt")):
        resume_path = os.path.join(config["checkpoint_dir"], "latest.pt")
if resume_path:
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    sd = ckpt["model_state"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    # Restore depth_usage_ema if present
    if "depth_usage_ema" in ckpt:
        depth_usage_ema = ckpt["depth_usage_ema"].to(device)
    if "best.pt" in resume_path:
        print("✅ Resumed best weights (fresh optimizer)")
        if args.resume_best:
            print("⚠️ --resume_best resets start_step to 0 → LR schedule restarts from warmup")
        start_step = 0
        best_eval_loss = ckpt.get("best_eval_loss", float("inf"))
    else:
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_step = ckpt["step"] + 1
        best_eval_loss = ckpt.get("best_eval_loss", float("inf"))
        print(f"✅ Resumed from {resume_path} at step {start_step}")
else:
    print("🆕 Fresh start")

if args.compile:
    try:
        model.forward = torch.compile(model.forward, fullgraph=False, dynamic=True)
        print("✅ torch.compile enabled")
    except Exception as e:
        print(f"⚠️ torch.compile failed: {e}")

