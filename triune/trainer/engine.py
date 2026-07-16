"""
Training engine extracted from the legacy trainer.

NOTE:
This is intentionally copied verbatim first.
The Trainer migration will happen afterwards.
"""

class TrainingEngine:

    def __init__(self, trainer):
        self.trainer = trainer

    def train(self):

        try:
            while step < config["total_steps"]:
                lr = lr_schedule(step)
                set_optimizer_lr(lr)

                # Sync global step to MoE layers
                for module in model.modules():
                    if isinstance(module, MoE_FFN):
                        module._global_step = step

                optimizer.zero_grad()
                acc_loss = 0.0
                acc_lm = 0.0
                acc_router = 0.0
                acc_balance = 0.0
                acc_overflow = 0
                last_route = None

                for micro in range(grad_accum):
                    x, y = next_batch()
                    x, y = x.to(device), y.to(device)

                    # ─── Router labels per sample ─────────────────────
                    with torch.no_grad():
                        # Ensure MoE layers don't update stats during label generation
                        reflex, limbic, cortex, _ = model.forward_all_exits(x, update_stats=False)
                        y_flat = y.contiguous().view(-1)
                        loss_reflex = loss_fn(reflex.contiguous().view(-1, vocab_size), y_flat)
                        loss_limbic = loss_fn(limbic.contiguous().view(-1, vocab_size), y_flat)
                        loss_cortex = loss_fn(cortex.contiguous().view(-1, vocab_size), y_flat)
                        valid_mask = (y_flat != pad_token_id)
                        valid_per_sample = valid_mask.view(x.shape[0], -1).sum(dim=1).clamp_min(1)
                        loss_reflex = (loss_reflex * valid_mask.float()).view(x.shape[0], -1).sum(dim=1) / valid_per_sample
                        loss_limbic = (loss_limbic * valid_mask.float()).view(x.shape[0], -1).sum(dim=1) / valid_per_sample
                        loss_cortex = (loss_cortex * valid_mask.float()).view(x.shape[0], -1).sum(dim=1) / valid_per_sample
                        losses = torch.stack([loss_reflex, loss_limbic, loss_cortex], dim=1)
                        usage_gap = target_depth_dist - depth_usage_ema
                        adjusted = losses - config["bias_strength"] * usage_gap.unsqueeze(0)
                        best_depth = adjusted.argmin(dim=1)
                        depth_labels = best_depth
                        one_hot = F.one_hot(best_depth, num_classes=3).float().mean(dim=0)
                        depth_usage_ema.mul_(config["usage_ema_decay"]).add_(one_hot * (1 - config["usage_ema_decay"]))

                    # Exploration
                    expl_type = config["exploration"]
                    expl_steps = config["exploration_steps"]
                    if expl_type == "none":
                        exploration_rate = 0.0
                    elif expl_type == "linear":
                        exploration_rate = max(0.0, 1.0 - step / expl_steps)
                    elif expl_type == "cosine":
                        if step < expl_steps:
                            exploration_rate = 0.5 * (1 + math.cos(math.pi * step / expl_steps))
                        else:
                            exploration_rate = 0.0
                    else:
                        exploration_rate = 0.0

                    if random.random() < exploration_rate:
                        chosen_depth = random.choice([0, 1, 2])
                    else:
                        chosen_depth = None

                    # ─── Forward pass ──────────────────────────────────
                    with model_autocast():
                        logits, route_logits = model(x, force_depth=chosen_depth)

                    y_flat = y.contiguous().view(-1)
                    lm_loss = loss_fn(logits.contiguous().view(-1, vocab_size), y_flat).mean()

                    rloss = router_loss_fn(route_logits, depth_labels)
                    z_loss = torch.logsumexp(route_logits, dim=-1).pow(2).mean()
                    probs = torch.softmax(route_logits, dim=-1)
                    mean_prob = probs.mean(dim=0)
                    balance_loss = (mean_prob - target_depth_dist).pow(2).mean()

                    micro_loss = lm_loss + 0.5 * rloss + z_loss_coef * z_loss + config["balance_coef"] * balance_loss
                    acc_router += rloss.item()
                    acc_balance += balance_loss.item()

                    (micro_loss / grad_accum).backward()
                    acc_loss += micro_loss.item()
                    acc_lm += lm_loss.item()
                    last_route = route_logits

                    # Accumulate overflow
                    for module in model.modules():
                        if isinstance(module, MoE_FFN):
                            acc_overflow += module.overflow_counter
                            module.overflow_counter = 0

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                loss_val = acc_loss / grad_accum
                lm_val = acc_lm / grad_accum
                router_val = acc_router / grad_accum
                balance_val = acc_balance / grad_accum

                if step % config["eval_every"] == 0 and step > 0:
                    eval_loss = run_eval(force_depth=2)       # Cortex
                    dynamic_eval_loss = run_eval(force_depth=None)  # Router's own choice
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        save_best(step, eval_loss)
                    sample = run_sample()
                    print(f"   📊 Eval (Cortex): {eval_loss:.4f} | Dynamic: {dynamic_eval_loss:.4f} | Perplexity: {math.exp(min(eval_loss,20)):.2f}")
                    print(f"   📝 Sample: {SAMPLE_PROMPT}{sample}\n")
                    if not args.no_wandb:
                        wandb.log({
                            "eval/loss": eval_loss,
                            "eval/dynamic_loss": dynamic_eval_loss,
                            "eval/perplexity": math.exp(min(eval_loss, 20)),
                            "best_eval_loss": best_eval_loss,
                        }, step=step)

                if step % config["save_every"] == 0 and step > 0:
                    save_latest(step, loss_val)

                if step % config["log_every"] == 0:
                    depth_map = {0: "Reflex", 1: "Limbic", 2: "Cortex"}
                    chosen = last_route.argmax(dim=-1)[0].item()
                    label = depth_labels[0].item()
                    usage_str = f"R:{depth_usage_ema[0]:.2f} L:{depth_usage_ema[1]:.2f} C:{depth_usage_ema[2]:.2f}"
                    print(f"Step {step:6d}/{config['total_steps']} | Loss: {loss_val:.4f} (LM: {lm_val:.4f}, Router: {router_val:.4f}, Bal: {balance_val:.4f}) "
                          f"| Router: {depth_map[chosen]} (target: {depth_map[label]}) "
                          f"| Usage: {usage_str} | LR: {lr:.2e} | VRAM: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB | Best Eval: {best_eval_loss:.4f} | Overflow: {acc_overflow}")

                if not args.no_wandb and step % config["log_every"] == 0:
                    wandb.log({
                        "train/loss": loss_val,
                        "train/lm_loss": lm_val,
                        "train/router_loss": router_val,
                        "train/balance_loss": balance_val,
                        "train/lr": lr,
                        "usage/reflex": depth_usage_ema[0].item(),
                        "usage/limbic": depth_usage_ema[1].item(),
                        "usage/cortex": depth_usage_ema[2].item(),
                        "vram": torch.cuda.memory_allocated(device)/1024**3,
                        "best_eval_loss": best_eval_loss,
                        "exploration_rate": exploration_rate,
                        "overflow": acc_overflow,
                    }, step=step)

                step += 1

        except KeyboardInterrupt:
            print("\n⚠️ KeyboardInterrupt – saving checkpoint...")
            save_latest(step, 0.0)
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Unhandled exception: {e}")
            save_latest(step, 0.0)
            raise

        save_latest(step-1, 0.0)
