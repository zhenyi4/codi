"""Guard tests for the GPT-2 depth-recurrent thinking stage.

Run on CPU:  python tests/test_depth_recurrent_gpt2.py

Checks (all must PASS):
  1. Equivalence: prelude + core-once + coda (each block once, K=1) matches the
     black-box latent step `self.codi(inputs_embeds=seed, past_key_values=qcache,
     use_cache=True, output_hidden_states=True)` for per-layer slot hidden states
     AND the appended slot KV. Tested un-padded, then left-padded batch.
  2. Cache length: after thinking with K=6, every layer's final cache seq length
     == q_len + 1 (not q_len + 6).
  3. Fixed position: the wpe position used is identical across all K iterations.
  4. Pad-invariance: perturbing a padded example's pad KV does not change the
     other example's slot output.
  5. Baseline: depth_recurrent=False forward returns a finite loss (CODI intact).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from peft import LoraConfig, TaskType

from src.model import CODI, ModelArguments, TrainingArguments

ATOL = 1e-4
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def build_model(depth_recurrent, use_prj=True):
    model_args = ModelArguments(model_name_or_path="gpt2", full_precision=True, train=True, lora_init=True)
    training_args = TrainingArguments(
        output_dir="/tmp/dr_test",
        bf16=False,  # -> fp16 load; we cast to fp32 below for clean CPU math
        num_latent=6,
        use_lora=True,
        use_prj=use_prj,
        prj_dim=768,
        prj_no_ln=False,
        depth_recurrent=depth_recurrent,
        num_prelude=1,
        num_coda=1,
        fix_attn_mask=False,
        print_loss=False,
        distill_loss_div_std=True,
    )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["c_attn", "c_proj", "c_fc"],
        init_lora_weights=True,
    )
    model = CODI(model_args, training_args, lora_config)
    model.float()
    return model


def black_box_latent_step(model, seed, qKV, enc_mask):
    # Replicate exactly what CODI's first latent step does, but pass the proper
    # 2D attention mask so the black box also masks the question's left-pad KVs.
    full2d = torch.cat([enc_mask, torch.ones(enc_mask.size(0), 1, dtype=enc_mask.dtype)], dim=1)
    return model.codi(
        inputs_embeds=seed,
        past_key_values=qKV,
        attention_mask=full2d,
        use_cache=True,
        output_hidden_states=True,
    )


def encode(model, ids, mask):
    out = model.codi(input_ids=ids, attention_mask=mask, use_cache=True, output_hidden_states=True)
    seed = out.hidden_states[-1][:, -1, :].unsqueeze(1)
    if model.use_prj:
        seed = model.prj(seed)
    return seed, out.past_key_values


def main():
    torch.manual_seed(0)
    model = build_model(depth_recurrent=True)
    model.eval()
    backend = model.depth_backend()
    L = model.get_base_model().config.num_hidden_layers

    # ---------- 1a. Equivalence: single un-padded example ----------
    with torch.no_grad():
        ids = torch.tensor([[15496, 11, 314, 716, 257]])
        mask = torch.ones_like(ids)
        seed, qKV = encode(model, ids, mask)
        bb = black_box_latent_step(model, seed, qKV, mask)
        final_past, hs, slot_kv = backend.think(seed, qKV, mask, num_iterations=1, return_hidden=True)

        q_len = qKV[0][0].shape[-2]
        d_hs = max((hs[i] - bb.hidden_states[i]).abs().max().item() for i in range(len(hs)))
        d_k = max((slot_kv[j][0] - bb.past_key_values[j][0][:, :, q_len:, :]).abs().max().item() for j in range(L))
        d_v = max((slot_kv[j][1] - bb.past_key_values[j][1][:, :, q_len:, :]).abs().max().item() for j in range(L))
        ok = len(hs) == len(bb.hidden_states) and d_hs < ATOL and d_k < ATOL and d_v < ATOL
        check("1a. equivalence (un-padded, K=1)", ok, f"d_hidden={d_hs:.2e} d_K={d_k:.2e} d_V={d_v:.2e}")

    # ---------- 1b. Equivalence: left-padded batch ----------
    with torch.no_grad():
        pad = model.pad_token_id
        ids_b = torch.tensor([[pad, pad, 15496, 11, 314], [15496, 11, 314, 716, 257]])
        mask_b = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
        seed_b, qKV_b = encode(model, ids_b, mask_b)
        bb_b = black_box_latent_step(model, seed_b, qKV_b, mask_b)
        fp_b, hs_b, slot_b = backend.think(seed_b, qKV_b, mask_b, num_iterations=1, return_hidden=True)
        q_len_b = qKV_b[0][0].shape[-2]
        d_hs = max((hs_b[i] - bb_b.hidden_states[i]).abs().max().item() for i in range(len(hs_b)))
        d_k = max((slot_b[j][0] - bb_b.past_key_values[j][0][:, :, q_len_b:, :]).abs().max().item() for j in range(L))
        check("1b. equivalence (left-padded batch, K=1)", d_hs < ATOL and d_k < ATOL, f"d_hidden={d_hs:.2e} d_K={d_k:.2e}")

    # ---------- 2. Cache length with K=6 ----------
    with torch.no_grad():
        fp6 = backend.think(seed_b, qKV_b, mask_b, num_iterations=6)
        lens = set(fp6[j][0].shape[-2] for j in range(L))
        check("2. cache length == q_len+1 for K=6", lens == {q_len_b + 1}, f"lens={lens}, expected {{{q_len_b + 1}}}")

    # ---------- 3. Fixed position across all K iterations ----------
    seen_positions = []
    orig_wpe = backend.transformer.wpe

    class WpeSpy(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, position_ids):
            seen_positions.append(int(position_ids.reshape(-1)[0].item()))
            return self.inner(position_ids)

    backend.transformer.wpe = WpeSpy(orig_wpe)
    try:
        with torch.no_grad():
            backend.think(seed_b, qKV_b, mask_b, num_iterations=6)
    finally:
        backend.transformer.wpe = orig_wpe
    fixed = len(set(seen_positions)) == 1 and seen_positions and seen_positions[0] == q_len_b
    check("3. fixed wpe position across K iterations", fixed, f"positions seen={set(seen_positions)}, q_len={q_len_b}")

    # ---------- 4. Pad-invariance ----------
    with torch.no_grad():
        ids_solo = torch.tensor([[15496, 11, 314, 716, 257]])
        mask_solo = torch.ones_like(ids_solo)
        seed_solo, qKV_solo = encode(model, ids_solo, mask_solo)
        _, hs_solo, _ = backend.think(seed_solo, qKV_solo, mask_solo, num_iterations=1, return_hidden=True)
        qKV_pert = tuple((k.clone(), v.clone()) for (k, v) in qKV_b)
        for j in range(L):
            qKV_pert[j][0][0, :, :2, :] += 7.0  # pad positions of example 0
            qKV_pert[j][1][0, :, :2, :] -= 3.0
        _, hs_pert, _ = backend.think(seed_b, qKV_pert, mask_b, num_iterations=1, return_hidden=True)
        d_other = (hs_pert[-1][1:2] - hs_solo[-1]).abs().max().item()
        check("4. pad-invariance (other example unaffected)", d_other < ATOL, f"d_other={d_other:.2e}")

    # ---------- 5. Baseline forward returns finite loss ----------
    base_model = build_model(depth_recurrent=False)
    base_model.float()
    base_model.train()
    batch = synthetic_batch(base_model)
    out = base_model(**batch)
    loss = out["loss"]
    check("5. baseline (depth_recurrent=False) finite loss", torch.isfinite(loss).item(), f"loss={loss.item():.4f}")

    # ---------- 5b. Depth-recurrent forward returns finite loss (smoke) ----------
    dr_model = build_model(depth_recurrent=True)
    dr_model.float()
    dr_model.train()
    batch2 = synthetic_batch(dr_model)
    out2 = dr_model(**batch2)
    loss2 = out2["loss"]
    check("5b. depth_recurrent=True finite loss (smoke)", torch.isfinite(loss2).item(), f"loss={loss2.item():.4f}")

    print()
    if all(results):
        print("ALL CHECKS PASSED")
        return 0
    print(f"{results.count(False)}/{len(results)} CHECKS FAILED")
    return 1


def synthetic_batch(model):
    """A tiny valid CODI batch (batch_size=2) with in-range answer positions."""
    pad = model.pad_token_id
    bot, eot = model.bot_id, model.eot_id
    eos = model.tokenizer.eos_token_id if model.tokenizer.eos_token_id is not None else 50256
    IGN = -100

    enc = torch.tensor([
        [pad, 10, 20, 30, eos, bot],
        [11, 21, 31, 41, eos, bot],
    ])
    enc_mask = enc.ne(pad)

    dec = torch.tensor([
        [eot, 100, 101, eos],
        [eot, 200, 201, eos],
    ])
    labels = dec.clone()

    ref = torch.tensor([
        [10, 20, 30, 100, 101, 100, 101, eos],
        [21, 31, 41, 200, 201, 200, 201, eos],
    ])
    ref_labels = ref.clone()
    ref_labels[:, :3] = IGN

    return dict(
        encoder_input_ids=enc,
        decoder_input_ids=dec,
        ref_input_ids=ref,
        labels=labels,
        encoder_attention_mask=enc_mask,
        ref_answer_position=torch.tensor([5, 5], dtype=torch.long),
        model_answer_position=torch.tensor([2, 2], dtype=torch.long),
        ref_attention_mask=ref.ne(pad),
        ref_labels=ref_labels,
    )


if __name__ == "__main__":
    sys.exit(main())
