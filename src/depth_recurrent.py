"""Depth-recurrent ("looped middle-block") thinking stage for CODI.

This module isolates the architecture-specific machinery for running CODI's
thinking stage as depth recurrence over a transformer's middle blocks, instead
of the baseline autoregressive latent loop.

Phase 1 implements the GPT-2 backend only. The seam (`DepthRecurrentBackend`
+ `build_depth_backend`) is intentionally small so Llama / NeoX backends can be
added additively in phase 2 without touching `CODI`.

Verified empirically against transformers==4.52.4:
  * `GPT2Block.forward(hidden_states, past_key_value=<Cache>, cache_position=...,
    attention_mask=<4D additive>, use_cache=True)` -> `(hidden_states, ...)`.
    The block does NOT return present KV; it mutates the passed `Cache` in place
    via `cache.update(k, v, layer_idx)`, keyed on the attention's fixed
    `layer_idx`. We therefore read the appended slot KV back out of the cache.
  * `GPT2LMHeadModel(..., use_cache=True)` returns `past_key_values` as a LEGACY
    tuple `((k, v), ...)` with k/v shaped `[B, n_head, seq, head_dim]`.
  * `GPT2Model.forward` adds the position embedding once at the input via
    `wpe(position_ids)` where `position_ids == past_seen_tokens` (the cache
    length); blocks 1..L-1 do NOT re-add it. We replicate that by adding
    `wpe(q_len)` once to the seed.
  * `output_hidden_states` returns L+1 tensors; the last one has `ln_f` applied.
"""

import torch
from transformers.cache_utils import DynamicCache


class DepthRecurrentBackend:
    """Interface for an architecture-specific depth-recurrent thinking stage."""

    def think(
        self,
        seed,
        question_past,
        encoder_attention_mask,
        num_iterations,
        num_prelude=1,
        num_coda=1,
        core_bptt_window=-1,
        return_hidden=False,
    ):
        """Run the looped-middle-block thinking stage for a single thought slot.

        Args:
            seed: `[B, 1, H]` post-`prj` hidden state (the existing `latent_embd`).
            question_past: the question's per-layer KV cache as a LEGACY tuple
                `((k, v), ...)`, exactly as returned by `self.codi(...).past_key_values`.
            encoder_attention_mask: `[B, q_len]` 1/0 mask for the question KV.
            num_iterations: K, the number of times the recurrent core is looped.
            num_prelude / num_coda: blocks run once before / after the core.
            core_bptt_window: if > 0, detach the carried hidden so gradients only
                flow through the last `core_bptt_window` core iterations.
            return_hidden: also return the per-layer slot hidden states (diagnostics;
                exact match to a plain forward only for the K=1, 1/1 anchor).

        Returns:
            final_past (a legacy tuple with one (key, value) pair per layer; each
            tensor has sequence length q_len + 1), or
            (final_past, hidden_states, slot_kv) when `return_hidden=True`.
        """
        raise NotImplementedError


class GPT2DepthBackend(DepthRecurrentBackend):
    def __init__(self, base):
        # `base` is the unwrapped GPT2LMHeadModel (PEFT base model). Its blocks may
        # still be LoRA-wrapped; calling them manually routes through the adapters.
        self.base = base
        self.transformer = base.transformer
        self.num_layers = base.config.num_hidden_layers

    def _build_attention_mask(self, encoder_attention_mask, dtype):
        # 4D additive mask [B, 1, 1, q_len + 1]: real question positions + the
        # slot's own position -> 0, padded question positions -> large negative.
        # Always applied in the thinking stage so the slot never attends to
        # left-pad KVs (independent of fix_attn_mask).
        batch_size = encoder_attention_mask.size(0)
        slot = torch.ones(
            batch_size, 1, dtype=encoder_attention_mask.dtype, device=encoder_attention_mask.device
        )
        full = torch.cat([encoder_attention_mask, slot], dim=1)  # [B, q_len + 1]
        neg = torch.finfo(dtype).min
        return (1.0 - full.to(dtype))[:, None, None, :] * neg

    def _run_block(self, block, x, layer_k, layer_v, layer_idx, attn_mask, cache_position):
        # Seed a fresh DynamicCache so the block always attends to the question KV
        # (length q_len) plus the single new slot token -- never a growing cache.
        cache = DynamicCache()
        for _ in range(layer_idx):
            cache.key_cache.append(torch.empty(0, device=layer_k.device))
            cache.value_cache.append(torch.empty(0, device=layer_k.device))
        cache.key_cache.append(layer_k)
        cache.value_cache.append(layer_v)

        out = block(
            x,
            past_key_value=cache,
            cache_position=cache_position,
            attention_mask=attn_mask,
            use_cache=True,
        )
        hidden = out[0]
        q_len = layer_k.shape[-2]
        slot_k = cache.key_cache[layer_idx][:, :, q_len:, :]
        slot_v = cache.value_cache[layer_idx][:, :, q_len:, :]
        return hidden, (slot_k, slot_v)

    def think(
        self,
        seed,
        question_past,
        encoder_attention_mask,
        num_iterations,
        num_prelude=1,
        num_coda=1,
        core_bptt_window=-1,
        return_hidden=False,
    ):
        L = self.num_layers
        if num_iterations < 1:
            raise ValueError(
                f"depth_recurrent requires num_iterations (K) >= 1; got {num_iterations}. "
                f"(K is num_latent in training / inf_latent_iterations in inference.)"
            )
        if num_prelude < 1 or num_coda < 1 or num_prelude + num_coda >= L:
            raise ValueError(
                f"num_prelude/num_coda must be >=1 and leave a non-empty core "
                f"(got prelude={num_prelude}, coda={num_coda}, L={L})"
            )

        tr = self.transformer
        dtype = seed.dtype
        q_len = question_past[0][0].shape[-2]

        # GPT-2 is absolute-position: add wpe once at the seed (the black-box
        # latent step adds wpe(past_length) internally). The slot's position is
        # the question cache length and is FIXED across all K iterations.
        position_ids = torch.full((1, 1), q_len, dtype=torch.long, device=seed.device)
        x = seed + tr.wpe(position_ids)
        x = tr.drop(x)

        attn_mask = self._build_attention_mask(encoder_attention_mask, dtype)
        cache_position = torch.tensor([q_len], device=seed.device)

        slot_kv = [None] * L
        # hidden_states is purely diagnostic (only used when return_hidden=True): the
        # answer cache is assembled from slot_kv, not from these. Skip the bookkeeping
        # -- including the coda ln_f -- on the normal training/inference path.
        hidden_states = [x] if return_hidden else None  # hidden_states[0] == input to block 0

        prelude_idx = list(range(0, num_prelude))
        core_idx = list(range(num_prelude, L - num_coda))
        coda_idx = list(range(L - num_coda, L))

        # --- Prelude: run once ---
        for j in prelude_idx:
            x, slot_kv[j] = self._run_block(
                tr.h[j], x, question_past[j][0], question_past[j][1], j, attn_mask, cache_position
            )
            if return_hidden:
                hidden_states.append(x)

        # --- Recurrent core: looped K times, weight-tied. Recurrent state is x. ---
        for k in range(num_iterations):
            for j in core_idx:
                x, kv = self._run_block(
                    tr.h[j], x, question_past[j][0], question_past[j][1], j, attn_mask, cache_position
                )
                if k == num_iterations - 1:  # expose only the last iteration's KV
                    slot_kv[j] = kv
                    if return_hidden:
                        hidden_states.append(x)
            if core_bptt_window > 0 and k < num_iterations - core_bptt_window:
                x = x.detach()

        # --- Coda: run once. (When return_hidden, ln_f of the final output is appended.) ---
        for n, j in enumerate(coda_idx):
            x, slot_kv[j] = self._run_block(
                tr.h[j], x, question_past[j][0], question_past[j][1], j, attn_mask, cache_position
            )
            if return_hidden and n < len(coda_idx) - 1:
                hidden_states.append(x)
        if return_hidden:
            hidden_states.append(tr.ln_f(x))

        # Assemble the final cache: question KV + the single thought slot KV.
        final_past = tuple(
            (
                torch.cat([question_past[j][0], slot_kv[j][0]], dim=2),
                torch.cat([question_past[j][1], slot_kv[j][1]], dim=2),
            )
            for j in range(L)
        )

        if return_hidden:
            return final_past, hidden_states, slot_kv
        return final_past


def build_depth_backend(base, model_name):
    """Dispatch on architecture. Phase 1: GPT-2 only. Llama/NeoX are additive here."""
    name = model_name.lower()
    if "gpt2" in name:
        return GPT2DepthBackend(base)
    raise NotImplementedError(
        f"depth_recurrent is only implemented for GPT-2 (phase 1); got '{model_name}'."
    )
