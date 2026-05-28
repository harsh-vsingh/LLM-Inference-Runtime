import torch
import flashinfer
from typing import Optional, Tuple, cast
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb

class FlashInferState:
    kv_pool: Optional[torch.Tensor] = None

    kv_indices: Optional[torch.Tensor] = None
    kv_indptr: Optional[torch.Tensor] = None
    kv_last_page_len: Optional[torch.Tensor] = None

    batch_indices: Optional[torch.Tensor] = None
    positions: Optional[torch.Tensor] = None

    workspace_buffer: Optional[torch.Tensor] = None

    prefill_wrapper: Optional[
        flashinfer.BatchPrefillWithPagedKVCacheWrapper
    ] = None

    decode_wrapper: Optional[
        flashinfer.BatchDecodeWithPagedKVCacheWrapper
    ] = None

    page_size: int = 16
    dtype: Optional[torch.dtype] = None


def init_flashinfer_state(
    *,
    kv_pool: torch.Tensor,
    page_size: int,
    device: str,
):
    FlashInferState.kv_pool = kv_pool
    FlashInferState.page_size = page_size
    FlashInferState.dtype = kv_pool.dtype

    FlashInferState.workspace_buffer = torch.empty(
        128 * 1024 * 1024,
        dtype=torch.uint8,
        device=device,
    )

    FlashInferState.prefill_wrapper = (
        flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            FlashInferState.workspace_buffer,
            "HND",
        )
    )

    FlashInferState.decode_wrapper = (
        flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            FlashInferState.workspace_buffer,
            "HND",
        )
    )


def build_batch_indices_positions(
    append_indptr: torch.Tensor,
    seq_lens: torch.Tensor,
):
    nnz = int(append_indptr[-1].item())

    (
        FlashInferState.batch_indices,
        FlashInferState.positions,
    ) = flashinfer.get_batch_indices_positions(
        append_indptr,
        seq_lens,
        nnz,
    )


def plan_prefill(
    *,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
):
    FlashInferState.kv_indptr = kv_indptr
    FlashInferState.kv_indices = kv_indices
    FlashInferState.kv_last_page_len = kv_last_page_len

    pw = cast(
        flashinfer.BatchPrefillWithPagedKVCacheWrapper,
        FlashInferState.prefill_wrapper,
    )

    pw.begin_forward(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        FlashInferState.page_size,
        q_data_type=FlashInferState.dtype,
    )


def plan_decode(
    *,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
):
    FlashInferState.kv_indptr = kv_indptr
    FlashInferState.kv_indices = kv_indices
    FlashInferState.kv_last_page_len = kv_last_page_len

    dw = cast(
        flashinfer.BatchDecodeWithPagedKVCacheWrapper,
        FlashInferState.decode_wrapper,
    )

    dw.begin_forward(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        FlashInferState.page_size,
        q_data_type=FlashInferState.dtype,
    )


def patched_llama_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    attention_mask=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
    cache_position=None,
    **kwargs,
):
    bsz, q_len, _ = hidden_states.shape

    q = self.q_proj(hidden_states)
    k = self.k_proj(hidden_states)
    v = self.v_proj(hidden_states)

    q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    k = k.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    v = v.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

    assert position_embeddings is not None

    cos, sin = position_embeddings
    
    q, k = apply_rotary_pos_emb(q, k, cos, sin)

    q_flat = q.transpose(1, 2).reshape(-1, self.num_heads, self.head_dim)
    k_flat = k.transpose(1, 2).reshape(-1, self.num_key_value_heads, self.head_dim)
    v_flat = v.reshape(-1, self.num_key_value_heads, self.head_dim)

    kv_pool = cast(torch.Tensor, FlashInferState.kv_pool)

    layer_kv_cache = kv_pool[self.layer_idx]

    flashinfer.append_paged_kv_cache(
        append_key=k_flat,
        append_value=v_flat,
        batch_indices=cast(
            torch.Tensor,
            FlashInferState.batch_indices,
        ),
        positions=cast(
            torch.Tensor,
            FlashInferState.positions,
        ),
        paged_kv_cache=layer_kv_cache,
        kv_indices=cast(
            torch.Tensor,
            FlashInferState.kv_indices,
        ),
        kv_indptr=cast(
            torch.Tensor,
            FlashInferState.kv_indptr,
        ),
        kv_last_page_len=cast(
            torch.Tensor,
            FlashInferState.kv_last_page_len,
        ),
        kv_layout="HND",
    )

    if q_len > 1:
        pw = cast(
            flashinfer.BatchPrefillWithPagedKVCacheWrapper,
            FlashInferState.prefill_wrapper,
        )

        out = pw.forward(
            q=q_flat,
            paged_kv_cache=layer_kv_cache,
            causal=True,
        )

        out = out.reshape(
            bsz,
            q_len,
            self.num_heads * self.head_dim,
        )

    else:
        dw = cast(
            flashinfer.BatchDecodeWithPagedKVCacheWrapper,
            FlashInferState.decode_wrapper,
        )

        out = dw.forward(
            q=q_flat,
            paged_kv_cache=layer_kv_cache,
        )

        out = out.reshape(
            bsz,
            1,
            self.num_heads * self.head_dim,
        )

    out = self.o_proj(out)

    return out, None


def patch_llama_model(model):
    for idx, layer in enumerate(model.model.layers):
        attn: LlamaAttention = layer.self_attn

        attn.layer_idx = idx
        attn.num_heads = model.config.num_attention_heads
        attn.num_key_value_heads = model.config.num_key_value_heads
        attn.head_dim = model.config.hidden_size // model.config.num_attention_heads

        attn.forward = patched_llama_attention_forward.__get__(
            attn,
            LlamaAttention,
        )
