from typing import cast

import flashinfer
import torch
from torch import nn
from transformers.models.llama.modeling_llama import LlamaAttention

try:
    import bitsandbytes as bnb
    from bitsandbytes.functional import dequantize_4bit
    from bitsandbytes.nn import Linear4bit, Params4bit
    _BNB_AVAILABLE = True
except ImportError:
    _BNB_AVAILABLE = False

class FlashInferState:
    kv_pool: torch.Tensor | None = None

    kv_indices: torch.Tensor | None = None
    kv_indptr: torch.Tensor | None = None
    kv_last_page_len: torch.Tensor | None = None

    batch_indices: torch.Tensor | None = None
    positions: torch.Tensor | None = None

    workspace_buffer: torch.Tensor | None = None

    prefill_wrapper: flashinfer.BatchPrefillWithPagedKVCacheWrapper | None = None

    decode_wrapper: flashinfer.BatchDecodeWithPagedKVCacheWrapper | None = None

    page_size: int = 16
    dtype: torch.dtype | None = None


    mode: str = "prefill"


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
    FlashInferState.mode = "prefill"
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
    FlashInferState.mode = "decode"
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
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    attention_mask=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
    cache_position=None,
    **kwargs,
):
    bsz, q_len, _ = hidden_states.shape

    qkv = self.qkv_proj(hidden_states)
    q, k, v = qkv.split(
        [self.q_size, self.kv_size, self.kv_size],
        dim=-1,
    )

    q_flat = q.view(-1, self.num_heads, self.head_dim)
    k_flat = k.view(-1, self.num_key_value_heads, self.head_dim)
    v_flat = v.view(-1, self.num_key_value_heads, self.head_dim)

    positions = cast(torch.Tensor, FlashInferState.positions)

    flashinfer.apply_rope_pos_ids_inplace(
        q_flat,
        k_flat,
        positions,
        rope_theta=self.rope_theta,
        rope_scale=self.rope_scale,
        interleave=False,
    )

    kv_pool = cast(torch.Tensor, FlashInferState.kv_pool)

    layer_kv_cache = kv_pool[self.layer_idx]

    flashinfer.append_paged_kv_cache(
        append_key=k_flat,
        append_value=v_flat,
        batch_indices=cast(
            torch.Tensor,
            FlashInferState.batch_indices,
        ),
        positions=positions,
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

    if FlashInferState.mode == "prefill":
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


def _is_quantized_4bit(proj: nn.Module) -> bool:
    return _BNB_AVAILABLE and isinstance(proj, Linear4bit)


def _fuse_qkv_proj_plain(attn: LlamaAttention) -> nn.Linear:
    """Build a single nn.Linear from plain-precision
    """
    q_proj, k_proj, v_proj = attn.q_proj, attn.k_proj, attn.v_proj

    in_features = q_proj.in_features
    out_features = q_proj.out_features + k_proj.out_features + v_proj.out_features

    fused = nn.Linear(
        in_features,
        out_features,
        bias=False,
        device=q_proj.weight.device,
        dtype=q_proj.weight.dtype,
    )

    with torch.no_grad():
        fused.weight.copy_(
            torch.cat(
                [q_proj.weight, k_proj.weight, v_proj.weight],
                dim=0,
            )
        )

    return fused


def _fuse_qkv_proj_4bit(attn: LlamaAttention) -> "Linear4bit":
    """Build a single Linear4bit from quantized q_proj/k_proj/v_proj.
    """
    assert _BNB_AVAILABLE, "bitsandbytes not installed but q_proj is Linear4bit"

    q_proj, k_proj, v_proj = attn.q_proj, attn.k_proj, attn.v_proj

    assert q_proj.bias is None and k_proj.bias is None and v_proj.bias is None, (
        "q_proj/k_proj/v_proj have biases; fused QKV path assumes bias=False "
        "as in standard Llama."
    )

    ref_w = q_proj.weight
    quant_state = ref_w.quant_state

    def _dequant(proj: "Linear4bit") -> torch.Tensor:
        w = proj.weight
        return dequantize_4bit(
            w.data,
            quant_state=w.quant_state,
        ).to(quant_state.dtype)

    with torch.no_grad():
        q_f = _dequant(q_proj)
        k_f = _dequant(k_proj)
        v_f = _dequant(v_proj)
        fused_weight_fp = torch.cat([q_f, k_f, v_f], dim=0)
        del q_f, k_f, v_f
        torch.cuda.empty_cache()

    in_features = q_proj.in_features
    out_features = q_proj.out_features + k_proj.out_features + v_proj.out_features

    fused = Linear4bit(
        in_features,
        out_features,
        bias=False,
        compute_dtype=q_proj.compute_dtype,
        compress_statistics=ref_w.compress_statistics,
        quant_type=ref_w.quant_type,
        quant_storage=ref_w.quant_storage,
        device="meta",
    )

    fused.weight = Params4bit(
        data=fused_weight_fp,
        requires_grad=False,
        compress_statistics=ref_w.compress_statistics,
        quant_type=ref_w.quant_type,
        quant_storage=ref_w.quant_storage,
    )
    fused = fused.to(ref_w.device)

    return fused


def _fuse_qkv_proj(attn: LlamaAttention):
    if _is_quantized_4bit(attn.q_proj):
        return _fuse_qkv_proj_4bit(attn)
    return _fuse_qkv_proj_plain(attn)


def patch_llama_model(model):
    rope_theta = getattr(model.config, "rope_theta", 10000.0)
    rope_scaling = getattr(model.config, "rope_scaling", None)

    if rope_scaling is not None:
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
        if rope_type not in (None, "default", "linear"):
            raise NotImplementedError(
                f"model.config.rope_scaling type '{rope_type}' is not "
                "representable by flashinfer.apply_rope_pos_ids_inplace, "
                "which only supports a linear rope_scale factor (no NTK-aware, "
                "yarn, longrope, or Llama3.1-style dynamic scaling). Positions "
                "would be computed incorrectly if this patch is applied as-is."
            )
        rope_scale = rope_scaling.get("factor", 1.0)
    else:
        rope_scale = 1.0

    for idx, layer in enumerate(model.model.layers):
        attn: LlamaAttention = layer.self_attn

        attn.layer_idx = idx
        attn.num_heads = model.config.num_attention_heads
        attn.num_key_value_heads = model.config.num_key_value_heads
        attn.head_dim = model.config.hidden_size // model.config.num_attention_heads

        attn.q_size = attn.num_heads * attn.head_dim
        attn.kv_size = attn.num_key_value_heads * attn.head_dim

        attn.rope_theta = rope_theta
        attn.rope_scale = rope_scale
        attn.qkv_proj = _fuse_qkv_proj(attn)
        del attn.q_proj, attn.k_proj, attn.v_proj

        attn.forward = patched_llama_attention_forward.__get__(
            attn,
            LlamaAttention,
        )