from typing import Union
import torch
from torch.distributed._tensor import distribute_tensor, DeviceMesh, Shard
from transformers import GPTBigCodeConfig, GPTBigCodeForCausalLM, PreTrainedModel
from fms.models.hf.gpt_bigcode.modeling_gpt_bigcode_hf import HFAdaptedGPTBigCodeForCausalLM

def convert_to_hf(
    fms_hf_model: HFAdaptedGPTBigCodeForCausalLM,
    device_mesh: DeviceMesh,
    placements: list,
) -> GPTBigCodeForCausalLM:
    """
    Convert an HF-Adapted FMS GPTBigCode model to an HF model with DTensor support.

    Parameters
    ----------
    fms_hf_model: HFAdaptedGPTBigCodeForCausalLM
        The HF-Adapted FMS GPTBigCode model.
    device_mesh: DeviceMesh
        Device mesh to distribute tensors across devices.
    placements: list
        List of DTensor placements (e.g., Shard(0), Shard(1), Replicate()).

    Returns
    -------
    GPTBigCodeForCausalLM
        A DTensor-distributed HF equivalent model.
    """
    hf_config = fms_hf_model.config
    oss_hf_model = GPTBigCodeForCausalLM(
        GPTBigCodeConfig(
            vocab_size=hf_config.vocab_size,
            n_embd=hf_config.hidden_size,
            layer_norm_epsilon=hf_config.ln_eps,
            n_head=hf_config.nheads,
            n_layer=hf_config.nlayers,
            n_inner=int(hf_config.hidden_size * hf_config.hidden_grow_factor),
            pad_token_id=hf_config.pad_token_id,
            bos_token_id=hf_config.bos_token_id,
            eos_token_id=hf_config.eos_token_id,
            n_positions=hf_config.max_expected_seq_len,
            scale_attention_softmax_in_fp32=False,
        )
    )
    with torch.no_grad():
        oss_hf_model.transformer.wte.weight = distribute_tensor(
            fms_hf_model.decoder.model.embedding.weight,
            device_mesh=device_mesh,
            placements=placements,
        )
        oss_hf_model.transformer.wpe.weight = distribute_tensor(
            fms_hf_model.decoder.model.position_embedding.weight,
            device_mesh=device_mesh,
            placements=placements,
        )
        for i, oss_hf_layer in enumerate(oss_hf_model.transformer.h):
            fms_hf_layer = fms_hf_model.decoder.model.layers[i]
            oss_hf_layer.attn.c_attn.weight = distribute_tensor(
                fms_hf_layer.attn.in_proj.qkv_fused.weight,
                device_mesh=device_mesh,
                placements=placements,
            )
            oss_hf_layer.attn.c_attn.bias = distribute_tensor(
                fms_hf_layer.attn.in_proj.qkv_fused.bias,
                device_mesh=device_mesh,
                placements=placements,
            )
            oss_hf_layer.attn.c_proj.weight = distribute_tensor(
                fms_hf_layer.attn.dense.weight,
                device_mesh=device_mesh,
                placements=placements,
            )
            oss_hf_layer.attn.c_proj.bias = distribute_tensor(
                fms_hf_layer.attn.dense.bias,
                device_mesh=device_mesh,
                placements=placements,
            )
            oss_hf_layer.mlp.c_fc.weight = distribute_tensor(
                fms_hf_layer.ff_sub_layer.w1.weight,
                device_mesh=device_mesh,
                placements=placements,
            )
            oss_hf_layer.mlp.c_fc.bias = distribute_tensor(
                fms_hf_layer.ff_sub_layer.w1.bias,
                device_mesh=device_mesh,
                placements=placements,
            )
            oss_hf_layer.mlp.c_proj.weight = distribute_tensor(
                fms_hf_layer.ff_sub_layer.w2.weight,
                device_mesh=device_mesh,
                placements=placements,
            )
            oss_hf_layer.mlp.c_proj.bias = distribute_tensor(
                fms_hf_layer.ff_sub_layer.w2.bias,
                device_mesh=device_mesh,
                placements=placements,
            )
            oss_hf_layer.ln_1.weight = distribute_tensor(
                fms_hf_layer.ln.weight, device_mesh=device_mesh, placements=placements
            )
            oss_hf_layer.ln_1.bias = distribute_tensor(
                fms_hf_layer.ln.bias, device_mesh=device_mesh, placements=placements
            )
            oss_hf_layer.ln_2.weight = distribute_tensor(
                fms_hf_layer.ff_ln.weight, device_mesh=device_mesh, placements=placements
            )
            oss_hf_layer.ln_2.bias = distribute_tensor(
                fms_hf_layer.ff_ln.bias, device_mesh=device_mesh, placements=placements
            )

        oss_hf_model.transformer.ln_f.weight = distribute_tensor(
            fms_hf_model.decoder.model.dec_norm.weight,
            device_mesh=device_mesh,
            placements=placements,
        )
        oss_hf_model.transformer.ln_f.bias = distribute_tensor(
            fms_hf_model.decoder.model.dec_norm.bias,
            device_mesh=device_mesh,
            placements=placements,
        )
    return oss_hf_model
