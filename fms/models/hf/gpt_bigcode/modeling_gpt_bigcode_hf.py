from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.distributed._tensor import distribute_tensor, DeviceMesh, Shard
from transformers import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from fms.models.gpt_bigcode import GPTBigCode, GPTBigCodeHeadless
from fms.models.hf.gpt_bigcode.configuration_gpt_bigcode_hf import HFAdaptedGPTBigCodeConfig
from fms.models.hf.lm_head_mixins import LMHeadModelLMHeadMixin
from fms.models.hf.modeling_hf_adapter import HFDecoder, HFDecoderModelArchitecture


class HFAdaptedGPTBigCodeDecoder(HFDecoder):
    def __init__(self, model: nn.Module, config: PretrainedConfig):
        super().__init__(model, config, attention_mask_dim=3)

    def _adapt(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        attn_algorithm: Optional[str] = None,
        enable_sequence_parallelism: bool = False,
        shard_dim: int = 1,
        *args,
        **kwargs,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        if enable_sequence_parallelism and input_ids is not None:
            shard_size = input_ids.size(shard_dim) // torch.distributed.get_world_size()
            input_ids = torch.split(input_ids, shard_size, dim=shard_dim)[torch.distributed.get_rank()]

        output, cache = self.model(
            x=input_ids,
            mask=attention_mask,
            position_ids=position_ids,
            past_key_value_states=past_key_values,
            use_cache=use_cache,
            attn_algorithm=attn_algorithm,
        )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=output, past_key_values=cache
        )


class HFAdaptedGPTBigCodeHeadless(HFDecoderModelArchitecture):
    config_class = HFAdaptedGPTBigCodeConfig
    base_model_prefix = "hf_adapted_gpt_bigcode"

    def __init__(self, config: PretrainedConfig, *args, **kwargs):
        device_mesh = DeviceMesh("cuda", torch.arange(torch.cuda.device_count()))
        placement = [Shard(0)]

        if "embedding" in kwargs:
            kwargs["embedding"] = distribute_tensor(kwargs["embedding"], device_mesh, placement)
        super().__init__(*args, config=config, **kwargs)


class HFAdaptedGPTBigCodeForCausalLM(LMHeadModelLMHeadMixin, HFAdaptedGPTBigCodeHeadless):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config: HFAdaptedGPTBigCodeConfig, *args, **kwargs):
        super().__init__(config=config, *args, **kwargs)

    def _tie_weights(self):
        device_mesh = DeviceMesh("cuda", torch.arange(torch.cuda.device_count()))
        placement = [Shard(0)]
        self.embedding.weight = distribute_tensor(self.lm_head.weight, device_mesh, placement)
        self.decoder.model.embedding.weight = self.embedding.weight
