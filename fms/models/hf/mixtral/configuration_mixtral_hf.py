from typing import Optional

from transformers import PretrainedConfig

from fms.models.mixtral import MixtralConfig


class HFAdaptedMixtralConfig(PretrainedConfig):
    model_type = "hf_adapted_mixtral"
    attribute_map = {
        "vocab_size": "src_vocab_size",
        "hidden_size": "dim",
        "num_attention_heads": "nheads",
        "num_hidden_layers": "nlayers",
        "num_key_value_heads": "kvheads",
        "num_local_experts": "num_experts",
        "num_experts_per_tok": "top_k_experts",
        "intermediate_size": "hidden_dim",
        "rms_norm_eps": "norm_eps",
        "max_position_embeddings": "max_expected_seq_len",
        "rope_theta": "rope_base",
        "attention_dropout": "p_dropout",
    }

    attention_dropout = 0.0

    def __init__(
        self,
        src_vocab_size: Optional[int] = 32000,
        dim: Optional[int] = 4096,
        hidden_dim: Optional[int] = 14336,
        nlayers: int = 32,
        nheads: int = 32,
        kvheads: int = 8,
        num_experts: int = 8,
        top_k_experts: int = 2,
        max_expected_seq_len: int = 32768,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        p_dropout: float = 0.0,
        norm_eps: float = 1e-5,
        rope_base: float = 1000000.0,
        use_cache: bool = True,
        is_decoder: bool = True,
        dtensor_device_mesh: Optional[list] = None,
        dtensor_placements: Optional[list] = None,
        enable_sequence_parallelism: bool = False,
        sequence_parallel_shard_dim: int = 1,
        **kwargs,
    ):
        self.src_vocab_size = src_vocab_size
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.norm_eps = norm_eps
        self.nheads = nheads
        self.kvheads = kvheads
        self.nlayers = nlayers
        self.num_experts = num_experts
        self.top_k_experts = top_k_experts
        self.rope_base = rope_base
        self.p_dropout = p_dropout
        self.max_expected_seq_len = max_expected_seq_len
        self.use_cache = use_cache
        self.dtensor_device_mesh = dtensor_device_mesh
        self.dtensor_placements = dtensor_placements
        self.enable_sequence_parallelism = enable_sequence_parallelism
        self.sequence_parallel_shard_dim = sequence_parallel_shard_dim

        super().__init__(
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            is_decoder=is_decoder,
            tie_word_embeddings=kwargs.pop(
                "tie_word_embeddings", False
            ),  # note: This was added here as we handle tying of heads with our underlying model, we may want to revisit this in future
            **kwargs,
        )

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, **kwargs
    ) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs
        )

        return cls.from_dict(config_dict, **kwargs)

    @classmethod
    def from_fms_config(cls, config: MixtralConfig, **hf_kwargs):
        config_dict = config.as_dict()
        config_dict["dtensor_device_mesh"] = getattr(config, "device_mesh", None)
        config_dict["dtensor_placements"] = getattr(config, "placements", None)
        config_dict["enable_sequence_parallelism"] = getattr(config, "enable_seq_parallelism", False)
        config_dict["sequence_parallel_shard_dim"] = getattr(config, "shard_dim", 1)
        return cls.from_dict(config_dict, **hf_kwargs)
