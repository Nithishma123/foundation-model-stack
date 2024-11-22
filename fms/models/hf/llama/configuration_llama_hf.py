from typing import Optional

from transformers import PretrainedConfig

from fms.models.llama import LLaMAConfig


class HFAdaptedLLaMAConfig(PretrainedConfig):
    model_type = "hf_adapted_llama"
    attribute_map = {
        "vocab_size": "src_vocab_size",
        "hidden_size": "emb_dim",
        "num_attention_heads": "nheads",
        "num_hidden_layers": "nlayers",
    }

    def __init__(
        self,
        src_vocab_size: Optional[int] = 32000,
        emb_dim: Optional[int] = 4096,
        norm_eps: float = 1e-5,
        nheads: int = 32,
        kvheads: int = 0,
        nlayers: int = 32,
        # note this is different from the non-hf config (which is -1), hf keeps a different default
        pad_token_id: int = 0,
        hidden_grow_factor: float = 8 / 3,
        multiple_of: int = 256,
        activation_fn: str = "swish",
        p_dropout: float = 0.0,
        max_expected_seq_len: int = 2048,
        use_cache: bool = True,
        eos_token_id: int = 2,
        bos_token_id: int = 1,
        is_decoder: bool = True,
        dtensor_device_mesh: Optional[list] = None,
        dtensor_placements: Optional[list] = None,
        enable_sequence_parallelism: bool = False,
        sequence_parallel_shard_dim: int = 1,
        **kwargs,
    ):
        self.src_vocab_size = src_vocab_size
        self.emb_dim = emb_dim
        self.norm_eps = norm_eps
        self.nheads = nheads
        self.kvheads = kvheads
        self.nlayers = nlayers
        self.hidden_grow_factor = hidden_grow_factor
        self.multiple_of = multiple_of
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        self.max_expected_seq_len = max_expected_seq_len
        self.use_cache = use_cache
        self.dtensor_device_mesh = dtensor_device_mesh
        self.dtensor_placements = dtensor_placements
        self.enable_sequence_parallelism = enable_sequence_parallelism
        self.sequence_parallel_shard_dim = sequence_parallel_shard_dim
        super().__init__(
            pad_token_id=pad_token_id,
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
    def from_fms_config(cls, config: LLaMAConfig, **hf_kwargs):
        config_dict = config.as_dict()
        config_dict["pad_token_id"] = config_dict.pop("pad_id")
        config_dict["dtensor_device_mesh"] = getattr(config, "device_mesh", None)
        config_dict["dtensor_placements"] = getattr(config, "placements", None)
        config_dict["enable_sequence_parallelism"] = getattr(config, "enable_seq_parallelism", False)
        config_dict["sequence_parallel_shard_dim"] = getattr(config, "shard_dim", 1)
        return cls.from_dict(config_dict, **hf_kwargs)
