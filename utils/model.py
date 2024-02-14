import numpy as np
import einops
from transformer_lens import HookedTransformerConfig, HookedTransformer
import torch as t


def empty_model_from_tracr(assembled_model):
    """generate an empty model from the tracr model
    code taken from: 
    https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Tracr_to_Transformer_Lens_Demo.ipynb#scrollTo=bgM5a_Ct5k1V
    

    Args:
        assembled_model: Tracr model

    Returns:
        HookedTransformer: empty model
    """
    
    n_heads = assembled_model.model_config.num_heads
    n_layers = assembled_model.model_config.num_layers
    d_head = assembled_model.model_config.key_size
    d_mlp = assembled_model.model_config.mlp_hidden_size
    act_fn = "relu"
    normalization_type = "LN"  if assembled_model.model_config.layer_norm else None
    attention_type = "causal"  if assembled_model.model_config.causal else "bidirectional"

    n_ctx = assembled_model.params["pos_embed"]['embeddings'].shape[0]
    # Equivalent to length of vocab, with BOS and PAD at the end
    d_vocab = assembled_model.params["token_embed"]['embeddings'].shape[0]
    # Residual stream width, I don't know of an easy way to infer it from the above config.
    d_model = assembled_model.params["token_embed"]['embeddings'].shape[1]

    # Equivalent to length of vocab, WITHOUT BOS and PAD at the end because we never care about these outputs
    # In practice, we always feed the logits into an argmax
    d_vocab_out = assembled_model.params["token_embed"]['embeddings'].shape[0] - 2

    cfg = HookedTransformerConfig(
        n_layers=n_layers,
        d_model=d_model,
        d_head=d_head,
        n_ctx=n_ctx,
        d_vocab=d_vocab,
        d_vocab_out=d_vocab_out,
        d_mlp=d_mlp,
        n_heads=n_heads,
        act_fn=act_fn,
        attention_dir=attention_type,
        normalization_type=normalization_type,
    )

    return HookedTransformer(cfg), cfg
def load_tracr_weights(model, tracr_model, cfg):
    """
    Load the weights from the tracr model, code taken from:
    https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Tracr_to_Transformer_Lens_Demo.ipynb#scrollTo=bgM5a_Ct5k1V
    """

    n_heads = cfg.n_heads
    n_layers = cfg.n_layers
    d_head = cfg.d_head
    d_model = cfg.d_model
    d_vocab_out = cfg.d_vocab_out


    sd = {}
    sd["pos_embed.W_pos"] =tracr_model.params["pos_embed"]['embeddings']
    sd["embed.W_E"] =tracr_model.params["token_embed"]['embeddings']
    # Equivalent to max_seq_len plus one, for the BOS

    # The unembed is just a projection onto the first few elements of the residual stream, these store output tokens
    # This is a NumPy array, the rest are Jax Arrays, but w/e it's fine.
    sd["unembed.W_U"] = np.eye(d_model, d_vocab_out)

    for l in range(n_layers):
        sd[f"blocks.{l}.attn.W_K"] = einops.rearrange(
        tracr_model.params[f"transformer/layer_{l}/attn/key"]["w"],
            "d_model (n_heads d_head) -> n_heads d_model d_head",
            d_head = d_head,
            n_heads = n_heads
        )
        sd[f"blocks.{l}.attn.b_K"] = einops.rearrange(
        tracr_model.params[f"transformer/layer_{l}/attn/key"]["b"],
            "(n_heads d_head) -> n_heads d_head",
            d_head = d_head,
            n_heads = n_heads
        )
        sd[f"blocks.{l}.attn.W_Q"] = einops.rearrange(
        tracr_model.params[f"transformer/layer_{l}/attn/query"]["w"],
            "d_model (n_heads d_head) -> n_heads d_model d_head",
            d_head = d_head,
            n_heads = n_heads
        )
        sd[f"blocks.{l}.attn.b_Q"] = einops.rearrange(
        tracr_model.params[f"transformer/layer_{l}/attn/query"]["b"],
            "(n_heads d_head) -> n_heads d_head",
            d_head = d_head,
            n_heads = n_heads
        )
        sd[f"blocks.{l}.attn.W_V"] = einops.rearrange(
        tracr_model.params[f"transformer/layer_{l}/attn/value"]["w"],
            "d_model (n_heads d_head) -> n_heads d_model d_head",
            d_head = d_head,
            n_heads = n_heads
        )
        sd[f"blocks.{l}.attn.b_V"] = einops.rearrange(
        tracr_model.params[f"transformer/layer_{l}/attn/value"]["b"],
            "(n_heads d_head) -> n_heads d_head",
            d_head = d_head,
            n_heads = n_heads
        )
        sd[f"blocks.{l}.attn.W_O"] = einops.rearrange(
        tracr_model.params[f"transformer/layer_{l}/attn/linear"]["w"],
            "(n_heads d_head) d_model -> n_heads d_head d_model",
            d_head = d_head,
            n_heads = n_heads
        )
        sd[f"blocks.{l}.attn.b_O"] =tracr_model.params[f"transformer/layer_{l}/attn/linear"]["b"]

        sd[f"blocks.{l}.mlp.W_in"] =tracr_model.params[f"transformer/layer_{l}/mlp/linear_1"]["w"]
        sd[f"blocks.{l}.mlp.b_in"] =tracr_model.params[f"transformer/layer_{l}/mlp/linear_1"]["b"]
        sd[f"blocks.{l}.mlp.W_out"] =tracr_model.params[f"transformer/layer_{l}/mlp/linear_2"]["w"]
        sd[f"blocks.{l}.mlp.b_out"] =tracr_model.params[f"transformer/layer_{l}/mlp/linear_2"]["b"]
        
    for k, v in sd.items():
        # I cannot figure out a neater way to go from a Jax array to a numpy array lol
        sd[k] = t.tensor(np.array(v))

    model.load_state_dict(sd, strict=False)
    return model