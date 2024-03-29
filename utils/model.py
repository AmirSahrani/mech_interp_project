import logging
import numpy as np
import einops

from transformer_lens import HookedTransformerConfig, HookedTransformer
from transformer_lens.HookedTransformer import Output
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache
from transformer_lens.utilities import devices
import transformer_lens.utils as utils
from transformer_lens.utils import USE_DEFAULT_VALUE

# from devinterp.utils
import torch as t
from torch import nn
from tqdm import tqdm

from typing import List, Optional, Tuple, Union
from jaxtyping import Float, Int
from typing_extensions import Literal

from utils.data import test_accuracy

SingleLoss = Float[t.Tensor, ""]  # Type alias for a single element tensor
LossPerToken = Float[t.Tensor, "batch pos-1"]
Loss = Union[SingleLoss, LossPerToken]


def cfg_from_tracr(assembled_model):
    """generate an empty model from the tracr model code taken from:
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
    # Activation function is a GeLu, this the standard activation for tracr as far as I can tell
    act_fn = "relu"
    normalization_type = "LN" if assembled_model.model_config.layer_norm else None
    attention_type = "causal" if assembled_model.model_config.causal else "bidirectional"

    n_ctx = assembled_model.params["pos_embed"]['embeddings'].shape[0]
    # Equivalent to length of vocab, with BOS and PAD at the end
    d_vocab = assembled_model.params["token_embed"]['embeddings'].shape[0]
    # Residual stream width, I don't know of an easy way to infer it from the above config.
    d_model = assembled_model.params["token_embed"]['embeddings'].shape[1]

    # Equivalent to length of vocab, WITHOUT BOS and PAD at the end because we never care about
    # these outputs. In practice, we always feed the logits into an argmax
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

    return cfg


def load_tracr_weights(tr_model, model, cfg):
    """
    Load the weights from the tracr model, code taken from:
    https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Tracr_to_Transformer_Lens_Demo.ipynb#scrollTo=bgM5a_Ct5k1V

    Args:
        tr_model: HookedTransformer, the empty model to which the weights will be loaded
        model: tracr model, the model from which the weights will be loaded
        cfg: configuration of the model

    Returns:
        HookedTransformer: model with weights from tracr model
    """

    n_heads = cfg.n_heads
    n_layers = cfg.n_layers
    d_head = cfg.d_head
    d_model = cfg.d_model
    d_vocab_out = cfg.d_vocab_out

    sd = {}
    sd["pos_embed.W_pos"] = model.params["pos_embed"]['embeddings']
    sd["embed.W_E"] = model.params["token_embed"]['embeddings']
    # Equivalent to max_seq_len plus one, for the BOS

    # The unembed is just a projection onto the first few elements of the residual stream, these store output tokens
    # This is a NumPy array, the rest are Jax Arrays, but w/e it's fine.
    sd["unembed.W_U"] = np.eye(d_model, d_vocab_out)
    sd["unembed.b_U"] = np.ones(d_vocab_out)

    for l in range(n_layers):
        sd[f"blocks.{l}.attn.W_K"] = einops.rearrange(
            model.params[f"transformer/layer_{l}/attn/key"]["w"],
            "d_model (n_heads d_head) -> n_heads d_model d_head",
            d_head=d_head,
            n_heads=n_heads
        )
        sd[f"blocks.{l}.attn.b_K"] = einops.rearrange(
            model.params[f"transformer/layer_{l}/attn/key"]["b"],
            "(n_heads d_head) -> n_heads d_head",
            d_head=d_head,
            n_heads=n_heads
        )
        sd[f"blocks.{l}.attn.W_Q"] = einops.rearrange(
            model.params[f"transformer/layer_{l}/attn/query"]["w"],
            "d_model (n_heads d_head) -> n_heads d_model d_head",
            d_head=d_head,
            n_heads=n_heads
        )
        sd[f"blocks.{l}.attn.b_Q"] = einops.rearrange(
            model.params[f"transformer/layer_{l}/attn/query"]["b"],
            "(n_heads d_head) -> n_heads d_head",
            d_head=d_head,
            n_heads=n_heads
        )
        sd[f"blocks.{l}.attn.W_V"] = einops.rearrange(
            model.params[f"transformer/layer_{l}/attn/value"]["w"],
            "d_model (n_heads d_head) -> n_heads d_model d_head",
            d_head=d_head,
            n_heads=n_heads
        )
        sd[f"blocks.{l}.attn.b_V"] = einops.rearrange(
            model.params[f"transformer/layer_{l}/attn/value"]["b"],
            "(n_heads d_head) -> n_heads d_head",
            d_head=d_head,
            n_heads=n_heads
        )
        sd[f"blocks.{l}.attn.W_O"] = einops.rearrange(
            model.params[f"transformer/layer_{l}/attn/linear"]["w"],
            "(n_heads d_head) d_model -> n_heads d_head d_model",
            d_head=d_head,
            n_heads=n_heads
        )
        sd[f"blocks.{l}.attn.b_O"] = model.params[f"transformer/layer_{l}/attn/linear"]["b"]

        sd[f"blocks.{l}.mlp.W_in"] = model.params[f"transformer/layer_{l}/mlp/linear_1"]["w"]
        sd[f"blocks.{l}.mlp.b_in"] = model.params[f"transformer/layer_{l}/mlp/linear_1"]["b"]
        sd[f"blocks.{l}.mlp.W_out"] = model.params[f"transformer/layer_{l}/mlp/linear_2"]["w"]
        sd[f"blocks.{l}.mlp.b_out"] = model.params[f"transformer/layer_{l}/mlp/linear_2"]["b"]

    for k, v in sd.items():
        # I cannot figure out a neater way to go from a Jax array to a numpy array
        sd[k] = t.tensor(np.array(v))

    tr_model.load_state_dict(sd, strict=False)

    for k, v in tr_model.state_dict().items():
        if k in sd.keys():
            assert v.shape == sd[k].shape, f'expected {sd[k].shape} but got {v.shape} for {k}'
            v = v.to(sd[k].dtype)
            assert t.allclose(v, sd[k].to(v.device)), f'difference between {k} and {v} in the model and the tracr model: {t.norm(v - sd[k].to(v.device))}'

    return tr_model


def train_model(model, optimizer, criterion, pentaly, train_loader, epochs, batch_size, input_size, len_vocab, bench_mark, save_path=None):
    """
    Trains a given model using the specified optimizer and criterion.

    Args:
        model (nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        criterion (torch.nn.Module): The loss function used for training.
        train_loader (DataLoader): The data loader for training data.
        epochs (int): The number of epochs to train the model.
        batch_size (int): The batch size used for training.
        input_size (int): The size of the input sequence.
        len_vocab (int): The length of the vocabulary.
        scheduler (torch.optim.lr_scheduler): Scheduler for the Learning rate
        save_path (str, optional): The file path to save the trained model. Defaults to None.
Returns:
        list: A list of losses at each epoch.
    """
    losses = []
    accuracy = []
    bench_mark_scores = [[], [], [], []]
    for epoch in tqdm(range(epochs)):
        for input_seq, target_seq in train_loader(batch_size, len_vocab, input_size, batch_size):
            optimizer.zero_grad()
            output, cache = model.run_with_cache(input_seq)
            loss = criterion(output.view(-1, output.shape[-1]), target_seq.view(-1))
            if pentaly[0] == "model":
                loss += pentaly[1](model)
            else:
                loss += pentaly[1](cache)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if save_path:
            if epoch == epochs - 1:
                t.save(model.state_dict(), save_path)

            path = save_path.split('/')
            path[0] = 'checkpoints/'
            path = ''.join(path)
            path += f'_{epoch}'
            t.save(model.state_dict(), path)

        if bench_mark:
            [bench_mark_scores[i].append(score) for i, score in enumerate(bench_mark(model, len_vocab, input_size))]
        accuracy.append(test_accuracy(model, len_vocab, input_size))

    return losses, accuracy, bench_mark_scores


class CompressionHook(nn.Module):
    """
    A class representing a compression hook for reducing the dimensionality of input data.

    Args:
        input_dim (int): The dimensionality of the input data.
        reduction_factor (int, optional): The reduction factor for compressing the input data. Default is 1.
        device (str, optional): The device to use for computation. Default is "cuda:0".

    Attributes:
        fc_compress (nn.Linear): The linear layer for compressing the input data.
        fc_decompress (nn.Linear): The linear layer for decompressing the compressed data.
        reduction_factor (int): The reduction factor for compressing the input data.
        weight (torch.Tensor): The weight matrix of the compression layer.

    Methods:
        compress(x): Compresses the input data.
        decompress(x): Decompresses the compressed data.
    """

    def __init__(self, input_dim, reduction_factor=1, device="cuda:0"):
        super(CompressionHook, self).__init__()

        # Use same number of output features as input for transpose-based decompression
        output_dim = input_dim

        self.fc_compress = nn.Linear(input_dim, int(output_dim / reduction_factor), bias=False)

        # Initialize weights for compression
        if reduction_factor != 1:
            t.nn.init.xavier_uniform_(self.fc_compress.weight)
        else:
            t.nn.init.eye_(self.fc_compress.weight)

        # Transpose of the compression matrix will be used for decompression
        self.fc_decompress = nn.Linear(
            int(output_dim / reduction_factor), output_dim, bias=False
        )

        # Initialize decompression matrix as the transpose of the compression matrix
        self.fc_decompress.weight.data = self.fc_compress.weight.data.T

        # Prevent the decompression weights from being updated during training
        # self.fc_decompress.weight.requires_grad = False

        self.fc_compress.requires_grad_(False)
        self.reduction_factor = reduction_factor
        self.to(device)
        self.weight = self.fc_compress.weight.data

    def compress(self, x):
        """
        Compresses the input data.

        Args:
            x (torch.Tensor): The input data to be compressed.

        Returns:
            torch.Tensor: The compressed data.
        """
        return self.fc_compress(x)

    def decompress(self, x):
        """
        Decompresses the compressed data.

        Args:
            x (torch.Tensor): The compressed data to be decompressed.

        Returns:
            torch.Tensor: The decompressed data.
        """
        return self.fc_decompress(x)


class HookedTransformer(HookedTransformer):
    """A class representing a hooked transformer model.

    This class extends the base `HookedTransformer` class and adds additional functionality for compression and caching.
    It also overwrites the `forward` method to include the compression hook.

    Args:
        cfg: The configuration object for the model.

    Attributes:
        compression: The compression hook for the model.
    """

    def __init__(self, cfg):
        super(HookedTransformer, self).__init__(cfg)
        self.compression = CompressionHook(cfg.d_model, 1, cfg.device)

    def forward(
        self,
        input: Union[
            str,
            List[str],
            Int[t.Tensor, "batch pos"],
            Float[t.Tensor, "batch pos d_model"],
        ],
        return_type: Optional[str] = "logits",
        loss_per_token: Optional[bool] = False,
        prepend_bos: Optional[Union[bool, None]] = USE_DEFAULT_VALUE,
        padding_side: Optional[Literal["left", "right"]] = USE_DEFAULT_VALUE,
        start_at_layer: Optional[int] = None,
        tokens: Optional[Int[t.Tensor, "batch pos"]] = None,
        shortformer_pos_embed: Optional[
            Float[t.Tensor, "batch pos d_model"]
        ] = None,
        attention_mask: Optional[t.Tensor] = None,  # [batch pos]
        stop_at_layer: Optional[int] = None,
        past_kv_cache: Optional[HookedTransformerKeyValueCache] = None,
    ) -> Union[
        None,
        Float[t.Tensor, "batch pos d_vocab"],
        Loss,
        Tuple[Float[t.Tensor, "batch pos d_vocab"], Loss],
    ]:
        """Forward Pass.

        Input is either a batch of tokens ([batch, pos]) or a text string, a string is automatically
        tokenized to a batch of a single element. The prepend_bos flag only applies when inputting a
        text string.

        Note that loss is the standard "predict the next token" cross-entropy loss for GPT-2 style
        language models - if you want a custom loss function, the recommended behaviour is returning
        the logits and then applying your custom loss function.

        Args:
            return_type Optional[str]: The type of output to return. Can be one of: None (return
                nothing, don't calculate logits), 'logits' (return logits), 'loss' (return
                cross-entropy loss), 'both' (return logits and loss).
            loss_per_token bool: Whether to return the (next token prediction) loss per token (True)
                or average (False). Average loss is a scalar (averaged over position *and* batch),
                per-token loss is a tensor ([batch, position-1]) - position-1 because we're
                predicting the next token, and there's no specified next token for the final token.
                Defaults to False.
            prepend_bos Optional[bool]: Overrides self.cfg.default_prepend_bos. Whether to prepend
                the BOS token to the input (only applies when input is a string). Defaults to None,
                implying usage of self.cfg.default_prepend_bos which is set to True unless specified
                otherwise. (Even for models not explicitly trained with a prepended BOS token, heads
                often use the first position as a resting position and accordingly lose information
                from the first token, so this empirically seems to give better results.) Pass True
                or False to locally override the default.
            padding_side Optional[Literal["left", "right"]]: Overrides self.tokenizer.padding_side.
                Specifies which side to pad on when tokenizing multiple strings of different
                lengths.
            start_at_layer Optional[int]: If not None, start the forward pass at the specified
                layer. Requires input to be the residual stream before the specified layer with
                shape [batch, pos, d_model]. Inclusive - ie, start_at_layer = 0 skips the embedding
                then runs the rest of the model. Supports negative indexing. start_at_layer = -1
                only runs the final block and the unembedding. Defaults to None (run the full
                model).
            tokens: Optional[Int[t.Tensor, "batch pos"]]: Tokenized input. Only use if
                start_at_layer is not None and return type is "loss" or "both".
            shortformer_pos_embed: Optional[Float[t.Tensor, "batch pos d_model"]]: Positional
                embedding for shortformer models. Only use if start_at_layer is not None and
                self.cfg.positional_embedding_type == "shortformer".
            attention_mask: Optional[t.Tensor]: The attention mask for padded tokens. Only use
                if start_at_layer is not None and (self.tokenizer.padding_side == "left" or
                past_kv_cache is not None).
            stop_at_layer Optional[int]: If not None, stop the forward pass at the specified layer.
                Exclusive - ie, stop_at_layer = 0 will only run the embedding layer, stop_at_layer =
                1 will run the embedding layer and the first transformer block, etc. Supports
                negative indexing. Useful for analysis of intermediate layers, eg finding neuron
                activations in layer 3 of a 24 layer model. Defaults to None (run the full model).
                If not None, we return the last residual stream computed.
            past_kv_cache Optional[HookedTransformerKeyValueCache]: If not None, keys and values
                will be stored for every attention head (unless the cache is frozen). If there are
                keys and values already in the cache, these will be prepended to the keys and values
                for the new input, so that the new tokens can pay attention to previous tokens. This
                is useful for generating text, because we don't need to repeat computation for
                tokens that have already been through the model. Also caches attention_mask so
                previous tokens are masked correctly (unless frozen). Padding should be ignored in
                all cases, so it's okay to eg. pass in left padded tokens twice in a row.
                Warning: Don't accidently prepend_bos to the second half of a prompt.
                Defaults to None (don't use caching).
        """

        with utils.LocallyOverridenDefaults(
            self, prepend_bos=prepend_bos, padding_side=padding_side
        ):
            if start_at_layer is None:
                (
                    residual,
                    tokens,
                    shortformer_pos_embed,
                    attention_mask,
                ) = self.input_to_embed(
                    input,
                    prepend_bos=prepend_bos,
                    padding_side=padding_side,
                    past_kv_cache=past_kv_cache,
                )
                residual = self.compression.compress(residual)
            else:
                assert type(input) is t.Tensor
                residual = self.compression.compress(input)

            if start_at_layer is None:
                start_at_layer = 0
            # If we explicitly want to start or stop at a layer, we only iterate through the blocks
            # between those indices. Note that start_at_layer is inclusive and stop_at_layer is
            # exclusive.
            # Eg: start_at_layer==None + stop_at_layer==0 means to only run the embed.
            # Eg: start_at_layer==3 + stop_at_layer==-1 means to run from layer 3 until the end of the PENULTIMATE layer
            blocks_and_idxs = list(zip(range(self.cfg.n_layers), self.blocks))
            for i, block in blocks_and_idxs[start_at_layer:stop_at_layer]:  # type: ignore
                # Note that each block includes skip connections, so we don't need
                # residual + block(residual)
                # If we're using multiple GPUs, we need to send the residual and shortformer_pos_embed to the correct GPU
                residual = residual.to(devices.get_device_for_block_index(i, self.cfg))
                if shortformer_pos_embed is not None:
                    shortformer_pos_embed = shortformer_pos_embed.to(
                        devices.get_device_for_block_index(i, self.cfg)
                    )

                residual = self.compression.compress(block(
                    self.compression.decompress(residual),
                    # Cache contains a list of HookedTransformerKeyValueCache objects, one for each
                    # block
                    past_kv_cache_entry=past_kv_cache[i]
                    if past_kv_cache is not None
                    else None,
                    shortformer_pos_embed=shortformer_pos_embed,
                    attention_mask=attention_mask,
                ))  # [batch, pos, d_model]

            if stop_at_layer is not None:
                # When we stop at an early layer, we end here rather than doing further computation
                return self.compression.compress(residual)

            if self.cfg.normalization_type is not None:
                residual = self.compression.compress(self.ln_final(self.compression.decompress(residual)))  # [batch, pos, d_model]
            if return_type is None:
                return None

            else:
                logits = self.unembed(self.compression.decompress(residual))  # [batch, pos, d_vocab]
                if return_type == "logits":
                    return logits
                else:
                    assert (
                        tokens is not None
                    ), "tokens must be passed in if return_type is 'loss' or 'both'"
                    loss = self.loss_fn(logits, tokens, per_token=loss_per_token)
                    if return_type == "loss":
                        return loss
                    elif return_type == "both":
                        return Output(logits, loss)
                    else:
                        logging.warning(f"Invalid return_type passed in: {return_type}")
                        return None


def l1_pentaly(lamb):
    return "model", lambda model: lamb * sum(p.abs().sum() for p in model.parameters() if p.requires_grad)


def l1_pentaly_activation(lamb):
    return "cache", lambda cache: lamb / 1000 * sum(t.min(t.abs(0 - p), 1 - t.abs(p)).sum() for k, p in cache.items() if 'hook_resid_post' in k)


def no_pentaly(lamb):
    return "model", lambda model: 0


def transformers_cross_entropy(inputs, outputs):
    return t.nn.functional.cross_entropy(
        inputs, outputs, reduce=True, reduction='mean'
    )
