import torch as t
from tracr.compiler.assemble import AssembledTransformerModel
from transformer_lens import HookedTransformer


def generate_data(n_samples, vocab, max_seq_len):
    device = "cuda" if t.cuda.is_available() else "cpu"
    seq = t.randint(vocab, size=(n_samples, max_seq_len), device=device)
    return seq, t.sort(seq, dim=-1).values


def train_loader(n_samples, vocab, max_seq_len, batch_size):
    for _ in range(n_samples // batch_size):
        x, y = generate_data(batch_size, vocab, max_seq_len)
        yield x, y


def make_tracr_friendly(x):
    x = x.squeeze().tolist()
    return ['bos'] + x


def create_model_input(input, input_encoder):
    input = make_tracr_friendly(input)
    encoding = input_encoder.encode(input)
    return t.tensor(encoding).unsqueeze(dim=0)


def decode_model_output(logits, output_encoder):
    max_output_indices = logits.squeeze().argmax(-1)

    if isinstance(logits, t.Tensor):
        decoded_output = output_encoder.decode(max_output_indices.tolist())
    else:
        decoded_output = output_encoder.decode(max_output_indices)

    return decoded_output[1:]


def prompt(model, seq, input_encoder=None, output_encoder=None):
    if isinstance(model, HookedTransformer):

        if input_encoder:
            seq_valid = create_model_input(seq, input_encoder)
            logit, cache = model.run_with_cache(seq_valid)
            return decode_model_output(logit, input_encoder)
        else:
            return model(seq).argmax(-1).squeeze(0).tolist()

    elif isinstance(model, AssembledTransformerModel):
        bos_x = make_tracr_friendly(seq)
        pred_assembled = model.apply(bos_x)
        return pred_assembled.decoded[1:]


def run_with_activations(model, seq, input_encoder=None):

    if isinstance(model, HookedTransformer):

        if input_encoder:
            seq = create_model_input(seq, input_encoder)

        return model.run_with_cache(seq)

    elif isinstance(model, AssembledTransformerModel):
        bos_x = make_tracr_friendly(seq)
        pred_assembled = model.apply(bos_x)
        return pred_assembled.attn_logits, pred_assembled


