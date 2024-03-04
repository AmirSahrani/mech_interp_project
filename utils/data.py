import torch as t
from tracr.compiler.assemble import AssembledTransformerModel
from transformer_lens import HookedTransformer


def generate_data(n_samples, vocab, max_seq_len):
    """
    Generate random sequences of integers.

    Args:
        n_samples (int): The number of sequences to generate.
        vocab (list): A list of integers representing the vocabulary.
        max_seq_len (int): The maximum length of each sequence.

    Returns:
        list: A list of randomly generated sequences.

    """
    device = "cuda" if t.cuda.is_available() else "cpu"
    seq = t.randint(vocab, size=(n_samples, max_seq_len), device=device)
    return seq, t.sort(seq, dim=-1).values


def train_loader(n_samples, vocab, max_seq_len, batch_size):
    """
    Generates batches of training data.

    Args:
        n_samples (int): Total number of samples to generate.
        vocab (list): List of vocabulary words.
        max_seq_len (int): Maximum sequence length.
        batch_size (int): Number of samples per batch.

    Yields:
        tuple: A tuple containing the input data (x) and the target data (y).
    """
    for _ in range(n_samples // batch_size):
        x, y = generate_data(batch_size, vocab, max_seq_len)
        yield x, y


def make_tracr_friendly(x):
    """
    Function to make a Tensor into a list that Tracr can deal with.

    Args:
        x (Tensor): 1D tensor to be converted to a list.

    Returns:
        List: List with bos token added to the beginning of the input.
    """
    x = x.squeeze().tolist()
    return ['bos'] + x


def create_model_input(input, input_encoder):
    """
    Create a model input tensor from the given input using the specified input encoder.

    Args:
        input (str): The input string to encode.
        input_encoder (Encoder): The input encoder object.

    Returns:
        torch.Tensor: The model input tensor.
    """
    input = make_tracr_friendly(input)
    encoding = input_encoder.encode(input)
    return t.tensor(encoding).unsqueeze(dim=0)


def decode_model_output(logits, output_encoder):
    """
    !!!! Might the the reason the model is not working properly.!!!
    Decode the model output tensor using the specified output encoder.

    Args:
        logits (torch.Tensor): The model output tensor.
        output_encoder (Encoder): The output encoder object.

    Returns:
        str: The decoded output string.    
    """
    max_output_indices = logits.squeeze().argmax(-1)

    if isinstance(logits, t.Tensor):
        decoded_output = output_encoder.decode(max_output_indices.tolist())
    else:
        decoded_output = output_encoder.decode(max_output_indices)

    return decoded_output[1:]


def prompt(model, seq, input_encoder=None, output_encoder=None):
    """
    Prompts the given model with a sequence and returns the predicted output. 
    If the model is a copy of the tracr model, the input and output encoders must be provided.

    Args:
        model (Union[HookedTransformer, AssembledTransformerModel]): The model to be prompted.
        seq (str): The input sequence.
        input_encoder (Optional[Encoder]): The input encoder to be used for encoding the sequence.
        output_encoder (Optional[Encoder]): The output encoder to be used for decoding the predicted output.

    Returns:
        Union[List[int], str]: The predicted output as a list of integers or a decoded string.
    """
    if isinstance(model, HookedTransformer):

        if input_encoder:
            seq_valid = create_model_input(seq, input_encoder)
            logit, cache = model.run_with_cache(seq_valid)
            return decode_model_output(logit, output_encoder)
        else:
            return model(seq).argmax(-1).squeeze(0).tolist()

    elif isinstance(model, AssembledTransformerModel):
        bos_x = make_tracr_friendly(seq)
        pred_assembled = model.apply(bos_x)
        return pred_assembled.decoded[1:]


def run_with_activations(model, seq, input_encoder=None):
    """
    Runs the model with the given sequence and returns the activations.

    Args:
        model (HookedTransformer or AssembledTransformerModel): The model to run.
        seq (str): The input sequence.
        input_encoder (Optional): The input encoder to use for creating the model input.

    Returns:
        If the model is a HookedTransformer:
            - If input_encoder is provided, returns the activations of the model with the encoded input sequence.
            - If input_encoder is not provided, returns the activations of the model with the original input sequence.
        If the model is an AssembledTransformerModel:
            - Returns the attention logits and the predictions of the model with the input sequence.
    """
    if isinstance(model, HookedTransformer):
        if input_encoder:
            seq = create_model_input(seq, input_encoder)
        return model.run_with_cache(seq)
    elif isinstance(model, AssembledTransformerModel):
        bos_x = make_tracr_friendly(seq)
        pred_assembled = model.apply(bos_x)
        return pred_assembled.attn_logits, pred_assembled


def test_accuracy(model, vocab, input_size, input_encoder=None, output_encoder=None):
    model.eval()
    correct = 0
    total = 0
    with t.no_grad():
        for i, (x, y) in enumerate(train_loader(1000, input_size, vocab, 1)):
            predicted = prompt(model, x, input_encoder, output_encoder)

            total += y.size(1)
            correct += sum((pred == orig for pred, orig in zip(predicted, y.tolist()[0])))
    return correct / total * 100
