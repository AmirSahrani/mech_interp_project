from circuitsvis.attention import attention_patterns
from functools import partial

from transformers.utils import to_numpy

import torch as t
import matplotlib.pyplot as plt
from IPython.display import display

from utils.data import run_with_activations


def logit_difference(logits, logit_corruprted):
    return t.mean(logits - logit_corruprted)


def kl_divergence(logits, logit_corrupted):
    return t.nn.functional.kl_div(t.nn.functional.log_softmax(logits, dim=1), t.nn.functional.softmax(logit_corrupted, dim=1))


def patch(model, tokens_corrupt, tokens_clean, measure, patching):
    logits, acti = run_with_activations(model, tokens_clean)
    divergence = partial(measure, logits)
    return patching(model, tokens_corrupt, acti, divergence)


def plot(patch, title):
    max_abs_value = t.max(t.abs(patch)).item()
    plt.imshow(to_numpy(patch).squeeze(), vmax=max_abs_value, vmin=-max_abs_value)
    plt.title(title)
    plt.xlabel('Token index')
    plt.ylabel('Layer index')
    plt.colorbar()
    plt.show()


def plot_patches(corrputed, clean, model, measure, names, patching):
    n_combination = len(corrputed)
    fig, ax = plt.subplots(1, n_combination, figsize=(5 * n_combination, 10))
    for i, (corr, cln) in enumerate(zip(corrputed, clean)):
        data = patch(model, corr, cln, measure, patching)
        max_abs_value = t.max(t.abs(data)).item()
        ax[i].imshow(to_numpy(data).squeeze(), vmax=max_abs_value, vmin=-max_abs_value)
        ax[i].set_title(f'Patch for {names[i]} corrupted tokens')
        ax[i].set_xlabel('Token index')
        ax[i].set_ylabel('Layer index')


def vis_attn(cache, tokens, name):
    for i in range(3):
        attention = to_numpy(cache['pattern', i].squeeze(0))
        tokens = to_numpy(tokens).squeeze()

        html = attention_patterns(attention, list(map(str, tokens)), f'Layer {i} - {name} tokens')
        display(html)


def get_layer_1_activations(model, tokens):
    _, cache = run_with_activations(model, tokens)
    layer_1_cache = dict(filter(lambda x: '0' in x[0] or 'embed' in x[0], cache.items()))
    layer_1_cache['logits'] = model.unembed(layer_1_cache['blocks.0.hook_resid_post'])
    return layer_1_cache


def label_to_title(label):
    # TODO: Fixed labels, https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/transformer-full-updated.png
    clean_dicts = {
        'resid': 'Residual Stream',
        'mlp': 'MLP',
        'pre': 'Residual Stream before Attention',
        'mid': 'Residual Stream before MLP',
        'post': 'Output',
        'logits': 'Logits',
        'attn': 'Attention',
        'v': 'Value',
        'q': 'Query',
        'k': 'Key',
        'z': 'Hidden State',
        'pattern': 'Attention Pattern',
        'scores': 'Attention Scores',
        'attn_out': 'Attention Output',
        'mlp_out': 'MLP after Activation Fuction',
        'hook_embed': 'Embedding',
        'hook_pos_embed': 'Positional Embedding'
    }

    label = label.split('.')
    activation = label[-1]
    activation = activation.split('_')
    if len(label) == 1:
        return clean_dicts.get(label[0], label[0])

    else:
        if activation[-1] == 'out':
            activation_type = clean_dicts.get('_'.join(activation[-2:]), activation[-2:])

        else:
            activation_type = clean_dicts.get(activation[-1], activation[-1].capitalize())
        if activation_type == "Output":
            activation_type = clean_dicts.get(activation[-2], 'MLP') + ' ' + activation_type
        if len(label) > 2:
            layer_index = int(label[1]) + 1
            return f"Layer {layer_index} {activation_type}"
        else:
            return f"Layer {activation_type}"


def gen_embedding_dict(model, tokens):
    _, cache = run_with_activations(model, tokens)
    embeddings = to_numpy(cache['hook_embed']).squeeze()
    embedding_dict = {i: embeddings[i, :] for i in range(10)}
    return embedding_dict
