from devinterp.optim import SGLD
from devinterp.slt import estimate_learning_coeff_with_summary
from transformer_lens import HookedTransformer

from tracr.compiler import compiling
from tracr.compiler import lib
from tracr.rasp import rasp

import torch as t
import glob
import pandas as pd
import multiprocessing
from functools import partial

if __name__ == "__main__":
    from model import cfg_from_tracr, transformers_cross_entropy
else:
    from utils.model import cfg_from_tracr, transformers_cross_entropy


class SortDataset(t.utils.data.Dataset):
    def __init__(self, input_size, num_samples, vocab):
        self.input_size = input_size
        self.num_samples = num_samples
        self.vocab = vocab

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        device = "cuda" if t.cuda.is_available() else "cpu"
        seq = t.randint(self.vocab, size=(self.input_size,), device=device).squeeze()
        return seq, t.sort(seq, dim=-1).values


def load_model(file, cfg):
    model = HookedTransformer(cfg)
    model.load_state_dict(t.load(file), strict=False)
    return model


def get_llc(model):
    loader = t.utils.data.DataLoader(
        SortDataset(10, 100, 10), batch_size=10, shuffle=True
    )

    learning_coeff_stats = estimate_learning_coeff_with_summary(
        model,
        loader=loader,
        criterion=transformers_cross_entropy,
        sampling_method=SGLD,
        optimizer_kwargs=dict(lr=2e-4, localization=100.0),
        num_chains=3,  # How many independent chains to run
        num_draws=500,  # How many samples to draw per chain
        num_burnin_steps=30,  # How many samples to discard at the beginning of each chain
        num_steps_bw_draws=1,  # How many steps to take between each sample
        device='cuda',
        online=True
    )
    return learning_coeff_stats['llc/moving_avg'].mean()


def get_cfg():
    input_size = 10
    vocab = {*range(input_size)}
    program = lib.make_sort(rasp.tokens, rasp.tokens, max_seq_len=input_size, min_key=0)

    assembled_model = compiling.compile_rasp_to_model(
        program=program,
        vocab=vocab,
        max_seq_len=input_size,
        compiler_bos="bos",
        mlp_exactness=100)
    return cfg_from_tracr(assembled_model)


def get_llc_for_model(model, cfg):
    return (model, get_llc(load_model(model, cfg)))


def main():
    checkpoints = glob.glob('checkpoints/*')

    cfg = get_cfg()
    llc_values = []

    # Define a pool of processes
    multiprocessing.set_start_method('spawn')
    pool = multiprocessing.Pool()
    model_llc = partial(get_llc_for_model, cfg=cfg)

    # Run get_llc for each checkpoint using parallel processing
    results = pool.map(
        model_llc,
        checkpoints)

    # Close the pool to free up resources
    pool.close()
    pool.join()

    # Store the results in the llc_values array
    for i, result in enumerate(results):
        llc_values.append(result)
    df = pd.DataFrame(llc_values, columns=['model', 'llc_value'])
    df.to_csv('data/llc_values.csv', index=False)
    print(df)


if __name__ == "__main__":
    main()
