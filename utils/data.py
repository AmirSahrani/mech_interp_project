import torch as t


def generate_data(n_samples, vocab, max_seq_len):
    device = "cuda" if t.cuda.is_available() else "cpu"
    seq = t.randint(vocab, size=(n_samples, max_seq_len), device=device)
    return seq, t.sort(seq, dim=-1).values


def train_loader(n_samples, vocab, max_seq_len, batch_size):
    for _ in range(n_samples // batch_size):
        x, y = generate_data(batch_size, vocab, max_seq_len)
        yield x, y