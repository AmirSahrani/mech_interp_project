import matplotlib.pyplot as plt
import numpy as np


def plot_activations(titles, *activations):
    """
    Plot activations.

    Args:
        titles (list): List of titles for each activation.
        *activations (ndarray): Variable number of activation arrays.

    Raises:
        AssertionError: If the number of dimensions is not between 2 and 3.

    Returns:
        None
    """

    shape = np.array(activations[0].shape)
    non_one_dims = shape > 1
    num_act = len(activations)
    assert 2 <= sum(non_one_dims) <= 3, "Wrong number of dims, can't automatically handle"

    if len(shape) > 2:
        n_figures = shape[0]
    else:
        n_figures = 1

    figure, axes = plt.subplots(num_act, n_figures, figsize=(num_act * 11, n_figures * 10))

    for fig in range(n_figures):
        # Case that there is only one set of activations to plot
        if num_act == 1:
            max_v = np.max(activations[0][fig, :, :].squeeze())
            im = axes[fig].imshow(activations[0][fig, :, :].squeeze(), vmin=-max_v, vmax=max_v)
            axes[fig].set_title(titles[fig])
            figure.colorbar(im)

        # Case that there is only one pair of matrices to plot
        elif n_figures == 1:
            for act in range(num_act):
                max_v = np.max(activations[act].squeeze())
                im = axes[act].imshow(activations[act].squeeze(), vmin=-max_v, vmax=max_v)
                axes[act].set_title(titles[act])
                figure.colorbar(im)

        # Case that there are multiple activations and multiple dimensions
        else:
            for act in range(num_act):
                max_v = np.max(activations[act][fig, :, :].squeeze())
                im = axes[fig, act].imshow(activations[act][fig, :, :].squeeze(), vmin=-max_v, vmax=max_v)
                axes[act].set_title(titles[act])
                figure.colorbar(im)
    plt.tight_layout()
    plt.show()
