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

    figure, axes = plt.subplots(num_act, n_figures, figsize=(num_act * 10, n_figures * 10))

    for fig in range(n_figures):
        # Case that there is only one set of activations to plot
        if num_act == 1:
            axes[fig].imshow(activations[0][fig, :, :].squeeze())
            axes[fig].set_title(titles[fig])

        # Case that there is only one pair of matrices to plot
        elif n_figures == 1:
            for act in range(num_act):
                axes[act].imshow(activations[act].squeeze())
                axes[act].set_title(titles[act])

        # Case that there are multiple activations and multiple dimensions
        else:
            for act in range(num_act):
                axes[fig, act].imshow(activations[act][fig, :, :].squeeze())
                axes[act].set_title(titles[act])
    plt.tight_layout()
    plt.show()
