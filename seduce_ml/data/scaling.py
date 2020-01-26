import numpy as np


def flatten(ll):
    return [lj for li in ll for lj in li]


def unscale_input(x, scaler, variables):
    """Unscales an input 2D array"""

    if len(x.shape) != 2:
        raise Exception("Expecting an 2D array")

    row_count, columns_count = x.shape
    additional_columns_count = len(variables) - columns_count

    augmented_array = np.insert(x, [columns_count] * additional_columns_count, -10, axis=1)

    unscaled_augmented_array = scaler.inverse_transform(augmented_array)

    return unscaled_augmented_array[:, :columns_count]


def rescale_input(x, scaler, variables):
    """Rescales an input 2D array"""

    if len(x.shape) != 2:
        raise Exception("Expecting an 2D array")

    row_count, columns_count = x.shape
    additional_columns_count = len(variables) - columns_count

    augmented_array = np.insert(x, [columns_count] * additional_columns_count, -10, axis=1)

    unscaled_augmented_array = scaler.transform(augmented_array)

    return unscaled_augmented_array[:, :columns_count]


def unscale_output(y, scaler, variables):
    """Unscales an output 2D array"""

    if len(y.shape) != 2:
        raise Exception("Expecting an 2D array")

    row_count, columns_count = y.shape
    additional_columns_count = len(variables) - columns_count

    augmented_array = np.insert(y, [0] * additional_columns_count, -10, axis=1)

    unscaled_augmented_array = scaler.inverse_transform(augmented_array)

    return unscaled_augmented_array[:, -columns_count:]


def rescale_output(y, scaler, variables):
    """Rescales an output 2D array"""

    if len(y.shape) != 2:
        raise Exception("Expecting an 2D array")

    row_count, columns_count = y.shape
    additional_columns_count = len(variables) - columns_count

    augmented_array = np.insert(y, [0] * additional_columns_count, -10, axis=1)

    unscaled_augmented_array = scaler.transform(augmented_array)

    return unscaled_augmented_array[:, -columns_count:]