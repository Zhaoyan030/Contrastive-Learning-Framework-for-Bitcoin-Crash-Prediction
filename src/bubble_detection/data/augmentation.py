import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d


def data_augmentation(windows, aug1_method, aug2_method, aug1_method_param, aug2_method_param, comb=False):
    """
    Do augmentation for each window.

    Args:
        windows: a 2-D array of windows in training set. the output of function get_train_test().
        aug1_method: a string to indicate the first desired augmentation method.
        aug2_method: a string to indicate the second desired augmentation method.
        aug1_method_param: a dict containing arguments needed in the first augmentation method.
        aug2_method_param: a dict containing arguments needed in the second augmentation method.
        comb: whether to use the combination of two augmentations.
            If True, then one head uses identity, the other uses aug2_method(aug1_method);
            if False, then one head uses aug1_method, the other uses aug2_method

    Returns: A 2D array of augmented windows. #row windows_aug = #row windows, #col windows_aug = 2 * #col windows

    """

    aug1 = map_aug(aug1_method)
    aug2 = map_aug(aug2_method)

    if comb:
        windows_aug = np.concatenate(
            (
                identity(windows),
                aug2(aug1(windows, **aug1_method_param), **aug2_method_param)
            ),
            axis=1
        )
    else:
        windows_aug = np.concatenate(
            (
                aug1(windows, **aug1_method_param),
                aug2(windows, **aug2_method_param)
            ),
            axis=1
        )
    return windows_aug


def map_aug(name):
    if name == 'identity':
        return identity
    elif name == 'jitter':
        return jitter
    elif name == 'scaling':
        return scaling
    elif name == 'permutation':
        return permutation
    elif name == 'mag_warp':
        return mag_warp
    elif name == 'time_warp':
        return time_warp
    elif name == 'lr_flip':
        return lr_flip
    elif name == 'crop_resize':
        return crop_resize
    elif name == 'smooth':
        return gaussian_smooth
    else:
        raise ValueError(f"Invalid augmentation method {name}")


def identity(X):
    return X


def jitter(x, sigma=1.0):
    # add zero-mean random number to every data point
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=0.6):
    # multiple random normal-distributed factor to every data point
    factor = abs(np.random.normal(loc=1., scale=sigma, size=(x.shape[0], 1)))
    return factor * x


def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[0, warp]
        else:
            ret[i] = pat
    return ret


def generate_random_curves(X, sigma=0.2, knot=10):
    xx = np.ones((X.shape[0], 1)) * (np.linspace(0, X.shape[1] - 1, num=knot+2, endpoint=True))  # n sample * (k+2) points
    yy = np.random.normal(loc=1.0, scale=sigma, size=(X.shape[0], knot + 2))  # n sample * (k+2) points
    x_range = np.arange(X.shape[1])
    res = []
    for i in np.arange(X.shape[0]):
        cs = CubicSpline(xx[i, :], yy[i, :])
        res.append(abs(cs(x_range)))    # if not adding abs(), then augmented values could be negative
    return np.array(res)


def mag_warp(X, sigma=0.6, knot=10):
    return X * generate_random_curves(X, sigma, knot)


def distort_timesteps(X, sigma=0.2, knot=10):
    tt = generate_random_curves(X, sigma, knot)
    tt_cum = np.cumsum(tt, axis=1)
    # Make the last value have X.shape[0]
    t_scale = (X.shape[1] - 1) / tt_cum[:, -1]
    t_scale = t_scale.reshape(-1, 1)
    res = tt_cum * t_scale
    return res


def time_warp(X, sigma=1.0, knot=8):
    tt_new = distort_timesteps(X, sigma, knot)
    x_range = np.arange(X.shape[1])
    res = []
    for i in np.arange(X.shape[0]):
        X_new = np.interp(x_range, tt_new[i, :], X[i, :])
        res.append(X_new)
    return np.array(res)


def lr_flip(X):
    return np.fliplr(X)


def crop_resize(X):
    X_inter = np.apply_along_axis(lambda x: [(a + b) / 2 for a, b in zip(x[:-1], x[1:])], 1, X)
    X_new = np.insert(X, np.arange(1, X.shape[1]), X_inter, axis=1)
    res = []
    for i in np.arange(X.shape[0]):
        k = np.random.randint(0, X.shape[1])
        res.append(X_new[i][k:(k + X.shape[1])])
    return np.array(res)


def gaussian_smooth(X):
    sigma = np.random.uniform(1, 2)
    X_new = np.apply_along_axis(lambda x: gaussian_filter1d(x, sigma, mode='nearest'), 1, X)
    return X_new