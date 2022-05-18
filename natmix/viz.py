
import warnings

import pandas as pd
import xarray as xr
from matplotlib.figure import Figure

from hong2p.olf import format_mix_from_strs, sort_odor_indices
from hong2p import viz
from hong2p.xarray import move_all_coords_to_index


# These both correspond to the typical presentation order in the non-pair recording,
# where things were roughly presented from weakest to strongest (with '~kiwi' being
# presented at 3 dilutions, from lowest to highest, all together).
panel2name_order = {
    # TODO here and elsewhere, probably rename '~kiwi' to 'kiwi mix'
    'kiwi': ('pfo', 'EtOH', 'IAol', 'IAA', 'EA', 'EB', '~kiwi'),
    'control': ('pfo', 'MS', 'VA', 'FUR', '2H', 'OCT', 'control mix'),
}

# TODO maybe pick title automatically based on metadata on corr (+ require that extra
# metadata if we dont have enough of it as-is), to further homogenize plots
# TODO set colormap in here (w/ context manager ideally)
def plot_corr(corr: xr.DataArray, *, panel=None, title='') -> Figure:

    if panel not in panel2name_order.keys():
        raise ValueError('must pass panel keyword argument, from among '
            f'{list(panel2name_order.keys())}'
        )

    # TODO maybe factor into hong2p.viz.callable_ticklabels (or similar wrapper
    # to plotting fns) (also may not always want this done here in plot_corr...)
    corr = move_all_coords_to_index(corr)

    # TODO TODO may want to check we have all names from name_order selected
    # (maybe barring pfo?)
    name_order = panel2name_order[panel]

    # TODO TODO TODO may need to not rely on sort_odor_indices, or special case handling
    # of pair experiment data, to make sure we always order the two pair odors in the
    # same order. since these matrices don't have each exclusively either on the rows or
    # columns, can't use transpose_sort_key i had used earlier for this
    # TODO TODO TODO update sorting to work w/ pair experiment input too, then stop
    # dropping that data here
    # TODO in the meantime, warn if input data has any is_pair[_b] == True
    corr = corr.sel(odor=(corr.is_pair == False), odor_b=(corr.is_pair_b == False)
        ).copy()

    # TODO maybe replace this w/ sort kwarg to matshow / to-be-added-by
    # callable_ticklabels (as normally the latter would make this to_pandas() call, but
    # now we need to sort, and easier to do that starting from a dataframe)
    # TODO TODO or make a hong2p.xarray fn for sorting indices w/ artibrary key (fns)
    # like to copy the pandas behavior i take advantage of
    corr = corr.to_pandas()
    corr = sort_odor_indices(corr, name_order=name_order)

    # TODO TODO might want to select between one of two orders based on whether we only
    # have is_pair==False data or not?

    xticklabels = format_mix_from_strs
    yticklabels = format_mix_from_strs

    with warnings.catch_warnings():
        # For the warning from format_mix_from_strs since we aren't dropping
        # 'repeat' level
        warnings.simplefilter('ignore', UserWarning)

        # TODO shared vmin/vmax? (thread kwargs thru?)
        # TODO colorbar label
        fig, _  = viz.matshow(corr, title=title,
            xticklabels=xticklabels, yticklabels=yticklabels,
            # NOTE: this would currently cause failure on the pair experiment data
            # (because the multiple solvent entries i assume)
            #group_ticklabels=True,
            # TODO TODO fix. i think it's causing failure when i add a limited
            # amount of pair expt data b/c of duplicate ea -4.2 etc
            group_ticklabels=False,
        )

    return fig


def plot_activation_strength(df: pd.DataFrame) -> Figure:
    raise NotImplementedError

