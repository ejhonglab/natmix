
import warnings

import pandas as pd
import xarray as xr
from matplotlib.figure import Figure
import seaborn as sns
import numpy as np

from hong2p.olf import format_mix_from_strs, sort_odors, panel_odor_orders
from hong2p import viz
from hong2p.viz import with_panel_orders
from hong2p.xarray import move_all_coords_to_index
from hong2p.util import add_group_id


# These both correspond to the typical presentation order in the non-pair recording,
# where things were roughly presented from weakest to strongest (with '~kiwi' being
# presented at 3 dilutions, from lowest to highest, all together).
panel2name_order = {
    # TODO TODO how to expand to support case where we want to have the option of
    # including thet pair data?
    # TODO here and elsewhere, probably rename '~kiwi' to 'kiwi mix'
    'kiwi': ['pfo', 'EtOH', 'IAol', 'IAA', 'EA', 'EB', '~kiwi'],
    'control': ['pfo', 'MS', 'VA', 'FUR', '2H', 'OCT', 'control mix'],
}
panel_order = list(panel2name_order.keys())

# TODO maybe pick title automatically based on metadata on corr (+ require that extra
# metadata if we dont have enough of it as-is), to further homogenize plots
# TODO set colormap in here (w/ context manager ideally)
def plot_corr(corr: xr.DataArray, *, panel=None, title='') -> Figure:
    """Shows correlations between representations of panel odors.
    """

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

    # TODO TODO TODO may need to not rely on sort_odors, or special case handling
    # of pair experiment data, to make sure we always order the two pair odors in the
    # same order. since these matrices don't have each exclusively either on the rows or
    # columns, can't use transpose_sort_key i had used earlier for this
    # TODO TODO TODO update sorting to work w/ pair experiment input too, then stop
    # dropping that data here
    # TODO in the meantime, warn if input data has any is_pair[_b] == True
    corr = corr.sel(odor=(corr.is_pair == False), odor_b=(corr.is_pair_b == False)
        ).copy()

    if len(corr) == 0:
        raise ValueError('corr did not contain any non-pair experiment data! '
            'currently pair experiment data is not analyzed in plot_corr.'
        )

    # TODO maybe replace this w/ sort kwarg to matshow / to-be-added-by
    # callable_ticklabels (as normally the latter would make this to_pandas() call, but
    # now we need to sort, and easier to do that starting from a dataframe)
    # TODO TODO or make a hong2p.xarray fn for sorting indices w/ artibrary key (fns)
    # like to copy the pandas behavior i take advantage of
    corr = corr.to_pandas()
    corr = sort_odors(corr, name_order=name_order)

    # TODO TODO might want to select between one of two orders based on whether we only
    # have is_pair==False data or not?

    xticklabels = format_mix_from_strs
    yticklabels = format_mix_from_strs

    with warnings.catch_warnings():
        # For the warning from format_mix_from_strs since we aren't dropping
        # 'repeat' level
        warnings.simplefilter('ignore', UserWarning)

        # TODO colorbar label (thread thru kwarg?)
        fig, _  = viz.matshow(corr, title=title,
            xticklabels=xticklabels, yticklabels=yticklabels,
            # NOTE: this would currently cause failure on the pair experiment data
            # (because the multiple solvent entries i assume)
            # TODO TODO fix. i think it's causing failure when i add a limited
            # amount of pair expt data b/c of duplicate ea -4.2 etc
            group_ticklabels=True,
            vmin=-0.2,
            vmax=1.0,
        )

    return fig


# TODO maybe give more generic name? (potentially factoring out core and calling that w/
# fn still of this name?)
# TODO remove _plot_fn kwarg after settling on one for color_flies=False
def plot_activation_strength(df: pd.DataFrame, activation_col='mean_dff',
    color_flies=False, _checks=False, _plot_fn=None) -> sns.FacetGrid:
    """Shows activation strength of each odor in each panel.

    Args:
        df: must have at least the following columns:
            - 'panel': only 'kiwi'/'control' rows used
            - 'is_pair': True/False
            - 'date'
            - 'fly_num'
            - 'odor1'
            - 'odor2'
            - Column specified by `activation_col`

        activation_col: the Y-axis variable

        color_flies: if True, will color points to indicate fly identity (shared across
            facets), as well as connecting points from the same fly together

    Returns a seaborn FacetGrid with one facet per panel.

    Currently only plotting data where `is_pair` is False.
    """

    # Dropping 'glomeruli_diagnostics' panel, if present
    df = df[df.panel.isin(panel2name_order)].copy()

    df = add_group_id(df[~df.is_pair], ['date', 'fly_num'], name='fly_id')

    nonpair_df = df[~df.is_pair].copy()
    nonpair_df.rename(columns={'odor1': 'odor'}, inplace=True)
    assert set(nonpair_df.odor2.unique()) == {'solvent'}

    df = nonpair_df

    panel2order = panel_odor_orders(df, panel2name_order)

    plot_fn_kws = dict(
        x='odor', y=activation_col #, order=order
    )

    # Just the ones shared between FacetGrid constructor and catplot kwargs.
    shared_facet_kws = dict(
        data=df, col='panel', col_order=panel_order, sharex=False,
    )

    if color_flies:
        shared_facet_kws['hue'] = 'fly_id'

        # TODO check if equiv to just using str 'hls'
        # TODO do w/o numpy call if easy way (-> remove np import)
        n_flies = df.fly_id.nunique()
        fly_colors = sns.color_palette('hls', n_flies)
        fly_id_palette = dict(zip(np.unique(df.fly_id), fly_colors))
        shared_facet_kws['palette'] = fly_id_palette
        #shared_facet_kws['palette'] = 'hls'

        # TODO TODO TODO test both w/ and w/o hue/palette kwargs above
        unwrapped_plot_fn = sns.pointplot
    else:
        # TODO delete
        if _plot_fn is not None:
            unwrapped_plot_fn = _plot_fn
        #
        else:
            unwrapped_plot_fn = sns.barplot

        # TODO TODO TODO probably use something other than pointplot when
        # color_flies=False (don't care about connecting lines then, and might still
        # want a point for each fly, but more focus on CI probably)

    plot_fn = with_panel_orders(unwrapped_plot_fn, panel2order)

    # This still doesn't drop stuff thats in the order but has no data for the panel.
    g = sns.FacetGrid(**shared_facet_kws, dropna=True)

    #g.map(plot_fn, 'odor', activation_col)
    g.map_dataframe(plot_fn, **plot_fn_kws)

    # TODO make ylabel nice
    # TODO de-emph (/hide?) lines connecting points?

    # TODO return to using this if i can resolve the order issue (seems like it would
    # take changes in my seaborn fork). in the meantime, may still want to compare
    # current plot to this to help ensure correctness of current plot.
    # TODO TODO also use fly_id_palette for testing against this plot
    #g = sns.catplot(**plot_fn_kws, **shared_facet_kws, kind='point', legend=False)

    g.set_titles('{col_name}')

    # TODO get that latex str + maybe just call that helper fn i had to only show xlabel
    # once, when the label is shared
    #g.set_axis_labels('Mean dF/F')

    # TODO try 45?
    g.set_xticklabels(rotation=90)

    # With a lot of flies, I don't think it's worth it.
    #g.add_legend(title='Fly')

    g.tight_layout()


    # TODO delete / somehow turn into test, after verifying it matches up w/ facetgrid
    # stuff using with_panel_orders
    if _checks:
        # TODO could try using matplotlib.testing.decorators.check_figures_equal.
        # i couldn't find a comparable testing utility function for Axes objects
        # (if i make some effort to get these in one figure like plot created via
        # seasborn + wrapper)
        # TODO otherwise, homebrew some equality check comparing lines/points/colors,
        # maybe? (maybe using ax.get_children() or ax.get_lines()?)
        import matplotlib.pyplot as plt
        for panel, order in panel2order.items():
            fig, ax = plt.subplots()
            unwrapped_plot_fn(**plot_fn_kws, data=df, order=order,
                # TODO TODO TODO might need to make a palette in advance, and share
                # between two plotting methods, to make comparable...
                hue='fly_id' if color_flies else None,
                palette=fly_id_palette if color_flies else None,
                #palette='hls',
            )
            plt.xticks(rotation=90)
            ax.set_title(panel)

        g.add_legend(title='Fly')
        warnings.warn('manually verify the plots show match, then re-run without '
            '_checks=True'
        )
        plt.show()
        import ipdb; ipdb.set_trace()
    #

    return g

