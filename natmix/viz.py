
import warnings
from typing import Optional
# TODO delete
import traceback
#

import pandas as pd
import xarray as xr
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from hong2p.olf import (format_mix_from_strs, sort_odors, panel_odor_orders,
    solvent_str
)
from hong2p import viz
from hong2p.viz import with_panel_orders, no_constrained_layout
from hong2p.xarray import move_all_coords_to_index
from hong2p.util import add_group_id, dff_latex

from natmix.olf import panel2name_order, panel_order, get_panel, drop_mix_dilutions


# TODO maybe warn if called w/ different input (hash sorted (date, fly_num) sequence for
# lookup?)?
# TODO parameterize dict return type hint
def get_fly_id_palette(df: pd.DataFrame) -> dict:

    if 'fly_id' not in df.columns:
        # This will sort on the ('date', 'fly_num') combinations, by default.
        # TODO replace w/ add_fly_id? don't i have a fn for that?
        add_group_id(df, ['date', 'fly_num'], name='fly_id', inplace=True)

    assert not df.fly_id.isna().any(), 'nunique does not count NaN, but unique does'
    n_flies = df.fly_id.nunique()
    # TODO default to cc.glasbey (but take kwarg to replace 'hls'?)?
    fly_colors = sns.color_palette('hls', n_flies)
    # TODO do w/o numpy call if easy way (-> remove np import). set()?
    return dict(zip(sorted(np.unique(df.fly_id)), fly_colors))


# TODO TODO also work on DataFrame input?
# TODO maybe pick title automatically based on metadata on corr (+ require that extra
# metadata if we dont have enough of it as-is), to further homogenize plots
# TODO set colormap in here (w/ context manager ideally)
# TODO if good values for figsize/cbar_shrink seem to depend on data (i.e. odor name
# lengths), maybe just revert to tight layout (give viz.matshow kwarg for this?)
# TODO probably want to just force cbar to be same height as matshow Axes anyway?
# use that + constrained layout?
# TODO TODO increase figsize / dpi so that corrs don't look quite as blurry as they do
# now that i added figsize (which is slightly smaller than default)
# TODO TODO move core of this to hong2p.viz, and maybe wrap here if needed?
# TODO type hint name_order
def plot_corr(corr: xr.DataArray, panel: Optional[str] = None, *, title='',
    mix_dilutions=False, vmin=-0.2, vmax=1.0, warn=False, figsize=(5, 4.8),
    cbar_shrink=0.736, name_order=None, sort: bool = True, **kwargs) -> Figure:
    """Shows correlations between representations of panel odors.

    Args:
        corr: pairwise correlations betweeen all odors in panel, of shape
            (# odors, # odors)

        panel: 'kiwi'/'control'

        kwargs: passed thru to `hong2p.viz.matshow`
    """
    if not mix_dilutions:
        # TODO add test that i haven't broken this (the only call currently using
        # DataArray input) by changing fn natmix.olf.drop_mix_dilutions.is_dilution
        corr = drop_mix_dilutions(corr)

    if name_order is None:
        # TODO deprecate panel argument once get_panel is working
        if panel is None:
            try:
                panel = get_panel(corr)
            except ValueError as err:
                warn_msg = f'{err}\nsorting correlation matrix alphabetically!'
                if warn:
                    warnings.warn(warn_msg)
        else:
            if panel not in panel2name_order.keys():
                raise ValueError('must pass panel keyword argument, from among '
                    f'{list(panel2name_order.keys())}'
                )

        if panel is not None:
            # TODO may want to check we have all names from name_order selected
            # (maybe barring pfo?)
            name_order = panel2name_order[panel]

    # TODO maybe factor into hong2p.viz.callable_ticklabels (or similar wrapper
    # to plotting fns) (also may not always want this done here in plot_corr...)
    corr = move_all_coords_to_index(corr)

    # Assuming input does not contain pair data if this variable not present.
    has_is_pair = 'is_pair' in corr.get_index('odor').names

    # TODO update sorting to work w/ pair experiment input too, then stop dropping that
    # data here
    # TODO in the meantime, warn if input data has any is_pair[_b] == True
    if has_is_pair:
        #'''
        try:
            # TODO delete if assertion below isn't failing
            old = corr.sel(
                odor=(corr.is_pair == False), odor_b=(corr.is_pair_b == False)
            ).copy()
            #

            # TODO factor out (+ use in al_analysis where i do this)
            corr = corr.sel(is_pair=False, is_pair_b=False).copy()

            # TODO delete if not failing
            assert corr.equals(old)

        # TODO was an AssertionError triggering this or something else?
        # presumably something else if i felt the need to print traceback?
        # test on old data that was triggering this!
        except:
            print('ERROR IN PLOT_CORR:')
            print(traceback.format_exc())
            print()
            print('END ERROR IN PLOT_CORR')
            import ipdb; ipdb.set_trace()
        #'''

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

    if sort:
        corr = sort_odors(corr, name_order=name_order)

    # TODO might want to select between one of two orders based on whether we only have
    # is_pair==False data or not?

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
            # (still true?)
            group_ticklabels=True, vmin=vmin, vmax=vmax, figsize=figsize,
            cbar_shrink=cbar_shrink, **kwargs
        )

    return fig


activation_col2label = {
    'mean_dff': f'mean {dff_latex}',
}

@no_constrained_layout
# TODO maybe give more generic name? (potentially factoring out core and calling that w/
# fn still of this name?) (rename activation_col to just 'y')
# TODO some version of this fn (or a more general fn this one calls) that includes a
# kwarg for putting vertical (dashed?) lines between level changes after application of
# some fn / at specific points (e.g. for grouping activation strengths of components vs
# mix at dilutions, or for grouping certain odor/odor correlations)
def plot_activation_strength(df: pd.DataFrame, *, activation_col: str ='mean_dff',
    color_flies: bool = False, mix_dilutions: bool = False, plot_mean_ci: bool = True,
    ylabel: Optional[str] = None, title: Optional[str] = None, seed=0,
    swarmplot_kws: Optional[dict] = None, _checks=False, _debug=False) -> sns.FacetGrid:
    # TODO how important are all these columns actually? want to relax so i can use
    # for model KC data...
    """Shows activation strength of each odor in each panel.

    Args:
        df: must have at least the following columns:
            - 'panel': only 'kiwi'/'control' rows used
            - 'date'
            - 'fly_num'
            - 'odor1'
            - column specified by `activation_col`

            may also have:
            - 'is_pair': True/False (indicating recording was a binary mixture ramp
                  experiment)

            - 'odor2' (for 2nd component of in-air binary mixtures)

        activation_col: the Y-axis variable

        ylabel: label for Y-axis. If not passed, will check whether
            natmix.viz.activation_col2label has a label for the current activation_col.
            Otherwise, will just use the column name.

        color_flies: if True, will color points to indicate fly identity (shared across
            facets), as well as connecting points from the same fly together

        plot_mean_ci: whether to use `sns.barplot` to show 95% CI on mean. ignored if
            `color_flies=True`

        seed: passed to `sns.barplot` (only relevant if `color_flies=False` and
            `plot_mean_ci=True`). seeded (=0) by default.

    Returns a seaborn FacetGrid with one facet per panel.

    Currently only plotting data where `is_pair` is False (set True for stuff with odor2
    defined alongside odor1, or False everywhere if input does not have 'odor2' column).
    """
    if not mix_dilutions:
        df = drop_mix_dilutions(df)

    # TODO err if we don't have both 'kiwi' / 'control' left?
    # or maybe warn if we only have one?
    #
    # dropping 'glomeruli_diagnostics' panel, if present
    df = df[df.panel.isin(panel2name_order)].copy()

    assert len(df) > 0, 'dropped all panels'

    if 'odor2' in df.columns:
        # TODO don't drop these? plot alongside the in-vial mixtures (or regardless, for
        # experiments that only have these)?
        df.loc[
            (df['odor1'] != solvent_str) & (df['odor2'] != solvent_str), 'is_pair'
        ] = True
    else:
        df['is_pair'] = False

    nonpair_df = df[~df.is_pair].copy()
    nonpair_df.rename(columns={'odor1': 'odor'}, inplace=True)
    df = nonpair_df

    panel2order = panel_odor_orders(df, panel2name_order)

    plot_fn_kws = dict(x='odor', y=activation_col)

    # Just the ones shared between FacetGrid constructor and catplot kwargs.
    shared_facet_kws = dict(
        data=df, col='panel', col_order=panel_order, sharex=False,
        # "Height (in inches) of each facet"
        height=5,
        # "Aspect ratio of each facet, so that aspect * height gives the width"
        aspect=1,
    )

    if ylabel is None:
        if activation_col in activation_col2label:
            ylabel = activation_col2label[activation_col]
        else:
            ylabel = activation_col

    if color_flies:
        fly_id_palette = get_fly_id_palette(df)

        #shared_facet_kws['hue'] = 'fly_id'
        # TODO check if equiv to just using str 'hls'
        #shared_facet_kws['palette'] = fly_id_palette

        # TODO check that (w/o dodge=True) plots are same as if we let FacetGrid handle
        # these kwargs
        plot_fn_kws['hue'] = 'fly_id'
        plot_fn_kws['palette'] = fly_id_palette
        plot_fn_kws['dodge'] = True

        def pointplot(*args, **kwargs):
            # NOTE: not possible to change alpha via palette passed in, at least not
            # with this pointplot function and seaborn 0.11.2 (maybe in current seaborn
            # it is?).
            #
            # there is a `seed=` kwarg available to this fn, but don't think i need to
            # pass to it, as pretty sure never using this path to show error bars
            return sns.pointplot(*args,
                #linestyles='dotted',
                # TODO fix deprecation (/delete):
                # The `scale` parameter is deprecated and will be removed in v0.15.0.
                # You can now control the size of each plot element using matplotlib
                # `Line2D` parameters (e.g., `linewidth`, `markersize`, etc.).
                #scale=0.5,
                **kwargs
            )

        unwrapped_plot_fns = [pointplot]
    else:
        # TODO expose this as kwarg?
        ci = 95
        # TODO option to put this on a second line? ylabel_ci_sep?
        ylabel += f' (with {ci:.0f}% CI)'

        def just_err_barplot(*args, **kwargs):
            return sns.barplot(*args,
                # TODO add mechanism to override these too? (err_kws?)
                #
                # this is an error bar on the mean (b/c default estimator='mean')
                errorbar=('ci', ci),
                capsize=0.2,
                facecolor=(1, 1, 1, 0),
                err_kws=dict(color=(0, 0, 0, 1.0), linewidth=1.5),
                seed=seed,
                **kwargs
            )

        if swarmplot_kws is None:
            swarmplot_kws = dict()

        _ax_id2color = dict()
        # no `seed=` kwarg to this fn (nor should i need one, as error bars are drawn
        # via `barplot` call above)
        def swarmplot(*args, **kwargs):

            ax_id = id(plt.gca())
            curr_color = kwargs['color']
            if ax_id in _ax_id2color:
                assert curr_color == _ax_id2color[ax_id]
            else:
                _ax_id2color[ax_id] = curr_color

            kwargs['color'] = (0, 0, 0)

            with warnings.catch_warnings():
                # TODO specific filter on message, to be more clear
                warnings.filterwarnings('error')

                try:
                    # default marker size=5 (points)
                    return sns.swarmplot(*args, **{
                        # default warn_thresh=0.05 (percentage of points not able to be
                        # placed, triggering warning we are filtering for here)
                        **dict(alpha=0.4, size=5, warn_thresh=0.001),
                        **swarmplot_kws, **kwargs
                    })

                except UserWarning as err:
                    # (this is in the message of the warning we are trying to catch)
                    assert 'points cannot be placed' in str(err)
                    raise err

        if plot_mean_ci:
            unwrapped_plot_fns = [just_err_barplot, swarmplot]
        else:
            unwrapped_plot_fns = [swarmplot]


    plot_fns = [with_panel_orders(fn, panel2order) for fn in unwrapped_plot_fns]

    # This still doesn't drop stuff thats in the order but has no data for the panel.
    g = sns.FacetGrid(**shared_facet_kws, dropna=True)

    for plot_fn in plot_fns:
        g.map_dataframe(plot_fn, **plot_fn_kws)

    # TODO TODO also use fly_id_palette for testing against this plot
    #g = sns.catplot(**plot_fn_kws, **shared_facet_kws, kind='point', legend=False)

    g.set_titles('{col_name}')

    # This looks kinda nice, but for arbtrary length text, the labelpad value can't
    # really just be hardcoded to one thing, and I'm not sure how to figure out what it
    # should be for a given label contents.
    # TODO could using constrained layout work if we don't do g.tight_layout() below?
    # default labelpad=4
    #g.set_ylabels(ylabel, rotation=0, labelpad=30)

    g.set_ylabels(ylabel)

    # 45 made it look like there was an offset, as if the labels were supposed to label
    # something further to the right than intended.
    g.set_xticklabels(rotation=90)

    if _debug:
        import matplotlib as mpl

        print(f'{color_flies=}')

        prefix = 'figure.subplot.'
        sp_vars = ['left', 'right', 'bottom', 'top', 'wspace', 'hspace']

        print('rcParam defaults:')
        for x in sp_vars:
            key = f'{prefix}{x}'
            print(f'{key}:', mpl.rcParams[key])
        print()

        print('before tight_layout:')
        for x in sp_vars:
            print(f'{x}:', getattr(g.fig.subplotpars, x))
        print()

    g.tight_layout()

    if _debug:
        print('after tight_layout:')
        for x in sp_vars:
            print(f'{x}:', getattr(g.fig.subplotpars, x))
        print()

    # All values in here taken from getting these attributes from g.fig.subplotpars,
    # running on my computer right after the g.tight_layout() call above. At least when
    # paired with the bbox_inches=None argument to savefig call, this seems to produce
    # the expected result on my computer. Yet to be seen whether it also fixes the issue
    # on Remy's end (where x/ylabel, potentially among other things, were not displayed,
    # perhapse because they were cut off).
    g.fig.subplots_adjust(left=0.0854, right=0.985, bottom=0.3178, top=0.9313,
        wspace=0.04515, hspace=0.2
    )

    if _debug:
        print('after manual g.fig.subplots_adjust:')
        for x in sp_vars:
            print(f'{x}:', getattr(g.fig.subplotpars, x))
        print()

    # NOTE: seems to be broken now anyway. line 299 in swarmplot has KeyError trying to
    # get 'color' from kwargs
    #
    # TODO delete / somehow turn into test, after verifying it matches up w/ facetgrid
    # stuff using with_panel_orders
    if _checks:
        # TODO could try using matplotlib.testing.decorators.check_figures_equal.
        # i couldn't find a comparable testing utility function for Axes objects
        # (if i make some effort to get these in one figure like plot created via
        # seasborn + wrapper)
        # TODO otherwise, homebrew some equality check comparing lines/points/colors,
        # maybe? (maybe using ax.get_children() or ax.get_lines()?)
        for panel, order in panel2order.items():
            fig, ax = plt.subplots()

            for unwrapped_plot_fn in unwrapped_plot_fns:
                unwrapped_plot_fn(ax=ax, **plot_fn_kws, data=df, order=order,
                    hue='fly_id' if color_flies else None,
                    palette=fly_id_palette if color_flies else None,
                )

            plt.xticks(rotation=90)
            ax.set_title(panel)

        g.add_legend(title='fly')
        warnings.warn('manually verify the plots show match, then re-run without '
            '_checks=True'
        )
        plt.show()
        import ipdb; ipdb.set_trace()
    #

    # TODO this work? (yes, but need to update how plotting is done so it shows
    # date/fly_num or fly_id for each) (assuming they are sorted tho...)
    if color_flies:
        g.add_legend(title='fly')

    if title is not None:
        # TODO 1.05 enough?
        g.fig.suptitle(title, y=1.05)

    return g

