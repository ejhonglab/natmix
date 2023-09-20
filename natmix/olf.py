
from pprint import pformat

import pandas as pd
import xarray as xr

from hong2p.olf import parse_odor_name, solvent_str
from hong2p.types import DataFrameOrDataArray


# These both correspond to the typical presentation order in the non-pair recording,
# where things were roughly presented from weakest to strongest (with '~kiwi' being
# presented at 3 dilutions, from lowest to highest, all together).
panel2name_order = {
    # TODO how to expand to support case where we want to have the option of including
    # the pair data?
    # TODO here and elsewhere, probably rename '~kiwi' to 'kiwi mix'
    # (and/or 'control mix' to just 'mix', though that would risk ambiguity if ever done
    # for more than just plotting...)
    # + 'pfo @ 0' -> 'pfo'
    'kiwi': ['pfo', 'EtOH', 'IAol', 'IaA', 'ea', 'eb', '~kiwi'],
    'control': ['pfo', 'ms', 'va', 'fur', '2h', 'oct', 'control mix'],
    # TODO make the swap to these at some point (other changes need to be made too)
    #'kiwi': ['pfo', 'EtOH', 'IAol', 'IaA', 'ea', 'eb', 'kmix'],
    #'control': ['pfo', 'ms', 'va', 'fur', '2h', 'oct', 'cmix'],
}
panel_order = list(panel2name_order.keys())


def _get_odor_var(data: DataFrameOrDataArray) -> str:
    if isinstance(data, pd.DataFrame):
        # TODO also support 'odor' (although may only be DataArray input that currently
        # has 'odor' instead of 'odor1' on input...)
        names = data.index.names
        if 'odor' in names and 'odor1' not in names:
            import ipdb; ipdb.set_trace()
            # TODO want to check answer on columns would be consistent? ever need to
            # support odor on just index and not columns?

        odor_var = 'odor1'

    elif isinstance(data, xr.DataArray):
        # TODO rewrite these conditionals/assertion to be more readable
        if hasattr(data, 'odor') and not hasattr(data, 'odor1'):
            odor_var =  'odor'
        else:
            assert hasattr(data, 'odor1')
            odor_var = 'odor1'
    else:
        raise NotImplementedError

    return odor_var


# TODO test
# TODO move to hong2p.olf (+ add module fns for setting / adding to a module level
# panel -> odor_names state, or at least add kwarg for it)
# TODO allow extra w/ kwarg (as long as we have all the minimum odors?)?
# TODO rename arr->data to be consistent w/ drop_mix_... ?
def get_panel(arr: DataFrameOrDataArray) -> str:
    """Returns str name for panel, given data with 'odor1' [/ 'odor'] column/coordinate.

    Input must have all the odors from a panel and nothing extra, beyond solvent/pfo,
    otherwise a ValueError is raised.

    Notes:
    - no handling of abbreviated/not odors here. names must match what would be
      parsed from names in panel2name_order above (currently abbreviations)

    - ignoring odor concentration, for the moment.
    """

    # actually just ignoring odor2 now, cause some inputs we might want to pass this fn
    # could also have the pair data, and we don't want that to cause failure.
    #if set(arr.odor2.values) != {solvent_str}:

    solvent_names = {'pfo', solvent_str}

    # TODO this actually work w/ (~equiv formatted) DataFrame input?

    odor_var = _get_odor_var(arr)
    # TODO delete if replacement equiv
    #arr_names = {parse_odor_name(o) for o in arr.odor1.values
    assert arr[odor_var].equals(getattr(arr, odor_var))
    #
    arr_names = {parse_odor_name(o) for o in arr[odor_var].values
        if o not in solvent_names
    }

    # In case there are some odor strings like 'pfo @ 0', which I think I have in some
    # places.
    arr_names -= solvent_names

    arr_panel = None
    for panel, panel_names in panel2name_order.items():
        panel_names = set(panel_names) - solvent_names
        missing_panel_odors = panel_names - arr_names
        if len(missing_panel_odors) > 0:
            continue

        extra_odors = arr_names - panel_names
        if len(extra_odors) > 0:
            raise ValueError(f'panel for {pformat(arr_names)} could not be identified.'
                f'\nhad all {panel} odors, but also had these non-solvent odors:\n'
                f'{extra_odors}\narr must have only data from one panel!'
            )

        assert arr_panel is None, 'multiple matching panels'
        arr_panel = panel

    if arr_panel is None:
        raise ValueError(f'panel for {arr_names} could not be identified from:\n' +
            pformat(panel2name_order)
        )

    return arr_panel


# TODO factor into hong2p.xarray (if i support DataFrame too probably + for the 'odor_b'
# thing)?
# TODO TODO maybe also work if there is 'odor' but no 'odor_b'? what does it do now?
def dropna_odors(arr: xr.DataArray, _checks=True) -> xr.DataArray:
    # TODO doc correct?
    # TODO can/should we check that sizes of dims other than 'odor'/'odor_b' don't
    # change?
    """Drops data where all NaN for either a given 'odor' or 'odor_b' index value.
    """
    if _checks:
        notna_before = arr.notnull().sum().item()

    # TODO need to alternate (i.e. does order ever matter? ever not idempotent?)?
    # "dropping along multiple dimensions simultaneously is not yet supported"
    arr = arr.dropna('odor', how='all').dropna('odor_b', how='all')

    if _checks:
        assert arr.notnull().sum().item() == notna_before

    return arr


# TODO unit test. enumerate dilution_[rows|cols] [not-]null combinations
# TODO update type hint to indicate it returns same type as input. possible?
# TODO TODO modify to also work if input just has 'odor'/'odor_b', and not
# 'odor1'/'odor1_b' (other stuff would need to change too, to fix natmix.plot_corr usage
# w/ non-multiindexed odor X odor input (e.g. when called on modelling outputs from
# al_analysis.py)
def drop_mix_dilutions(data: DataFrameOrDataArray) -> DataFrameOrDataArray:
    """Drops '~kiwi' / 'control mix' at concentrations other than undiluted ('@ 0').
    """
    # TODO add note to doc clarifying why no dealing w/ 'odor2[_b]' here
    # (b/c mixes were always alone, and thus 'odor1[_b]' right? but in some pair stuff
    # lone odors could be 'odor2[_b]'... so what ensured mix stuff couldn't?)

    mix_names = ('~kiwi', 'control mix')

    odor_var = _get_odor_var(data)

    # Only used for DataFrame input
    old_index_levels = None
    if isinstance(data, pd.DataFrame):
        # TODO maybe flag to disable this, or only if needed levels are not already
        # columns? currently assuming DataFrame input will always have them in index
        # levels
        old_index_levels = data.index.names
        assert odor_var in old_index_levels
        data = data.reset_index()

    def is_dilution(name, arr):
        """Returns boolean array, True were arr contains dilutions of odor with `name`
        """
        # TODO also support just missing '@' delimiter (if '@ 0' were implied in this
        # case)?
        return arr.str.startswith(name) & ~ arr.str.endswith('@ 0')

    dilution_rows = None
    dilution_cols = None
    for name in mix_names:
        curr_dilution_rows = is_dilution(name, data[odor_var])
        if dilution_rows is None:
            dilution_rows = curr_dilution_rows
        else:
            dilution_rows = dilution_rows | curr_dilution_rows

        # Should currently only be True for correlation matrix DataArray input.
        if hasattr(data, f'{odor_var}_b'):
            # TODO delete if below works
            #curr_dilution_cols = is_dilution(name, data.odor1_b)
            curr_dilution_cols = is_dilution(name, data[f'{odor_var}_b'])

            if dilution_cols is None:
                dilution_cols = curr_dilution_cols
            else:
                dilution_cols = dilution_cols | curr_dilution_cols

    # TODO TODO but does earlier code actually work w/ dataframe input? test!
    if isinstance(data, pd.DataFrame):
        assert dilution_cols is None

        # TODO maybe just do data[dilutions_rows].copy(), cause simpler
        data = data.drop(index=dilution_rows[dilution_rows].index)

        assert old_index_levels is not None
        return data.set_index(old_index_levels)
    else:
        # might trigger... would need to revert so something like commented code if so
        # (currently triggering on new diag+megamat data where there are no natural mix
        # dilutions)
        #assert dilution_rows is not None
        #assert dilution_cols is not None
        # TODO TODO why did i decide to comment this again?
        '''
        if dilution_rows.any().item():
            # TODO better way? .sel?
            data = data.where(~ dilution_rows, drop=True)

        if dilution_cols:
            data = data.where(~ dilution_cols, drop=True)
        '''
        # TODO test versions gated behind None tests still works as expected with old
        # input that actually had natural mix diluitions (added tests here to work on
        # new data lacking any of these odors)
        if dilution_rows is not None:
            data = data.where(~ dilution_rows, drop=True)

        if dilution_cols is not None:
            data = data.where(~ dilution_cols, drop=True)

        return data


# TODO TODO support if arr has both odor (w/ odor[1|2]) and odor_b (w/ odor[1|2]_b)
# TODO TODO after fixing to work w/ corr input (w/ odor_b), use in place of dropping all
# is_pair stuff in plot_corrs (so that if i wanted to show some stuff from pair expt, i
# could)
def drop_nonlone_pair_expt_odors(arr: xr.DataArray) -> xr.DataArray:
    """
    Drops (along 'odor' dim) presentations that are any of:
    - solvent-only
    - >1 odors presented simultaneously (mixed in air) w/ non-zero concentration

    This is to make some pair experiment data comparable alongside the same odors
    presented from the non-pair experiment.
    """
    odor1 = arr.odor1
    odor2 = arr.odor2
    is_pair = arr.is_pair

    # Once we drop these odors, the only data we should be left with (for
    # the pair experiments) are odors presented by themselves.
    pair_expt_odors_to_drop = (
        ((odor1 == 'solvent') & (odor2 == 'solvent')) |
        ((odor1 != 'solvent') & (odor2 != 'solvent'))
    )
    # TODO adapt this to work w/ either is_pair or is_pair_b
    assert pair_expt_odors_to_drop[pair_expt_odors_to_drop].is_pair.all().item()

    # TODO better name
    mask = (is_pair == False) | ~pair_expt_odors_to_drop

    # TODO copy?
    return arr[mask]

