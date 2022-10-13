
from typing import Union
from pprint import pformat

import pandas as pd
import xarray as xr

from hong2p.olf import parse_odor_name, solvent_str


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
    'kiwi': ['pfo', 'EtOH', 'IAol', 'IAA', 'EA', 'EB', '~kiwi'],
    'control': ['pfo', 'MS', 'VA', 'FUR', '2H', 'OCT', 'control mix'],
    # TODO make the swap to these at some point (other changes need to be made too)
    #'kiwi': ['pfo', 'EtOH', 'IAol', 'IAA', 'EA', 'EB', 'kmix'],
    #'control': ['pfo', 'MS', 'VA', 'FUR', '2H', 'OCT', 'cmix'],
}
panel_order = list(panel2name_order.keys())


# TODO test
# TODO move to hong2p.olf (+ add module fns for setting / adding to a module level
# panel -> odor_names state, or at least add kwarg for it)
def get_panel(arr: Union[xr.DataArray, pd.DataFrame]) -> str:
    """Returns str name for panel, given data with 'odor1' column/coordinate.

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
    arr_names = {parse_odor_name(o) for o in arr.odor1.values
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


def drop_mix_dilutions(data):
    """Drops '~kiwi' / 'control mix' at concentrations other than undiluted ('@ 0').
    """
    mix_names = ('~kiwi', 'control mix')

    def is_dilution(arr):
        """Returns boolean vector of length arr True were arr contains mix dilution data
        """
        return arr.str.startswith(name) & ~ arr.str.endswith('@ 0')

    #if isinstance(data, pd.DataFrame):
    dilution_rows = None
    dilution_cols = None
    for name in mix_names:
        curr_dilution_rows = is_dilution(data.odor1)
        if dilution_rows is None:
            dilution_rows = curr_dilution_rows
        else:
            dilution_rows = dilution_rows | curr_dilution_rows

        # Should currently only be True for correlation matrix DataArray input.
        if hasattr(data, 'odor_b'):
            curr_dilution_cols = is_dilution(data.odor1_b)
            if dilution_cols is None:
                dilution_cols = curr_dilution_cols
            else:
                dilution_cols = dilution_cols | curr_dilution_cols

    if isinstance(data, pd.DataFrame):
        assert dilution_cols is None
        # TODO maybe just do data[dilutions_rows].copy(), cause simpler
        return data.drop(index=dilution_rows[dilution_rows].index)
    else:
        # TODO better way? .sel?
        data = data.where(~ dilution_rows, drop=True)
        data = data.where(~ dilution_cols, drop=True)
        return data


# TODO TODO support if arr has both odor (w/ odor[1|2]) and odor_b (w/ odor[1|2]_b)
# TODO TODO after fixing to work w/ corr input (w/ odor_b), use in place of dropping all
# is_pair stuff in plot_corrs (so that if i wanted to show some stuff from pair expt, i
# could)
def drop_nonlone_pair_expt_odors(arr):
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


