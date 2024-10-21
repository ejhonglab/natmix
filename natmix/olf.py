
from pprint import pformat

import numpy as np
import pandas as pd
import xarray as xr

from hong2p.olf import parse_odor_name, parse_log10_conc, solvent_str
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
    # NOTE: 'ea+eb' is the in-vial mixture. in air mixture should currently have spaces
    # around the '+' delimiter (would have been true, if I didn't replace the mixture
    # str for the air-mix in some code, with values like 'eb+ea (air mix)')
    # TODO or do i want 2-component mixtures after 5-component mixes?
    # '~kiwi' and 'kmix' the same.
    'kiwi': ['pfo', 'EtOH', 'IAol', 'IaA', 'ea', 'eb', 'ea+eb', 'ea+eb (air mix)',
        'eb+ea (air mix)', '~kiwi', 'kmix'
    ],
    # 'oct' and '1o3ol' are the same thing. 'oct' was my old abbreviation for it.
    # 'control mix' and 'cmix' the same.
    # TODO try to have mixes in consistent order (prob 2h first in both cases)
    # TODO fix? seems that current code generating '2h+1o3ol (air mix)' (and similar)
    # does not have a deterministic component order...
    'control': ['pfo', 'ms', 'va', 'fur', '2h', 'oct', '1o3ol', '1o3ol+2h',
        '2h+oct (air mix)', '2h+1o3ol (air mix)',  '1o3ol+2h (air mix)', 'control mix',
        'cmix'
    ],
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

    mix_names = ('~kiwi', 'kmix', 'control mix', 'cmix')

    odor_var = _get_odor_var(data)

    # only used for DataFrame input
    old_index_levels = None
    if isinstance(data, pd.DataFrame):
        # if data already had reset index, .names would be == [None]
        if data.index.names != [None]:
            old_index_levels = data.index.names

            assert odor_var in old_index_levels, \
                f'{odor_var=} not in {old_index_levels=}'

            old_columns = data.columns.copy()

            # NOTE: this changes column dypes since it inserts previous index levels as
            # columns (w/ str names equal to index level names) (and column levels other
            # than those used to insert str index level names probably NaN/similar, or
            # maybe empty str)
            data = data.reset_index()
        else:
            assert odor_var in data.columns

    def is_dilution(name, arr):
        """Returns boolean array, True were arr contains dilutions of odor with `name`
        """
        is_dilution_mask = np.array([parse_log10_conc(x) for x in arr.values]) != 0

        # TODO delete. all masks in here seemed equiv to that from list comp above
        # (for both DataArray and DataFrame data)
        '''
        if isinstance(data, xr.DataArray):
            # TODO is there really no simpler way to apply func to these? would be ideal
            # if same interface could be applied to pandas and xarray objects, but the
            # latter don't seem to have either the .apply/.map the pandas objects do
            is_dilution_mask1 = (
                xr.apply_ufunc(parse_log10_conc, arr, vectorize=True) != 0
            )
            # TODO delete (just keeping this old way to check against new way)
            is_dilution2 = ~ (arr.str.endswith('@ 0') | arr.str.endswith('@ 0.0'))
            assert is_dilution2.identical(is_dilution_mask1)
            #
        else:
            # arr should be a pd.Series here
            is_dilution_mask1 = arr.map(parse_log10_conc) != 0

        assert np.array_equal(is_dilution_mask, is_dilution_mask1)
        '''
        #

        # TODO also change this startswith call to a check on full name (after parsing
        # it out w/ other olf fn)?
        return arr.str.startswith(name) & is_dilution_mask

    dilution_rows = None
    dilution_cols = None
    # TODO for DataArray corr input (from al_analysis.plot_corrs, via
    # natmix.viz.plot_corr), w/ coordinates like this:
    # Coordinates:
    #     odor1     (odor) object '1o3ol @ -3' '1o3ol @ -3' ... 'va @ -3' 'va @ -3'
    #     odor2     (odor) object '2h @ -5' '2h @ -5' ... 'solvent' 'solvent'
    #     repeat    (odor) int64 0 1 2 0 1 2 0 1 2 0 1 2 0 1 2 0 1 2 0 1 2 0 1 2 0 1 2
    #     odor1_b   (odor_b) object '1o3ol @ -3' '1o3ol @ -3' ... 'va @ -3' 'va @ -3'
    #     odor2_b   (odor_b) object '2h @ -5' '2h @ -5' ... 'solvent' 'solvent'
    #     repeat_b  (odor_b) int64 0 1 2 0 1 2 0 1 2 0 1 2 ... 0 1 2 0 1 2 0 1 2 0 1 2
    # Dimensions without coordinates: odor, odor_b
    # ...are mix dilutions in odor2[_b] levels getting dropped correctly?
    # (odor2 is only currently ever used for binary in-air mixtures, which will never
    # have the these mixtures as either component, so it is irrelevant)
    for name in mix_names:
        curr_dilution_rows = is_dilution(name, data[odor_var])
        if dilution_rows is None:
            dilution_rows = curr_dilution_rows
        else:
            dilution_rows = dilution_rows | curr_dilution_rows

        # Should currently only be True for correlation matrix DataArray input.
        if hasattr(data, f'{odor_var}_b'):
            curr_dilution_cols = is_dilution(name, data[f'{odor_var}_b'])

            if dilution_cols is None:
                dilution_cols = curr_dilution_cols
            else:
                dilution_cols = dilution_cols | curr_dilution_cols

    if isinstance(data, pd.DataFrame):
        assert dilution_cols is None

        # TODO maybe just do data[dilutions_rows].copy(), cause simpler
        data = data.drop(index=dilution_rows[dilution_rows].index)

        if old_index_levels is not None:
            data = data.set_index(old_index_levels)

            assert np.array_equal(
                # comparing *columns.values would probably also work, but the .shape of
                # each of those was (n,) (with list elements, I think) rather than
                # (n, 3), and I'm more confident in array_equal behavior in latter case
                old_columns.to_frame(index=False), data.columns.to_frame(index=False)
            )
            # to restore column level dtypes
            data.columns = old_columns

        return data
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

