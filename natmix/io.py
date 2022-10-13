
from pathlib import Path
import pickle
import warnings
from typing import Optional

import pandas as pd
import xarray as xr

from hong2p.olf import (load_stimulus_yaml, odor_lists_to_multiindex, parse_odor,
    odor2abbrev
)
from hong2p.types import Pathlike

from natmix.olf import get_panel, panel2name_order


# TODO TODO migrate all the pickles to netcdf -> remove pickle code

# TODO TODO factor out functions to just check the data types / format(maybe those
# should go in in a checks or util module tho)

REMY_STIMFILE_DIR = Path(
    '/mnt/matrix/Remy-Data/projects/natural_mixtures/olfactometer_configs'
)

# TODO move to natmix.olf
remy2natmix_odor_names = {
    'IaOH': 'IAol',
    'IaA': 'IAA',
    'ea': 'EA',
    'eb': 'EB',
    # TODO TODO TODO handle new kmix/cmix from here too
    #'': '',
    # TODO what does she call this?
    #'': '~kiwi',
    # TODO do control odors too
}

# TODO delete add_missing_panel_odors if it doesn't end up being useful / necessary when
# sharing plot_corrs with stuff load this way
def load_remy_corr(path: Path, drop_repeats_beyond: Optional[int] = 3,
    drop_first: bool = False, drop_nonpanel_odors: bool = True,
    add_missing_panel_odors: bool = False) -> xr.DataArray:
    """
    Args:
        path: path to one of Remy's single-recording netCDF correlation DataSet files

        drop_repeats_beyond: None to return all repeats. if an int, will drop any
            repeats greater than this (0 indexed, as my repeat indexing).

        drop_first: only relevant if drop_repeats_beyond is an int. if True, drops the
            first drop_repeats_beyond presentations, else drops the last
            drop_repeats_beyond presentations.

        drop_nonpanel_odors: drops odors whose name is not listed among odors (in
            natmix.olf.get_panel) for the detected panel

        add_missing_panel_odors: adds any missing odors for the detected panel (with
            corresponding data set to NaN)
    """
    # TODO maybe add kwarg to clear all attrs (to not distract in comparisons to my
    # data / not accidentally rely on them)

    # Fc = suite2p term: Fluorescence - 0.7 * Fneu
    # All correlations Pearson by default.
    dataset = xr.load_dataset(path)
    attrs = dataset.attrs
    dataarray = dataset['Fc_zscore']
    dataarray.attrs.update(dataset.attrs)

    def odor_strs_to_multiindex(odor_strs):

        # TODO TODO TODO need to handle mixtures here too (w/ '+' in them, for
        # example. how does remy format that? does she even have that data?)
        odor_dicts = [parse_odor(o) for o in odor_strs]

        for o in odor_dicts:
            name = o['name']
            if name in remy2natmix_odor_names:
                o['name'] = remy2natmix_odor_names[name]

        # All length one lists indicating no gas phase mixtures (for now).
        odor_lists = [[o] for o in odor_dicts]

        return odor_lists_to_multiindex(odor_lists), odor_lists

    row_odor_index, odor_lists = odor_strs_to_multiindex(dataarray.stim_row.values)
    col_odor_index, _          = odor_strs_to_multiindex(dataarray.stim_col.values)

    # if i relax this, couldn't just use one of the odor_lists from the helper.
    assert row_odor_index.identical(col_odor_index)

    col_odor_index.names = [f'{n}_b' for n in col_odor_index.names]

    dataarray = dataarray.rename({'trial_row': 'odor', 'trial_col': 'odor_b'})
    dataarray = dataarray.assign_coords(odor=row_odor_index, odor_b=col_odor_index)
    dataarray = dataarray.drop_vars(['stim_row', 'stim_col'])


    # TODO TODO TODO compute is_pair[_b]
    # (figure out how remy encodes is_pair, if she even has any of that data)

    # TODO could get stimuli YAML files from Remy, load them here, and compute
    # is_pair/panel that way, if nothing else
    is_pair = False

    attrs = dataarray.attrs

    date_str = attrs['date_imaged']
    fly_num = attrs['fly_num']
    thorimage = attrs['thorimage']

    panel = 'kiwi' if 'kiwi' in attrs['thorimage'] else 'control'

    meta = {
        'date': pd.Timestamp(date_str),
        'fly_num': fly_num,
        'thorimage_id': thorimage,

        # TODO TODO TODO compute panel
        # TODO do i have a fn for going from odors to panel?
        #
        # (just assuming remy is only gonna give me one of these two types of data for
        # now, and that she always named roughly how i did)
        'panel': panel,

        # TODO TODO is movie_type what i need to get pair recording status?
        # (e.g. 'kiwi')
        #attrs['movie_type']

        # TODO what is good_xid? why start at 2 in first test file?
        # TODO TODO factor out this to hong2p.xarray thing (also in
        # al_analysis.process_experiment.add_metadata)
        'is_pair': ('odor', [is_pair] * dataarray.sizes['odor']),
        'is_pair_b': ('odor_b', [is_pair] * dataarray.sizes['odor_b']),
    }
    for k, v in meta.items():
        dataarray[k] = v

    # TODO perhaps also factor this along w/ part mentioned inside meta creation
    dataarray = dataarray.set_index(odor=['is_pair'], append=True).set_index(
        odor_b=['is_pair_b'], append=True
    )

    # TODO TODO TODO want to get all these from remy then, and organize in a standard
    # place
    # TODO TODO TODO or/also use any mismatches as a means of finding which odors to
    # replace w/ my standard names + replace data w/ NaN for (e.g. to fix the case where
    # remy recorded bad 'kiwi' data, and then renamed it to 'kiwi_no_etoh')
    # TODO TODO TODO if i use plot_corrs for this too, may want to adapt it's N
    # calculation to also be able to plot a grid of the N at each point on the
    # correlation matrix (so that i can NaN out some artifacts from some flies, at least
    # for some internal calculations of mean correlation matrices, using some data from
    # flies w/ motion artifacts / other transient issues)
    stimfile_path = REMY_STIMFILE_DIR / dataarray.olf_config
    yaml_data, yaml_odor_lists = load_stimulus_yaml(stimfile_path)
    yaml_index = odor_lists_to_multiindex(yaml_odor_lists)

    assert len(yaml_odor_lists) == len(odor_lists)
    for yaml_trial_odors, trial_odors in zip(yaml_odor_lists, odor_lists):
        to_compare = []
        for odor in yaml_trial_odors:
            # Getting rid of 'abbrev' and extra stuff that might be in YAML config
            to_compare.append(
                {k: v for k, v in odor.items() if k in ('name', 'log10_conc')}
            )

        # TODO check this still evaluates True if log10_conc only varies in int/float
        # type (e.g. 2 vs 2.0). if not, maybe implement my own equality check.
        if trial_odors != to_compare:
            warnings.warn(f'{date_str}/{fly_num}/{thorimage} odor mismatch. '
                f'Remy: {trial_odors}, YAML: {to_compare}'
            )

    # TODO delete
    da = dataarray
    #

    # TODO probably rename this to add_missing_yaml_odors
    if add_missing_panel_odors:
        raise NotImplementedError
        # TODO should i implement this by just the different wrt yaml, or should i
        # actually enumerate panel odors (+ concentrations?) (from something in
        # natmix.olf?)
        import ipdb; ipdb.set_trace()

    # TODO should this be dropping based just on the name? or also concentration?
    # i'm not sure i currently have the names enumerated WITH concentrations anywhere
    # as-is.
    # TODO rename to drop_nonyaml_odors (probably not?)?
    if drop_nonpanel_odors:

        name_order = panel2name_order[panel]
        import ipdb; ipdb.set_trace()

        # TODO TODO did i even need to do this? can't i just compare the indices to each
        # other? ig i do need to just get name(s) one way or another...
        # TODO factor this bit to hong2p.olf?
        abbreved_odor_lists = []
        for trial_odors in odor_lists:

            abbreved_trial_odors = []
            for odor in trial_odors:
                abbreved_odor = dict(odor)

                name = odor['name']
                if name in odor2abbrev:
                    odor2abbrev[odor['name']]

                abbreved_trial_odors.append(abbreved_odor)

            abbreved_odor_lists.append(abbreved_trial_odors)

        # TODO to handle mixtures (i.e. where odor2 is not all 'solvent'), would
        # probably want to get the set of names at each trial, and compare those between
        # loading-from-YAML and calculating-from-Remy's-odor-strings sources
        import ipdb; ipdb.set_trace()

    # TODO TODO TODO probably drop non-recognized odors / pad to full set of panel
    # odors, maybe filling in NaN for data in places where remy just e.g. changed the
    # name of the odor ('kiwi' -> 'kiwi_no_etoh', since she forgot to add the ethanol).

    if drop_repeats_beyond is not None:
        import ipdb; ipdb.set_trace()

    # TODO TODO TODO also drop all but 3 repeats of an odor (or don't, controllable via
    # kwarg) (+ allow picking either first 3, last 3, or averaging all, via kwarg)

    # TODO will these all need to be sorted into a particular order (along the odor
    # indices within one call of this fn), before concatenating? or will they be
    # automatically aligned just fine anyway?

    return dataarray


# default = 'netcdf4'
# need to `pip install h5netcdf` for this
#netcdf_engine = 'h5netcdf'
# TODO explicitly specify an engine in xr call (w/ opt to override via kwarg), in xr
# load call, to ensure compat in case there are any differences. same w/ output.
# (or gauranteed to be load the same across engines?)
def load_corr_dataarray(path: Pathlike) -> xr.DataArray:
    """Loads xarray object and restores odor MultiIndex (['odor1','odor2','repeat'])
    """
    path = Path(path)

    # NOTE: no longer using netcdf as it's had an obtuse error when trying to use it tot
    # serialize across-fly data:
    # "ValueError: could not broadcast input array from shape (2709,21) into shape
    # (2709,31)"
    # (related to odor index, it seems, as 2709 is the length of that)

    # TODO see if other drivers for loading are faster
    #arr = xr.load_dataarray(path, engine=netcdf_engine)
    #return arr.set_index({'odor': ['odor1', 'odor2', 'repeat']})

    with open(path, 'rb') as f:
        #return pickle.load(f)
        data = pickle.load(f)

    # TODO TODO TODO finish / fix / factor out tests
    '''
    # TODO delete/refactor (break current test data to unit tests)
    remy_dir = Path('~/src/al_analysis/remy').expanduser()

    new_dir = remy_dir / 'cells_x_trials'

    path1 = (new_dir / '2022-04-09__fly01__kiwi' /
        '2022-04-09__fly01__kiwi__trialRDM__mean_peak__correlation.nc'
    )
    path2 = (new_dir /
        '2022-07-12__fly05__kiwi_components_again_with_partial_and_probes' /
        ('2022-07-12__fly05__kiwi_components_again_with_partial_and_probes__trialRDM'
        '__mean_peak__correlation.nc'
        )
    )
    c1 = load_remy_corr(path1)
    c2 = load_remy_corr(path2)

    import ipdb; ipdb.set_trace()
    '''
    return data


def write_corr_dataarray(arr: xr.DataArray, path: Pathlike) -> None:
    """Writes xarray object with odor MultiIndex (w/ levels ['odor1','odor2','repeat'])
    """
    # Only doing reset_index so we can serialize the DataArray via netCDF (recommended
    # format in xarray docs). If odor MultiIndex is left in, it causes an error which
    # says to reset_index() and references: https://github.com/pydata/xarray/issues/1077
    #arr.reset_index('odor').to_netcdf(path, engine=netcdf_engine)
    # TODO delete eventually
    #assert arr.identical(load_corr_dataarray(path, engine=netcdf_engine))
    #
    with open(path, 'wb') as f:
        # "use the highest protocol (-1) because it is way faster than the default text
        # based pickle format"
        pickle.dump(arr, f, protocol=-1)

    # TODO also write as netcdf, in a standard format (mainly for more interop across
    # versions of xarray or whatever)?

