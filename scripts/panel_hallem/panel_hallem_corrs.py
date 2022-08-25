#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

from hong2p import viz


def main():
    # TODO update drosolf to allow loading the lower conc + fruit data too, then use
    # that instead (+ make sure it's using this CSV, or at least not that one that had a
    # row shifted for one odor somewhere)
    hdf = pd.read_csv('hc_data.csv', skiprows=1)
    hdf = hdf.drop(columns='class').set_index('odor')
    hdf = hdf.iloc[:-1, :] + hdf.loc['spontaneous firing rate']

    # As is, receptors are the columns for hdf
    glomeruli = pd.read_csv('hc_data.csv', nrows=0).columns[2:]

    odors = [
        'ethyl butyrate',
        'ethyl acetate',
        # This one is synonymous with 'isoamyl acetate'.
        # https://pubchem.ncbi.nlm.nih.gov/compound/Isoamyl-acetate
        'isopentyl acetate',
        # This one is synonymous with 'isoamyl alcohol' / '3-methyl-1-butanol'
        '3-methylbutanol',
        'ethanol',

        '1-octen-3-ol',
        '2-heptanone',
        'furfural',
        # Synonymous with 'valeric acid'
        'pentanoic acid',
        'methyl salicylate',

        # TODO TODO add all diagnostic odors too
        'E2-hexenal',
        'ethyl acetate',
        'methyl acetate',
        'methyl octanoate',
        'ethyl 3-hydroxybutyrate',
        '2,3-butanedione',
        'ethyl lactate',
        'ethyl trans-2-butenoate',
        # (acetoin not in Hallem)
        '2-butanone',
    ]
    for o in odors:
        assert o in hdf.index, o

    hdf.columns = glomeruli

    # Either just as above, or with concentration at end like 'ethyl acetate -2'
    odor_rows = [r for o in odors for r in hdf.index if r.startswith(o)]
    df = hdf.loc[odor_rows, :]

    # TODO TODO TODO separate plot where i enumerate all glomeruli around a few
    # ambiguous ones, and then plot all Hallem (or ideally DoOR) data for them
    # (to try to manually identify diagnostic odors that might work)

    renames = {
        'isopentyl acetate': 'isoamyl acetate',
        # This one is synonymous with 'isoamyl alcohol' / '3-methyl-1-butanol'
        '3-methylbutanol': 'isoamyl alcohol',
        'pentanoic acid': 'valeric acid',
        'E2-hexenal': 'trans-2-hexenal',
        'ethyl trans-2-butenoate': 'ethyl crotonate',
    }
    for old, new in renames.items():
        df.index = df.index.str.replace(old, new)

    # TODO  try sorting by avg/max activation to these odors?
    # TODO is DM1 really not in hallem?
    df.sort_index(axis='columns', inplace=True)

    fig, _ = viz.matshow(df, dpi=300)
    fig.savefig('panel_hallem.png')

    viz.matshow(df.T.corr())

    # TODO savefigs

    plt.show()
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()

