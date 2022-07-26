#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

from hong2p import viz


def main():
    # TODO update drosolf to allow loading the lower conc + fruit data too, then use
    # that instead (+ make sure it's using this CSV, or at least not that one that had a
    # row shifted for one odor somewhere)
    hdf = pd.read_csv('hc_data.csv', skiprows=1)
    hdf = hdf.drop(columns='class').set_index('odorant')
    hdf = hdf.iloc[:-1, :] + hdf.loc['spontaneous firing rate']

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
    ]
    for o in odors:
        assert o in hdf.index, o

    # Either just as above, or with concentration at end like 'ethyl acetate -2'
    odor_rows = [r for o in odors for r in hdf.index if r.startswith(o)]
    df = hdf.loc[odor_rows, :]

    renames = {
        'isopentyl acetate': 'isoamyl acetate',
        # This one is synonymous with 'isoamyl alcohol' / '3-methyl-1-butanol'
        '3-methylbutanol': 'isoamyl alcohol',
        'pentanoic acid': 'valeric acid',
    }
    for old, new in renames.items():
        df.index = df.index.str.replace(old, new)

    viz.matshow(df)

    viz.matshow(df.T.corr())

    plt.show()
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()

