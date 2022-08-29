#!/usr/bin/env python3

from pprint import pprint

import pandas as pd
import matplotlib.pyplot as plt

from hong2p import viz

# TODO use
#from natmix import plot_corr


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
    fig.savefig('hallem_panel.png')

    viz.matshow(df.T.corr())

    # TODO save above figs

    panel2odors = {
        'kiwi': [
            'ethanol',
            # This one is synonymous with 'isoamyl alcohol' / '3-methyl-1-butanol'
            #'3-methylbutanol',
            'isoamyl alcohol',
            # This one is synonymous with 'isoamyl acetate'.
            # https://pubchem.ncbi.nlm.nih.gov/compound/Isoamyl-acetate
            #'isopentyl acetate',
            'isoamyl acetate',
            'ethyl acetate',
            'ethyl butyrate',
        ],
        'control': [
            # Synonymous with 'valeric acid'
            #'pentanoic acid',
            'valeric acid',
            'methyl salicylate',
            'furfural',
            '2-heptanone',
            '1-octen-3-ol',
        ],
    }
    has_closer_conc = {
        # -3.5
        'ethyl butyrate': [-4],
        # -4.2
        'ethyl acetate': [-4],

        # -5. could try either -4 or -6
        '2-heptanone': [-4, -6],
        # -3. could try -4, but -2 might be closer...
        '1-octen-3-ol': [None, -4],
    }

    for panel, podors in panel2odors.items():
        podors_closer_concs = []
        for o in podors:
            if o in has_closer_conc:
                for c in has_closer_conc[o]:
                    ostr = f'{o} {c}' if c is not None else o
                    podors_closer_concs.append(ostr)
            else:
                podors_closer_concs.append(o)

        pprint(podors)
        pprint(podors_closer_concs)
        print()

        pdf = df.loc[podors]
        pdf = pd.DataFrame(index=pdf.index.drop_duplicates(), data=pdf.drop_duplicates())
        # TODO deal w/ cause of eta being duplicated

        f1, _ = viz.matshow(pdf.T.corr())
        f1.savefig(f'{panel}_all_default_minus2.png')

        pdf = df.loc[podors_closer_concs]
        pdf = pd.DataFrame(index=pdf.index.drop_duplicates(), data=pdf.drop_duplicates())

        f2, _ = viz.matshow(pdf.T.corr())
        f2.savefig(f'{panel}_with_closer.png')
        # TODO savefig

    plt.show()
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()

