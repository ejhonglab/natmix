
from setuptools import setup, find_packages


try:
    import hong2p
    # Just to support installing a local copy for development.
    install_requires = []

except ImportError:
    install_requires = [
        # 2023-07-12: commented just b/c conflict installing current
        # al_analysis/requirements.txt
        #
        # TODO how to get this to not cause pip dependency resolver to fail when i want
        # to use an editable local install? and ideally without breaking the ability to
        # make wheels... (e.g. by adding a conditional checking if hong2p is installed,
        # like i'm currently doing)
        #
        # https://stackoverflow.com/questions/32688688
        'hong2p @ git+https://github.com/ejhonglab/hong2p',
        #
        # was trying to adapt Anfernee's answer from:
        # https://stackoverflow.com/questions/35668295
        # (doesn't work)
        ##'hong2p @ file://localhost/../hong2p',
    ]

setup(
    name='natmix',
    version='0.0.0',
    packages=find_packages(),
    install_requires=install_requires,
    extras_require={
        'dev': [
            'pre-commit',
            'pytest',
            'ipdb',

            'sphinx',
            'sphinx-rtd-theme',
            'sphinxcontrib-apidoc',
            'sphinx-prompt',
            'sphinx-autodoc-typehints',
        ],
    },
    author="Tom O'Connell",
    author_email='toconnel@caltech.edu',
    license='GPLv3',
    url='https://github.com/ejhonglab/natmix',
)
