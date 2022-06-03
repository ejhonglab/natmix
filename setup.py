
from setuptools import setup, find_packages

setup(
    name='natmix',
    version='0.0.0',
    packages=find_packages(),
    install_requires=[
        # https://stackoverflow.com/questions/32688688
        'hong2p @ git+https://github.com/ejhonglab/hong2p',
    ],
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
