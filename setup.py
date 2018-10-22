#!/usr/bin/python
"""multikernel setup script.

Author: Federico Tomasi, Vanessa D'Amario
Copyright (c) 2018, Federico Tomasi, Vanessa D'Amario.
Licensed under the BSD 3-Clause License (see LICENSE.txt).
"""

from setuptools import find_packages, setup

from multikernel import __version__ as version

setup(
    name='multikernel',
    version=version,

    description=('MKL (Multiple Kernel Learning)'),
    long_description=open('README.md').read(),
    author='Federico Tomasi, Vanessa D\'Amario',
    author_email='federico.tomasi@dibris.unige.it, vanessa.damario@dibris.unige.it',
    maintainer='Federico Tomasi, Vanessa D\'Amario',
    maintainer_email='federico.tomasi@dibris.unige.it', 'vanessa.damario@dibris.unige.it',
    url='https://github.com/fdtomasi/multikernel',
    download_url='https://github.com/fdtomasi/multikernel/archive/'
                 'v%s.tar.gz' % version,
    keywords=['kernel', 'machine learning', 'time series', 'epilepsy', 'wavelet'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Programming Language :: Python',
        'License :: OSI Approved :: BSD License',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Natural Language :: English',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python'
    ],
    license='FreeBSD',
    packages=find_packages(exclude=["*.__old", "*.tests"]),
    include_package_data=True,
    requires=['numpy (>=1.11)',
              'scipy (>=0.16.1,>=1.0)',
              'sklearn (>=0.17)',
              'pywt',
              'six'],
)
