from setuptools import setup
from os import path as os_path

import surf

this_directory = os_path.abspath(os_path.dirname(__file__))


def read_file(filename):
    with open(os_path.join(this_directory, filename), encoding='utf-8') as f:
        long_description = f.read()
    return long_description


def read_requirements(filename):
    return [line.strip() for line in read_file(filename).splitlines()
            if not line.startswith('#')]


setup(
    name='pySURF',
    python_requires='>=3.0',
    version=surf.__version__,
    description="Spatial Uncertainty Research Framework",
    long_description=read_file('README.md'),
    long_description_content_type="text/markdown",
    author="Charles Wang",
    author_email='c_w@berkeley.edu',
    url='https://github.com/NHERI-SimCenter/SURF',
    packages=['surf'],
    zip_safe=False,
    install_requires=read_requirements('requirements.txt'),
    include_package_data=True,
    license="BSD 3-Clause",
    keywords=['random field', 'spatial uncertainty', 'surf', 'surf library', 'surf framework'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
