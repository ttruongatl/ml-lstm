from setuptools import setup, find_packages

setup(
    name='mllstm',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'tensorflow==2.0.0',
        'matplotlib==3.1.1',
        'pandas==0.25.3'
    ],
    setup_requires=[],
    tests_require=[
        'pytest==2.9.2',
        'coverage',
        'pytest-cov',
        'pytest-xdist',
    ],
    entry_points='''
    ''',
)
