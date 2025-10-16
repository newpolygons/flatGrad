import setuptools

with open('README.md', 'r') as f:
    longdesc = f.read()


setuptools.setup(
    name='flatgrad',
    version='0.0.1',
    author='newpolygons',
    description='Another neural network library in Python. I wouldnt use this. Looking to learn something here.',
    long_description=longdesc, 
    long_description_content_type='text/markdown', 
    url='https://github.com/newpolygons/flatgrad',
    packages=setuptools.find_packages(),
    install_requires=[
        "graphviz",
        "numpy",
        "matplotlib",

    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Lisence :: GPLv3',
        'Operating System :: No Bill Gates',
    ],
    python_requires='>=3.10',

)