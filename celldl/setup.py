from setuptools import setup, find_packages

setup(
    name='CellDL',
    version='0.1.0',
    author='Yin yusong',
    author_email='yyusong526@gmail.com',
    description='CellDL: Defining Cell Identity by Learning Transcriptome Distributions from Single-Cell Data',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)