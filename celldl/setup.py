from setuptools import setup, find_packages

setup(
    name='CellDL',
    version='0.1.0',
    author='Yin yusong',
    author_email='yyusong526@gmail.com',
    description='CellDL: A Generative Framework for Explicit Distribution Learning in Single-Cell RNA Sequencing',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',  # 所需Python版本
)