# setup.py

from setuptools import setup, find_packages

setup(
    name='quantcli',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'Click',
        'requests',
        'pdfplumber',
        'spacy',
        'openai',
        'python-dotenv',
        'pygments',
        'InquirerPy',
    ],
    entry_points='''
        [console_scripts]
        quantcli=quantcli.cli:cli
    ''',
    author='SL-MAR',
    author_email='your.email@example.com',
    description='A CLI tool for generating QuantConnect algorithms from research articles.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/quantcli',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
