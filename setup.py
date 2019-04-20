from io import open
from setuptools import find_packages, setup

setup(
    name="bert_score",
    version="0.1.0",
    author="Tianyi Zhang*, Varsha Kishore*, Felix Wu*, Kilian Q. Weinberger, and Yoav Artzi",
    author_email="tzhang@asapp.com",
    description="PyTorch implementation of BERT score",
    long_description=open("README.md", "r", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    keywords='BERT NLP deep learning google metric',
    license='MIT',
    url="https://github.com/Tiiiger/bert_score",
    packages=find_packages(exclude=["*.tests", "*.tests.*",
                                    "tests.*", "tests"]),
    install_requires=['torch>=0.4.1',
                      'numpy',
                      'requests',
                      'tqdm',
                      'matplotlib',
                      'pytorch-pretrained-bert>=0.6.1'],
    entry_points={
        'console_scripts': [
            "bert-score=cli.score:main",
        ]
    },
    # python_requires='>=3.5.0',
    tests_require=['pytest'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],

)
