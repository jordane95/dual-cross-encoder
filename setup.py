from setuptools import setup, find_packages

setup(
    name='mvdr',
    version='0.0.1',
    packages=find_packages("src"),
    package_dir={'': 'src'},
    license='Apache 2.0',
    description='mvdr: A toolkit for training multi-vector dense retrieval models.',
    python_requires='>=3.7',
    install_requires=[
        "transformers>=4.10.0",
        "datasets>=1.1.3",
        "faiss_cpu",
        "pyserini",
        "pytrec_eval",
    ]
)
