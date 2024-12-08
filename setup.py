from setuptools import find_packages, setup

setup(
    name='w2vSent',  # Package name
    version='0.1.0',  # Version number
    packages=find_packages(),  # Automatically find packages
    python_requires='>=3.8,<3.9',  # Specify Python version requirement
    install_requires=[
        'numpy==1.24.3',
        'scipy==1.10.1',
        'scikit-learn==1.3.0',
        'matplotlib==3.7.2',
        'tqdm==4.66.5',
        'docopt==0.6.2',
        'notebook==7.0.8',
        'sentencepiece==0.2.0',
        'sacrebleu==2.4.3',
        'nltk==3.9.1',
        'timeout_decorator==0.5.0',
        'tensorboard',
        # Exclude torch due to complex installation requirements
    ],
    # You can include other metadata like author, description, etc.
)
