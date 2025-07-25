from setuptools import setup, find_packages

setup(
    name='fashion-classification-cnn',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A CNN model for fashion classification using a dataset from Google Drive.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'tensorflow>=2.0.0',
        'numpy>=1.18.0',
        'pandas>=1.0.0',
        'matplotlib>=3.0.0',
        'scikit-learn>=0.22.0',
        'PyYAML>=5.1',
        'opencv-python>=4.0.0'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)