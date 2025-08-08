from setuptools import setup, find_packages

setup(
    name='nca-package',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
    ],
    author='Hubert',
    author_email='',
    description='An implementation of the Network Component Analysis',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/your-username/nca-package',
)
