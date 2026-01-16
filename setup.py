"""
Singularity Capital OS - Setup Script
Professional package installer with metadata
"""

from setuptools import setup, find_packages
import os

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r') as f:
            return [line.strip() for line in f 
                    if line.strip() and not line.startswith('#')]
    return ['numpy>=1.21.0', 'pandas>=1.3.0', 'scipy>=1.7.0']

# Read README if exists
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return 'Singularity Capital OS - Self-Evolving Capital Intelligence System'

setup(
    name='singularity-capital-os',
    version='1.0.0',
    author='Singularity Capital',
    author_email='dev@singularitycapital.io',
    description='Self-Evolving Capital Intelligence System for Trading',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/singularitycapital/singularity-os',
    
    py_modules=[
        'singularity_core',
        'multi_agent_system',
        'Complete_Trading_System',
        'xauusd_agent',
        'launcher'
    ],
    
    install_requires=read_requirements(),
    
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Financial and Insurance Industry',
        'Topic :: Office/Business :: Financial :: Investment',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    
    python_requires='>=3.8',
    
    entry_points={
        'console_scripts': [
            'singularity=launcher:main',
            'singularity-core=singularity_core:main',
            'singularity-trader=Complete_Trading_System:main',
        ],
    },
    
    keywords=['trading', 'quantitative', 'bayesian', 'portfolio', 'ai', 'machine-learning'],
    
    project_urls={
        'Bug Reports': 'https://github.com/singularitycapital/singularity-os/issues',
        'Source': 'https://github.com/singularitycapital/singularity-os',
    },
)
