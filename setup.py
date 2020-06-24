from setuptools import setup
import io

with io.open('README.md', 'r', encoding='utf-8') as readme_file:
    readme = readme_file.read()

setup_args = {
    'name': 'analytic_diffuse',
    'author': 'Adam E. Lanman and Steven G. Murray',
    'url': 'https://github.com/aelanman/analytic_diffuse',
    'license': 'MIT',
    'description': 'Diffuse sky models with exact or series solutions.',
    'long_description': readme,
    'long_description_content_type': 'text/markdown',
    'package_dir': {'analytic_diffuse': 'analytic_diffuse'},
    'packages': ['analytic_diffuse', 'analytic_diffuse.tests'],
    'version': '0.0.1',
    'include_package_data': True,
    'install_requires': ['numpy', 'scipy'],
    'classifiers': ['Development Status :: 3 - Alpha',
                    'Intended Audience :: Science/Research',
                    'License :: OSI Approved :: MIT License',
                    'Programming Language :: Python :: 3.6',
                    'Topic :: Scientific/Engineering :: Physics'],
    'keywords': 'radio astronomy diffuse simulation'
}

if __name__ == '__main__':
    setup(**setup_args)
