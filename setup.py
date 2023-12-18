import sys
import warnings

try:
    from setuptools import setup
    from Cython.Distutils import build_ext, Extension
    HAVE_CYTHON = True
except ImportError as e:
    warnings.warn(e.args[0])
    from setuptools import setup, Extension
    from setuptools.command.build_ext import build_ext
    HAVE_CYTHON = False


class CustomBuildExtCommand(build_ext):
    """build_ext command for use when numpy headers are needed."""
    def run(self):
        import numpy
        self.include_dirs.append(numpy.get_include())
        super().run()


def readme():
    with open('README.rst') as readme_file:
        return readme_file.read()


def requirements():
    # The dependencies are the same as the contents of requirements.txt
    with open('requirements.txt') as f:
        return [l for line in f if (l := line.strip())]


def make_extension(name, parallel=False, cpp=False):
    kwargs = {}
    link_args = []
    compile_args = ['-Wno-deprecated-declarations', '-Wno-unreachable-code', '-Wno-uninitialized']
    if parallel:
        if sys.platform == 'darwin':
            # compile_args.append('-Xpreprocessor')
            link_args.append('-lomp')
        else:
            compile_args.append('-fopenmp')
            link_args.append('-fopenmp')
    if cpp:
        # compile_args += ['-std=c++11', '-stdlib=libc++']
        compile_args.append('-std=c++11')
        # link_args.append('-stdlib=libc++')
        kwargs['language'] = 'c++'
    return Extension(
        name=f'hdbscan.{name}',
        sources=[f'hdbscan/{name}.pyx'],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        cython_directives={'language_level': 3},
        **kwargs,
    )


extensions = [make_extension(n) for n in (
    '_hdbscan_linkage',
    '_hdbscan_boruvka',
    '_prediction_utils',
)]
extensions += [make_extension(name=n, parallel=True) for n in (
    '_hdbscan_reachability',
    'dist_metrics',
)]
extensions.append(make_extension('_hdbscan_tree', parallel=True, cpp=True))

data_files = [f'hdbscan/{x}' for x in (
    'dist_metrics.pxd',
    'disjointsets.pxd',
)]


configuration = {
    'name': 'hdbscan',
    'version': '0.8.26',
    'description': 'Clustering based on density with variable density clusters',
    'long_description': readme(),
    'classifiers': [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved',
        'Programming Language :: C',
        'Programming Language :: Python',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3.10',
    ],
    'keywords': 'cluster clustering density hierarchical',
    'url': 'http://github.com/scikit-learn-contrib/hdbscan',
    'maintainer': 'Leland McInnes',
    'maintainer_email': 'leland.mcinnes@gmail.com',
    'license': 'BSD',
    'packages': ['hdbscan', 'hdbscan.tests'],
    'install_requires': requirements(),
    'ext_modules': extensions,
    'cmdclass': {'build_ext': CustomBuildExtCommand},
    'test_suite': 'nose.collector',
    'tests_require': ['nose'],
    'data_files': data_files,
}

if not HAVE_CYTHON:
    warnings.warn('Due to incompatibilities with Python 3.7 hdbscan now'
                  'requires Cython to be installed in order to build it')
    raise ImportError('Cython not found! Please install cython and try again')

setup(**configuration)
