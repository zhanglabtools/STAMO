from setuptools import Command, find_packages, setup

__lib_name__ = "STAMO"
__lib_version__ = "1.0.0"
__description__ = "STAMO is designed for diagonal integration of unpaired spatial multi-omics data"
__url__ = "https://github.com/zhanglabtools/STAMO"
__author__ = "Xiang Zhou"
__author_email__ = "xzhou@amss.ac.cn"
__license__ = "MIT"
__keywords__ = ["unpaired spatial multi-omics", "diagonal integration", "graph attention neural network", "spatial domain", "cross-omics generation"]
__requires__ = ["requests",]

setup(
    name = __lib_name__,
    version = __lib_version__,
    description = __description__,
    url = __url__,
    author = __author__,
    author_email = __author_email__,
    license = __license__,
    ## packages = ['STAMO'],
    packages = find_packages(include=['STAMO', 'STAMO.*']),
    install_requires = __requires__,
    zip_safe = False,
    include_package_data = True,
)
