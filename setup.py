from setuptools import setup, find_packages

NAME = 'LongitudeML'
VERSION = '0.0'
AUTHOR = "Adithya V Ganesan"
AUTHOR_EMAIL = "avirinchipur@cs.stonybrook.edu"
DESCRIPTION = ""
LONG_DESCRIPTION=open("README.md", "r", encoding="utf-8").read()
LONG_DESCRIPTION_CONTENT_TYPE="text/markdown"
KEYWORDS="Deep Learning, Longitudinal Modeling, Time Series Forecasting, PyTorch"
LICENSE="Apache License 2.0"
URL = "https://github.com:adithya8/LongitudeML"
DOWNLOAD_URL = "https://github.com:adithya8/LongitudeML"

INSTALL_REQUIRES = []
PACKAGES_DIR = {"LongitudeML": "src"}
PACKAGES = find_packages(where="src")
PACKAGE_DATA = {"": ["**/*.cu", "**/*.cpp", "**/*.cuh", "**/*.h", "**/*.pyx", "**/*.json"]}
INCLUDE_PACKAGE_DATA = True
EXTRAS_REQUIRE = {}
PYTHON_VERSION = ">=3.8"

CLASSIFIERS=[
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: Apache Software License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
SCRIPTS = []

if __name__ == "__main__":
    setup(name=NAME,
        version=VERSION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
        keywords=KEYWORDS,
        license=LICENSE,
        url=URL,
        download_url=DOWNLOAD_URL,
        install_requires=INSTALL_REQUIRES,
        package_dir=PACKAGES_DIR,
        packages=PACKAGES,
        package_data=PACKAGE_DATA,
        include_package_data=INCLUDE_PACKAGE_DATA,
        extras_require=EXTRAS_REQUIRE,
        python_requires=PYTHON_VERSION,
        classifiers=CLASSIFIERS,
        # scripts=SCRIPTS
        )

