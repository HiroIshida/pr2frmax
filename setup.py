from setuptools import find_packages, setup

setup(
    name="pr2dmp",
    version="0.0.0",
    description="pr2dmp",
    author="Hirokazu Ishida",
    author_email="h-ishida@jsk.imi.i.u-tokyo.ac.jp",
    install_requires=[],
    packages=find_packages(),
    package_data={"pr2dmp": ["py.typed"]},
)
