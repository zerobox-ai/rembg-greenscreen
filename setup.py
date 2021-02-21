import pathlib

from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

with open("requirements.txt") as f:
    requireds = f.read().splitlines()

setup(
    name="rembg-greenscreen",
    version="2.1.2",
    description="Rembg Virtual Green Screen",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ecsplendid/rembg-greenscreen",
    author="Tim Scarfe",
    author_email="tim@developer-x.com",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="remove, background, u2net, greenscreen, green screen, matte",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8, <4",
    install_requires=requireds,
    entry_points={
        "console_scripts": [
            "greenscreen=rembg.cmd.cli:main"
        ],
    },
)
