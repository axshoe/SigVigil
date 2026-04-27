from setuptools import setup, find_packages
setup(
    name="sigvigil",
    version="1.0.0",
    packages=find_packages(include=["sigvigil", "sigvigil.*"]),
)
