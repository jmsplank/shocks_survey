from setuptools import setup, find_packages

with open("requirements.txt", "r") as file:
    requirements = file.read().splitlines()
print(find_packages())
setup(
    name="shocksurvey",
    version="1",
    author="JP",
    author_email="jp5g16@soton.ac.uk",
    description="Functions for use in MMS Shocks Survey",
    packages=find_packages(),
    install_requires=requirements,
)
