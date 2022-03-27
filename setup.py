import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sac-jax",
    version="0.0.1",
    author="Alicia Fortes Machado & Ivelina Stoyanova & Félix Lefebvre & Ramzi Dakhmouche",
    author_email="aliciafortesmachado@gmail.com",
    description="Jax implementation of Soft Actor Critic.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aliciafmachado/sac",
    project_urls={
        "Bug Tracker": "https://github.com/aliciafmachado/sac/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    package_dir={"sac": "src"},
    # where="src"
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    # install_requires=[
    #             'gym',
    #             'matplotlib',
    #             'numpy',
    #             'jax',
    #             'dm-acme',
    #             'chex',
    #             'dm_env',
    #         ],
)

print(setuptools.find_packages())