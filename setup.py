import site
site.ENABLE_USER_SITE=True

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="xmot",
    version="0.0.1",
    # author="Example Author",
    # author_email="author@example.com",
    description="XPCI multi object tracker",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/velatkilic/mot",
    # project_urls={
    #     "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    # },
    # classifiers=[
    #     "Programming Language :: Python :: 3",
    #     "License :: OSI Approved :: MIT License",
    #     "Operating System :: OS Independent",
    # ],
    #package_dir={"": "xmot"},
    #packages=setuptools.find_packages(where="xmot"),
    packages= ["xmot", "xmot.analyzer", "xmot.datagen", "xmot.digraph", "xmot.mot"],
    # python_requires=">=3.6",
    install_requires = [
        "numpy",
        "scipy",
        "torch",
        "torchvision",
        "opencv-contrib-python",
        "imageio",
        "imageio-ffmpeg",
        # "crefi",
        "pycocotools",
        "Pillow",
        "click",
        "matplotlib",
        "scikit-image"
    ],

    extras_require = {
        "dev": [
            "pytest",
            "twine",
        ],
    },
)
