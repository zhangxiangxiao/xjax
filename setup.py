import setuptools

setuptools.setup(
    name="xjax",
    version="0.0.1",
    author="Xiang Zhang",
    author_email="fancyzhx@gmail.com",
    description="A simple JAX framework for neural networks",
    url="https://github.com/zhangxiangxiao/xjax",
    project_urls={
        "Bug Tracker": "https://github.com/zhangxiangxiao/xjax/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD 3-Clause License",
        "Operating System :: OS Independent",
    ],
    packages=['xjax'],
    python_requires=">=3.8",
    install_requires=['jax>=0.3.7'],
)
