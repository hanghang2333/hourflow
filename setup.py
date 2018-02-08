#!/usr/bin/env python
# coding=utf-8

from setuptools import setup, find_packages

setup(
    name='hourflow',
    version=0.1,
    description=(
        'A simple test libiary for deeplearning study'
    ),
    long_description=open('README.rst').read(),
    author='lihang',
    author_email='1092978787@qq.com',
    maintainer='lihang',
    maintainer_email='1092978787@qq.com',
    license='BSD License',
    packages=find_packages(),
    platforms=["all"],
    url='https://github.com/hanghang2333/hourflow.git',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: Implementation',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries'
    ],
)