# Release Process

## Purpose
This document serves as a reminder for jsonpickle developers on how to tag, build, and publish jsonpickle releases for Git, GitHub and PyPi. All commands included are meant for a Linux system with the appropiate prerequisites met (such as Python being installed, Git being installed, etc.) The instructions here mostly parallel those in garden.yaml, but are meant for quicker and more intuitive reference for those without garden installed.

## Prerequisites
- Clear staging area of irrelevant files
- Clear pre-existing files from build/ and dist/
- Make sure everything important has been committed and pushed already
- Run tests and **ensure everything passes**

## Git
```sh
git tag -a vx.y.z
git push --tags origin main
```

# GitHub
GitHub releases are created automatically by GitHub Actions when the git tag is pushed.

# PyPi
```sh
# create wheels
python3 -m build -n .

# create attestations (use github account)
python3 -m pypi_attestations sign dist/*

# publish to PyPi
twine upload --attestations dist/*.whl dist/*.tar.gz
```
