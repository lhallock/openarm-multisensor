# Developing on this repository

This README describes the workflow expected of contributors to this repository to maintain code cleanliness, legibility, and accuracy.

Note that the documentation below assumes a Unix-based system. If you're using Windows, godspeed.

## Workflow (AKA how to commit code once everything is set up)

Please execute the following steps each time you want to work on this repository. This section is intended as a quick reference; these commands are documented more fully in the sections below, which should be consulted the first few times you work with this repository.

**Every time you open up the code to work**, run

```bash
source venv/bin/activate
```

to make sure you're using the correct Python instance. Then, create or rejoin your development branch via

```bash
git checkout -b [your_new_branch]
```

Omit `-b` if you're checking out a branch you've already created, and follow all branch naming conventions below.

Develop all the code you want. When you're ready to commit, execute

```bash
make format
make lint
```

Address all comments made by the linter, then re-run these two commands until they come up clean. Then run

```bash
make test
make coverage
```

Ensure all your tests pass, and that coverage is reasonably high. Last, run

```bash
pip freeze > requirements.txt
```

to propagate any installed packages to the requirements file.

Add all desired files to your commit (e.g., using `git add .` and `git status` to check), and commit your changes via

```bash
git commit -m "[descriptive commit message]"
```

When you're ready to merge all your changes into master, execute a pull request using the instructions [here](https://help.github.com/en/articles/creating-a-pull-request#creating-the-pull-request). Your code will then be reviewed and approved and/or changes will be requested.

Once the pull request is resolved and your changes merged into master, your old branch will be deleted. You should then make sure your local copy of the repository is up-to-date (`git pull`), create a new branch, and keep developing!

## Set Up (AKA how to use this repository the first time)

### Clone this repository and work on your own branch

Run

```bash
git clone https://github.com/lhallock/openarm-multisensor.git
git checkout -b [your_new_branch]
```

**Everyone should be developing on their own branch, always.** Please name your branch `[your_git_username]-dev/[topic]` (e.g., if I were working on a feature to visualize the data, I might use `lhallock-dev/viz`). Be as specific as possible while keeping topic names short and sweet.

When you're ready to merge into `master`, execute a pull request. (This is a stupid legacy name; it's really a merge request.) To do this, follow the instructions [here](https://help.github.com/en/articles/creating-a-pull-request#creating-the-pull-request), taking special note of how to designate the branches you're merging to and from.

### Set up virtualenv and install packages

From the top-level project folder, run

```bash
virtualenv --python=python3.5 venv
```

Then *every time a new python package is required to run the code*, install like this, again from the top-level project folder:

```bash
source venv/bin/activate
pip install [package]
```

You'll know you did it right if your command line is preceded by `(venv)`.

If the package you install is not already in `requirements.txt`, run

```bash
pip freeze > requirements.txt
```

This replaces this list of requirements with your current set of installed packages, so your virtual environment isn't version controlled, but the requirements are.

**Note that whenever you run code in this repository, you should run** `source venv/bin/activate` **first!**

### Set up linting, testing, and formatting

Inside your virtual environment, run

```bash
pip install prospector[with_everything]
pip install coverage
pip install pytest
pip install yapf
```

In particular make sure `prospector --version` is at least 1.1.7 — otherwise, prospector's kind of finicky and might not run properly. If not, run `pip install --upgrade prospector==1.1.7` to install the new version.

Then, anytime you're about to commit, run

```bash
make format
make lint
make test

```

You should also run

```bash
make coverage
```

periodically to ensure that your tests are testing everything. (You may need to modify the `Makefile` to do this.)

## General coding practices and associated resources

While the formatting and linting tools above will maintain general consistency, please make use of the following tips and resources to keep our code as beautiful and release-ready as possible. :)

- Pattern match all boilerplate with existing packages and modules.
- When running `make lint`, address all noted points of ill formatting (missing docstrings, etc.).
- Format your docstrings according to the Google [styleguide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) (and follow the styleguide in general). Another good example resource on this point is [here](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).
- *Write tests for your code as you go.*
