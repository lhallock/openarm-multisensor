export PYTHONPATH := $(CURDIR)

PYTHON_FILES = $(shell find . -name '*.py' ! -path '*/\.*' ! -path './venv/*')
VERSION=$(shell git describe --tags --dirty 2>/dev/null || echo "0.0.0")

.PHONY: version format lint test python bash
.DEFAULT_GOAL := help

#################################################################################
# General purpose development commands
#################################################################################

## Autoformat the repository
format: 
	isort $(PYTHON_FILES)
	yapf --style=google -pir $(PYTHON_FILES)
	@find . -type f -name '*.txt' -exec sed --in-place 's/[[:space:]]\+$$//' {} \+

## Run linters
lint: 
	prospector -i sandbox 

## Run unit tests
test:
	python -m pytest test

## Check test coverage
coverage:
	coverage run --source multisensorimport -m pytest 
	coverage report

## Copy version info from git tags - NOT IMPLEMENTED
version:
	@sed -i "s/.*__version__.*/__version__ = '$(VERSION)'/" multisensorimport/__init__.py

## Create a release object - NOT IMPLEMENTED
release: version
	python3 setup.py sdist


##############################################################################
# Autodocument commands
##############################################################################
# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
help:
	@echo "$$(tput bold)Available commands:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
