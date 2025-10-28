VENV			= .venv
VENV_BIN		= $(VENV)/bin
VENV_PYTHON		= $(VENV_BIN)/python3
SYSTEM_PYTHON	= $(or $(shell which python3), $(shell which python))
DOCKER = sudo docker


help:  ## Display this help output
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

$(VENV_PYTHON):
	$(SYSTEM_PYTHON) -m venv --system-site-packages $(VENV)
	$(VENV_PYTHON) -m pip install --upgrade pip
# 	$(VENV_PYTHON) -m pip install poetry


local_packages: $(VENV_PYTHON) ## Install required packages
	$(VENV_PYTHON) -m pip install --no-cache-dir --verbose -r requirements.txt
	$(VENV_PYTHON) -m pip install --no-cache-dir --verbose -r requirements_torch.txt

OPTS = --show
run: ## Run the script
	$(VENV_PYTHON) -m src.zone_process --image example.jpg --points points_garage.json $(OPTS)
	$(VENV_PYTHON) -m src.zone_process --image example.jpg --points points_test.json $(OPTS)
# 	$(VENV_PYTHON) -m src.zone_process --image example.jpg --garage-area 370,550,2100,1400 $(OPTS)
# 	$(VENV_PYTHON) -m src.zone_process --image example.jpg --garage-area 370,550,2100,2000 $(OPTS)

setup:
	$(VENV_PYTHON) -m src.polygon_picker


# poetry: $(VENV_PYTHON)  ## Install packages in local venv using poetry
# 	export PYTHON_KEYRING_BACKEND=keyring.backends.fail.Keyring
# 	$(VENV_PYTHON) -m poetry install --no-interaction --with opencv,backend


# poetry-lock: ## Update poetry lockfile
# 	export PYTHON_KEYRING_BACKEND=keyring.backends.fail.Keyring
# 	$(VENV_PYTHON) -m poetry lock

# POETRY_GROUPS=opencv,backend,dev
# poetry-update: poetry-lock ## poetry packages update helper
# 	# $(VENV_PYTHON) -m poetry show --outdated --with $(POETRY_GROUPS)
# 	$(VENV_PYTHON) -m poetry show --outdated --with $(POETRY_GROUPS)
# 	$(VENV_PYTHON) -m poetry show --outdated --with $(POETRY_GROUPS) \
# 	| awk -v ORS=" " '{print $$1"@latest"}' \
# 	| xargs poetry add

