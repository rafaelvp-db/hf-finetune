# import config.
# You can change the default config with `make cnf="config_special.env" build`
cnf ?= .env
include $(cnf)
export $(shell sed 's/=.*//' $(cnf))

# HELP
# This will output the help for each task
# thanks to https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
.PHONY: help frontend backend

help: ## This help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.DEFAULT_GOAL := help

backend:
	docker build --tag persuasion4good/backend:1.0 ./backend

frontend:
	docker build --tag persuasion4good/frontend:1.0 ./frontend

build:
	docker build --tag persuasion4good/backend:1.0 ./backend && \
	docker build --tag persuasion4good/frontend:1.0 ./frontend

run-frontend:
	docker run -it --name frontend --net tmpnetwork --rm -p 8080:80 ${APP_NAME}/frontend:1.0

run-backend:
	docker run -it --name backend --net tmpnetwork --rm -p 8000:8080 --env-file=.env ${APP_NAME}/backend:1.0

network:
	docker network prune -f && \
	docker network create --driver bridge tmpnetwork
