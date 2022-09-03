.PHONY: lint format

lint:
	flake8 persuasion4good

format:
	black persuasion4good