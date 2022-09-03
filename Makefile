.PHONY: lint format unit

lint:
	flake8 persuasion4good

format:
	black persuasion4good

unit:
	pytest --cov-report term --cov=persuasion4good tests/unit