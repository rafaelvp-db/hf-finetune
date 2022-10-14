import os, sys

# For relative imports
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# For global vars
os.environ["MAX_LENGTH"] = "100"
os.environ["SEP_TOKEN"] = "50236"
os.environ["MAX_TURNS"] = "5"
os.environ["MODEL_ENDPOINT_URL"] = "https://adb-984752964297111.11.azuredatabricks.net/model-endpoint/persuasion4good/2/invocations"