# AI-POC
ML | AI related POC examples

# How to setup
```sh
curl -sSL https://install.python-poetry.org | python3 - --version 1.7.0
poetry py3.11 ai-poc
poetry config virtualenvs.in-project true
poetry env use 3.11
source .venv/bin/activate
poetry config virtualenvs.prompt 'ai-poc-py3.11'
poetry config --list
```

# How to run
```sh
git clone https://github.com/AleksNeStu/AI-POC.git
poetry install --no-root
source .venv/bin/activate
```

# Structure
[frameworks](frameworks) - Frameworks for AI tasks\
[graph](graph) - Graphical solutions for AI\
[libs](libs) - Libraries for AI tasks\
[models](models) - AI models solution for different cases\
[tasks](tasks) - AI tasks solution for different cases