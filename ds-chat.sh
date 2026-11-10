#!/bin/bash

if [ ! -d ~/llm/scripts/venv ]; then
	python -m venv venv
	echo "venv created successfully!"
else
	echo "venv already existed! Continuing..."
fi

source ~/llm/scripts/venv/bin/activate

if python -m pip freeze | grep -qi "rich"; then
	echo "rich is already installed! Continuing..."
else
	if python -m pip install rich; then
		echo "rich installed successfully!"
	else
		echo "rich could not be installed!"
		exit 1
	fi
fi

python ~/llm/scripts/deepseek_chat.py
deactivate
