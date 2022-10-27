.PHONY : build shell exec

all: build

build:
	docker build -f Dockerfile.AI -t ai-container .

shell:
	docker run -it -v ${PWD}/src/:/home/aiuser/src -p 5000:5000 ai-container bash

exec:
	docker run -it -v ${PWD}/src/:/home/aiuser/src -p 5000:5000 ai-container python app.py