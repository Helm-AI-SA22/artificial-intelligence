.PHONY : build shell exec

all: build create-environment install

build:
	docker build -f Dockerfile.AI.dev -t ai-container .
	docker run -itd --name helmai -v ${CURDIR}/:/home/aiuser/workdir -p 5000:5000 ai-container bash -c "/bin/sleep infinity"

destroy:
	docker stop helmai
	docker rm -v helmai

create-environment:
	docker exec helmai bash -c "python -m venv .env"
	docker exec helmai bash -c "source .env/bin/activate; pip install --upgrade pip"

install:
	docker exec helmai bash -c "source .env/bin/activate; pip install -r requirements.txt"

shell:
	docker exec -it helmai bash

exec:
	docker exec -it helmai bash -c "source .env/bin/activate; python app.py"
