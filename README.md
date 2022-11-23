# artificial-intelligence

The AI repository for the Helm project. 

## How to run the server with Docker
Clone the repository with the command:

`git clone https://github.com/Helm-AI-SA22/artificial-intelligence.git`

and move in the project directory with the command `cd artificial-intelligence`.

Then create the virtual enviroment with the required dependencies and install the docker conatiner with the command `make init`.

Once the installation is complete you can run the server by executing the command `make exec`.

At this point the server will run on your localhost on port 5000, and it will accept requests on the two exposed APIs `fast` and `slow`.

You can also run the container in interactive mode with the command `make shell`. Once you're inside the container run the command:

`source .env/bin/activate`

to activate the virtual environment.