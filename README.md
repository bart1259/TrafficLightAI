# TrafficLightAI
An A.I. to determine ideal timings of traffic lights.

## Running the Backend Server
For ease of modifiability and debugability, the backend server is run from a Jupyter notebook.

## Installing no GIL Python

To install python without the global interpeter lock simply run

```bash
sh install-python-no-gil.sh
```
To run a script using this new version of python run
```bash
# Sets proper enviornment variables
source ~/.bashrc

# Now run 
nogil-python script.py

# Pip install using
nogil-python -m pip install numpy
```

## SSH Tunnel Setup for Remote Server
If the backend server is run on a machine other than your local computer (e.g. ROSIE) an SSH tunnel must be created for the requests to reach the remote session. To do this, run the `tunnel.sh` script on your LOCAL machine IN BASH. You will be prompted for the node the backend server is running on (this can be found by looking at the URL of your Jupyter notebook) and your username. If SSH keys have been exchanged, you will not need to provide a password. From here an SSH tunnel we be created. To verify this, with the server running, go to [localhost:3000](http://localhost:3000) and see if you can get a response from the server.
