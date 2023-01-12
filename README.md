# TrafficLightAI
An A.I. to determine ideal timings of traffic lights.

## Running the Backend Server
For ease of modifiability and debugability, the backend server is run from a Jupyter notebook. If the server ever needs to be updated, the stop server cell must be run before the start server cell is run again (if not, flask will complain about the port already being used).

## SSH Tunnel Setup for Remote Server
If the backend server is run on a machine other than your local computer (e.g. ROSIE) an SSH tunnel must be created for the requests to reach the remote session. To do this, run the `tunnel.sh` script on your LOCAL machine. You will be prompted for the node the backend server is running on (this can be found by looking at the URL of your Jupyter notebook) and your username. If SSH keys have been exchanged, you will not need to provide a password. From here an SSH tunnel we be created. To verify this, with the server running, go to [localhost:3000](http://localhost:3000) and see if you can get a response from the server.
