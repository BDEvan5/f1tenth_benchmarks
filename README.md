# f1tenth_sim

## Getting Started

- Build the Docker image using the Dockerfile
```
sudo docker build -t f1tenth_sim -f Dockerfile .
```
- Start the docker image with
```
sudo docker compose up
```
Doing this mounts the current folder as a volume.
- Enter the docker container using,
```
sudo docker exec -it f1tenth_sim-sim-1 /bin/bash
```
- You can now run commands in the interactive shell.



