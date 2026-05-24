# Docker Installation for MME-VLA Policy Learning

This guide explains how to set up Docker and NVIDIA GPU support so you can build and run the MME-VLA image.

## 1) Install Docker Engine
> Skip this if you have already installed Docker.

Follow Docker’s official instructions for Ubuntu:
- Docker Engine install guide: `https://docs.docker.com/engine/install/ubuntu/`

After installing, make sure the service is running:

```bash
docker run --rm hello-world
```

## 2) Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.14.1/install-guide.html) (GPU support)

> Skip this if you have already installed `nvidia-ctk`.

Install the toolkit (Ubuntu):

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor --batch --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

Configure Docker to use the NVIDIA runtime and restart Docker:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify GPU access inside a container:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

## 3) Build the MME-VLA Docker image

From the repository root:

```bash
docker build -t <image_name>:<tag> .
# e.g., run `docker build -t mme_vla:cuda12.8 .`
```

Start the container:
```bash
export PORT=8001
docker run --rm -it --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video \
  -v "$PWD/runs:/app/runs" -v "$PWD/data:/app/data" \
  -p $PORT:$PORT \
  mme_vla:cuda12.8
```
`-e` sets an environment variable inside the container (e.g., `NVIDIA_DRIVER_CAPABILITIES`).  
`-v` mounts a host path into the container as a **bind mount**. Here we mount the host `./runs` and `./data` directories to `/app/runs` and `/app/data` inside the container.

Because these are bind mounts, files you create/modify inside the container will be visible to you on the host (and vice versa) as they are the same underlying directories.
`-p` publishes a container port to a host port (port mapping).  

> **Permissions note:** this image runs as `root` by default, so any *new* files created in the mounted `runs/` or `data/` dirs will become `root`-owned on the host.
>
> To fix ownership on the host (run outside the container):
> `sudo chown -R "$USER:$USER" runs data`
>
> Alternatively, run the container as your UID/GID to keep created files owned by you:
> ```bash
> docker run --rm -it --gpus all \
>   --user "$(id -u):$(id -g)" \
>   -e NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video \
>   -v "$PWD/runs:/app/runs" -v "$PWD/data:/app/data" \
>   -p $PORT:$PORT \
>   mme_vla:cuda12.8
> ```
> If you use `--user`, `apt-get update` inside the container may fail because non-root users typically can’t write to `/var/lib/apt`.



## 4) Evaluate the policy
```
# terminal 0
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --seed=7  --port=$PORT policy:checkpoint --policy.dir=runs/ckpts/mme_vla_suite/perceptual-framesamp-modul/79999 --policy.config=mme_vla_suite

# terminal 1 
eval "$(micromamba shell hook --shell bash)"
micromamba activate robomme
CUDA_VISIBLE_DEVICES=1 python examples/robomme/eval.py --args.model_seed=7 --args.port=$PORT --args.policy_name=perceptual-framesamp-modul --args.model_ckpt_id=79999
```


## 5) Other Hints

To stop the container:
```bash
docker ps
docker stop <container_id_or_name>
```
Alternatively, inside the container shell you can stop the session with `exit` (or `Ctrl-D`).

To rebuild the Docker image:
```bash
docker build --no-cache -t <image_name>:<tag> .
```

To detach from the running container (without stopping it), press `Ctrl-p` then `Ctrl-q`.

To re-attach the session:
```bash 
docker ps
docker exec -it <container_id_or_name> bash
```

To start a detached container directly, use `-d`:
```
docker run -d --rm -it --gpus all ...
```

To install additional packages inside the container, run:
```
apt-get update
apt-get install <package>
```