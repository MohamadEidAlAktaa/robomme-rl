# RoboMME Challenge Guide: Docker Submission

This document explains how to package your policy into a Docker image that organizers can pull and run for the CVPR challenge evaluation.

We use an MME-VLA (framesamp+modul) model as an example.

## What you (the participant) provide

- **A Docker image** containing your policy server code and all dependencies.
- **A checkpoint location** that the organizers can download.
- **One command** to start your policy server inside the container.  

### 1) Implement the policy interface and serving script

Implement the `Policy` class compatible with the challenge [interface](https://github.com/RoboMME/robomme_benchmark/blob/a2ad6f6a09bf117167cef3237f6ac6ecba418307/challenge_interface/policy.py#L12).

- Copy the [challenge_interface](https://github.com/RoboMME/robomme_benchmark/tree/main/challenge_interface) directory from the benchmark repo into your repo.

  For example, in this repo, we copied the participant-oriented files into the `challenge_interface` [directory](..).

  ```
  challenge_interface
  ├── __init__.py
  ├── msgpack_numpy.py
  ├── policy.py
  ├── scripts
  │   └── deploy.py
  └── server.py
  ```

- Override **`infer`** and **`reset`** in your policy implementation.
  
  For example, we wrapped the original MME-VLA policy in the [`MyPolicy_for_CVPR_Challenge`](https://github.com/RoboMME/robomme_policy_learning/blob/main/challenge_interface/policy.py#L29) class for the challenge.

- Adjust `challenge_interface/scripts/deploy.py` for your own policy.

  For example, in this repo, we modified it into [this](https://github.com/RoboMME/robomme_policy_learning/blob/main/challenge_interface/scripts/deploy.py#L53) for the `MyPolicy_for_CVPR_Challenge` class.


### 2) Upload your checkpoint(s)

Upload your model checkpoint(s) somewhere the organizers can download them.

- For example, we uploaded the framesamp+modul MME-VLA model to [Hugging Face](https://huggingface.co/Yinpei/perceptual-framesamp-modul).

### 3) Build the Docker image

We provide a `challenge_interface/docs/Dockerfile` example.

You may edit it to include any additional dependencies your policy requires, or use your own Dockerfile.

Build the Docker image:

```bash
docker build -f challenge_interface/docs/Dockerfile -t <my_cool_model_name>:latest .
```


### 4) Self-check locally with the benchmark eval client

1) Run your container locally:

```bash
docker run --rm -it --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video \
  -v "$PWD/<dir>:/app/<dir>" \
  -p <port>:<port> \
  my_cool_model_name:latest
```
Map the server port and mount the directory correctly. Here we put all the model checkpoints under the `runs` directory and use port 8001.

2) Inside the container, start the policy server using your modified `deploy.py`.
```
# Inside the container
uv run python -m challenge_interface.scripts.deploy --port <port> --checkpoint-dir <dir>
```

3) From another terminal, run the [benchmark eval client](https://github.com/RoboMME/robomme_benchmark/blob/main/challenge_interface/scripts/phase1_eval.py) outside the policy server container for evaluation.

```
cd <robomme_benchmark>
uv run python -m challenge_interface.scripts.phase1_eval --port <port>
```

### 5) Push the Docker image to a registry

Push your image to a registry so the organizers can pull it from Docker Hub.

```bash
docker tag <my_cool_model_name>:latest <dockerhub_user>/<my_cool_model_name>:latest
docker login
docker push <dockerhub_user>/<my_cool_model_name>:latest
```

For example, organizers pushed an image for [framesamp+modul](https://hub.docker.com/repository/docker/yinpeidai/perceptual-framesamp-modul/general) to Docker Hub.


### 6) Submit your policy

Prepare the following information:

- **policy_name**
- **email**
- **action_space**: you can only choose one of "joint_angle", "ee_pose", or "waypoint".
- **evaluation_method**: set as `docker`.
- **Checkpoint URL** (downloadable by organizers), e.g. `https://huggingface.co/Yinpei/perceptual-framesamp-modul`
- **Docker image** (registry path + tag), e.g. `<dockerhub_user>/my_cool_model_name:latest`
- **Command to start the policy server**, e.g. `uv run python -m challenge_interface.scripts.deploy --checkpoint-dir runs/ckpts/perceptual-framesamp-modul/79999`. The organizers will run `deploy.py` to start your policy server, then run evaluation.
- **Other flags**: `use_depth`, `use_camera_params` (default: `false`)

An example JSON file can be found [here](https://github.com/RoboMME/robomme_policy_learning/blob/main/challenge_interface/docs/submission_example_docker.json).


---

## What the organizers will do

After we receive your submitted JSON file, we will:

1) **Pull your docker image** (based on the image name/tag you provided), for example:

```bash
docker pull yinpeidai/perceptual-framesamp-modul:latest
```

2) **Download your checkpoint(s)** (based on the URL you provided), for example:

```bash
git clone https://huggingface.co/YinpeiDai/perceptual-framesamp-modul runs/ckpts/perceptual-framesamp-modul
```

3) **Run your container**, for example:

```bash
docker run --rm -it --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video \
  -v "$PWD/runs:/app/runs" \
  -p 8001:8001 \
  yinpeidai/perceptual-framesamp-modul:latest
```

Then, inside the container, start the policy server based on the command you provided, for example:

```bash
uv run python -m challenge_interface.scripts.deploy --port 8001 --checkpoint-dir runs/ckpts/perceptual-framesamp-modul/79999
```

4) **Run evaluation** (phase 1), using the eval script from the RoboMME benchmark repo:

```bash
cd robomme_benchmark
uv run python -m challenge_interface.scripts.phase1_eval --port 8001 --action_space joint_angle --team_id 0000
```

After determining the top 5–10 teams, the organizers will run phase 2 evaluation.

