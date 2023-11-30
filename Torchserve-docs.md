# Install torchserve

```
pip install nvgpu
# Minimum required pytorch 1.5
pip install torchserve torch-model-archiver torch-workflow-archiver

# Install captum https://github.com/pytorch/serve/issues/966
pip install captum
```

Java OpenJDK11 minimum required
```
# Check version
java -version

# Install if needed
sudo apt-get install openjdk-11-jdk

# Change Java version if needed https://www.baeldung.com/linux/java-choose-default-version
sudo update-alternatives --config java
``` 


# Local Environment

## Generate Model Archiver

All dependencies (extra files) must be added manualy to the Model Archiver using the `--extra-files` argument. They will be placed at the top folder, making it necessary to change the routes of the files in the imports.

```
torch-model-archiver --model-name vToonify --version 1.0 \
--serialized-file ./checkpoint/arcane/vtoonify_s_d.pt \
--model-file ./model/vtoonify.py \
--handler main \
--extra-files util.py,./model/vtoonify.py,./model/dualstylegan.py,./model/stylegan/stylegan_model.py,./model/stylegan/op/__init__.py,./model/stylegan/op/upfirdn2d_pkg.py,./model/stylegan/op/fused_act.py,./model/encoder/align_all_parallel.py,./model/bisenet/bisnet_model.py,./model/bisenet/resnet.py,./model/stylegan/op/upfirdn2d_kernel.cu,./model/stylegan/op/fused_bias_act.cpp,./model/stylegan/op/fused_bias_act_kernel.cu,./model/stylegan/op/upfirdn2d.cpp,./model/stylegan/op/conv2d_gradfix.py,./model/encoder/encoders/psp_encoders.py,./model/encoder/encoders/helpers.py,./checkpoint/arcane/vtoonify_s_d.pt,./checkpoint/faceparsing.pth,./checkpoint/encoder.pt,./checkpoint/arcane/exstyle_code.npy,./checkpoint/shape_predictor_68_face_landmarks.dat,./remove_background.py
```

Create `model_store` folder if missing
```
mkdir model_store
```

Move `.mar` file to `model_stsore`
```
mv vToonify.mar model_store/
```

## Run TorchServe

### Stop anydesk
```
java.io.IOException: Failed to bind to address 0.0.0.0/0.0.0.0:7070
```

Same error in: https://discuss.pytorch.org/t/torchserve-stopped-failed-to-bind-address-already-in-use/115232

```
sudo lsof -i:7070
systemctl status anydesk.service
systemctl stop anydesk.service
```

### Start TorchServe
```
torchserve --start --model-store model_store --models vToonify=vToonify.mar --ts-config config.properties
```


## Test model running
```
curl http://localhost:8081/models
```

## Run Inference
The `test.py` file has been created to encode an image, send it to the model and save the results in the `./test` folder.

```
python test.py
```

## Stop TorchServe
```
torchserve --stop
```

# With Docker

## Create TorchServe Docker Image

This is done to create a Docker Image that has torhcserve, torch-model-archiver and the required dependencies, that is `numpy`, `opencv-ptyhon`, `Pillow`, `scipy`, `dlib`,  `Ninja` and the nvidia `nvcc` compiler. Otherwise the `.mar` creation will work but the `torchserve --start` command will fail.

```
docker build -t elface-torchserve-image . -f docker/Dockerfile
```

This image will be used to both generate the `.mar` file and run the model.

*NOTE*: This image takes more than 30 min to be built.

## Run TorchServe Image

We will now run the docker image with the main code folder as a volume so the important files can be added to the `.mar` file.

```
docker run --rm -it -p 8080:8080 -p 8081:8081 \
            -v $(pwd):/home/model-server/crypsis-delizziosa-model \
            --name elface-torchserve elface-torchserve-image:latest
```
For GPU
```
docker run --rm -it --gpus all \
            -p 127.0.0.1:8080:8080 -p 127.0.0.1:8081:8081 -p 127.0.0.1:8082:8082 -p 127.0.0.1:7070:7070 -p 127.0.0.1:7071:7071 \
            --name elface-torchserve elface-torchserve-image:latest
```

## Create .mar file
Enter into the Docker Container

```
docker exec -it --user root elface-torchserve /bin/bash
```

Execute the `torch-model-archiver`. The same command as [Generate Model Archiver section](#generate-model-archiver) the docker but:
- File paths being `./crypsis-delizziosa-model` instead of `./` and adding
- Add `--export-path /home/model-server/crypsis-delizziosa-model/model_store` to store the `.mar` in the local `model_store` folder
- Add requirements file to install dependencies.
  - While running on local they are present in the environment but they need to be added and the installed when starting torchserve

```
torch-model-archiver --model-name vToonify --version 1.0 \
--serialized-file ./crypsis-delizziosa-model/checkpoint/arcane/vtoonify_s_d.pt \
--model-file ./crypsis-delizziosa-model/model/vtoonify.py \
--handler ./crypsis-delizziosa-model/main \
--export-path /home/model-server/crypsis-delizziosa-model/model_store \
--extra-files ./crypsis-delizziosa-model/util.py,./crypsis-delizziosa-model/model/vtoonify.py,./crypsis-delizziosa-model/model/dualstylegan.py,./crypsis-delizziosa-model/model/stylegan/stylegan_model.py,./crypsis-delizziosa-model/model/stylegan/op/__init__.py,./crypsis-delizziosa-model/model/stylegan/op/upfirdn2d_pkg.py,./crypsis-delizziosa-model/model/stylegan/op/fused_act.py,./crypsis-delizziosa-model/model/encoder/align_all_parallel.py,./crypsis-delizziosa-model/model/bisenet/bisnet_model.py,./crypsis-delizziosa-model/model/bisenet/resnet.py,./crypsis-delizziosa-model/model/stylegan/op/upfirdn2d_kernel.cu,./crypsis-delizziosa-model/model/stylegan/op/fused_bias_act.cpp,./crypsis-delizziosa-model/model/stylegan/op/fused_bias_act_kernel.cu,./crypsis-delizziosa-model/model/stylegan/op/upfirdn2d.cpp,./crypsis-delizziosa-model/model/stylegan/op/conv2d_gradfix.py,./crypsis-delizziosa-model/model/encoder/encoders/psp_encoders.py,./crypsis-delizziosa-model/model/encoder/encoders/helpers.py,./crypsis-delizziosa-model/checkpoint/arcane/vtoonify_s_d.pt,./crypsis-delizziosa-model/checkpoint/faceparsing.pth,./crypsis-delizziosa-model/checkpoint/encoder.pt,./crypsis-delizziosa-model/checkpoint/arcane/exstyle_code.npy,./crypsis-delizziosa-model/checkpoint/shape_predictor_68_face_landmarks.dat,./crypsis-delizziosa-model/style_pictures/Peter_Mohrbacher-0048.jpeg,./crypsis-delizziosa-model/style_pictures/Peter_Mohrbacher-0054.jpeg
```

To add requirements file use `--requirements-file crypsis-delizziosa-model/requirements.txt`. Should be needed since we are craeting image with dependencies

## Run TorchServe with model

In order to run the model we will run the same docker image but with some differences:
- Use only `model_store` folder with the `.mar` as volume to keep the main code outside of the container
- Add `config.properties` via volume
- Use `--shm-size` to give 14gb of server memory to ensure it has all the necessary resources reserved. From docs "*It enables memory-intensive containers to run faster by giving more access to allocated memory.*"
- Use `--ulimit memlock=-1` to set the maximum locked-in-memory address space to no limit. This is useful to allow TorchServe to use as much memory as it needs
- `--ulimit stack` may also be usefull but don't know yet. In the example is set to `67108864`

```
docker run --rm -it --runtime=nvidia --gpus all --shm-size=14g -p 8080:8080 -p 8081:8081 -p 8082:8082 -p 7070:7070 -p 7071:7071 --ulimit memlock=-1 --name elface-torchserve elface-torchserve-image:latest
```

Alternatively, running it using --user root could be necessary to let the creation of logs by TorchServe:

```
docker run --rm -it --runtime=nvidia --gpus all --shm-size=14g --user root -p 8080:8080 -p 8081:8081 -p 8082:8082 -p 7070:7070 -p 7071:7071 --ulimit memlock=-1 --name elface-torchserve elface-torchserve-image:latest
```

### Model configuration via config.properties

- max_response_size is set to `104862526` to allow for large videos to be sent as response, otherwise response to large errors may appear
- inference_address `http://0.0.0.0:8080` 
  - **THIS HAS TO BE REVIEWED SINCE IT'S LISTENING TO ALL ADRESSES TO BE AVAILABLE OUTSIDE THE CONTAINER** as when running the `python test.py` script. This can be insecure we have to see if it's needed or how to handle this in convination with Nginx
- management_address `http://0.0.0.0:8081`
  - **THIS HAS TO BE REVIEWED SINCE IT'S LISTENING TO ALL ADRESSES TO BE AVAILABLE OUTSIDE THE CONTAINER** as when running the `python test.py` script. This can be insecure we have to see if it's needed or how to handle this in convination with Nginx

### Docker-Compose

The `docker-compose.yml` replicates the run command from above and can be used to run the model by simply running:

```
docker-compose up --build
```

This allows for easy integration of other services such as Nginx

# Refs
[How to Serve PyTorch Models with TorchServe Youtube Video](https://www.youtube.com/watch?v=XlO7iQMV3Ik)
[pytroch/serve/examples/image_classifier/resnet_18](https://github.com/pytorch/serve/tree/master/examples/image_classifier/resnet_18)

# upFirdn2d error:
```
TypeError: upfirdn2d(): incompatible function arguments. The following argument types are supported: 1. (arg0: at::Tensor, arg1: at::Tensor, arg2: int, arg3: int, arg4: int, arg5: int, arg6: int, arg7: int, arg8: int, arg9: int) -> at::Tensor
```

This function is used for up-sampling and down-sampling images in the StyleGAN architecture.

[Possible solution](https://github.com/rosinality/stylegan2-pytorch/issues/304) renaming the upfirdn2d.py file to upfirdn2dpkg.py and using PyTorch 1.9.0. However, the user did not provide a clear explanation of why this solution worked. **FUCKING WORKS**

[Another user suggested](https://github.com/sapphire497/style-transformer/issues/12) that the issue might be related to the torch.cpp_extension in the stylegan.op path github.com. This could indicate a problem with the compilation of the custom C++/CUDA code used in StyleGAN.

To troubleshoot this issue, you should confirm that:

* The correct number and type of arguments are being passed to the upfirdn2d() function.
* You are using a compatible version of PyTorch. You might want to try PyTorch 1.9.0, as suggested by the user.
* The custom C++/CUDA code in StyleGAN has been compiled correctly. If you are using a precompiled version of StyleGAN, you might want to try compiling it yourself to ensure that it is compatible with your specific system configuration.

## Change UpFirDn2d.apply inputs
Change inputs from 
```
out = UpFirDn2d.apply(
    input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1]
)
```
to
```
out = UpFirDn2d.apply(
    input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1]
)
```
did not work.

## Check compiled
`Your torch version is 1.6.0 which does not support torch.compile` Message appears, upgrade to compatible version

torch.compile is [only available in pytorch>=2.0](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html). New environment wiht pytorch2.1 created

Same error happens:
```
TypeError: upfirdn2d(): incompatible function arguments. The following argument 1. (arg0: torch.Tensor, arg1: torch.Tensor, arg2: int, arg3: int, arg4: int, arg5: int, arg6: int, arg7: int, arg8: int, arg9: int) -> torch.Tensor
```

# Big File Error
[Reference](https://github.com/pytorch/serve/blob/master/docs/Troubleshooting.md)
