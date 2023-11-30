import random
import sys, os, json, argparse
from datetime import datetime

import anyio
import dagger

async def main(args):
    config = dagger.Config(log_output=sys.stdout)

    async with dagger.Connection(config) as client:
        # Load Pulumi
        if args.pulumi_token is None:
            pulumi_access_token = client.set_secret("PULUMI_ACCESS_TOKEN", os.environ["PULUMI_ACCESS_TOKEN"])
        else:
            pulumi_access_token = client.set_secret("PULUMI_ACCESS_TOKEN", args.pulumi_token)

        if args.aws_id is None:
            aws_access_key_id = client.set_secret("AWS_ACCESS_KEY_ID", os.environ["AWS_ACCESS_KEY_ID"])
        else:
            aws_access_key_id = client.set_secret("AWS_ACCESS_KEY_ID", args.aws_id)

        if args.aws_secret is None:
            aws_secret_access_key = client.set_secret("AWS_SECRET_ACCESS_KEY", os.environ["AWS_SECRET_ACCESS_KEY"])
        else:
            aws_secret_access_key = client.set_secret("AWS_SECRET_ACCESS_KEY", args.aws_secret)

        if args.aws_token is None:
            aws_session_token = client.set_secret("AWS_SESSION_TOKEN", os.environ["AWS_SESSION_TOKEN"])
        else:
            aws_session_token = client.set_secret("AWS_SESSION_TOKEN", args.aws_token)

        pulumi_stack = args.stack

        pulumi_container = (
            client.container()
            .from_("pulumi/pulumi-python")
            .with_exec(["/bin/bash", "-c", "pulumi plugin install resource aws"])
            .with_secret_variable("PULUMI_ACCESS_TOKEN", pulumi_access_token)
            .with_directory(
                "/pulumi/projects",
                client.host().directory("ci/infra"),
                exclude=["ecr/venv", "ecs/venv", "ec2/venv"]
            )
            # Mount AWS credentials
            .with_secret_variable("AWS_ACCESS_KEY_ID", aws_access_key_id)
            .with_secret_variable("AWS_SECRET_ACCESS_KEY", aws_secret_access_key)
            .with_secret_variable("AWS_SESSION_TOKEN", aws_session_token)
        )

        # Create the ECR registry:
        ecr_output = await (
            pulumi_container
            .with_env_variable("CACHEBUSTER", str(datetime.now()))
            .with_exec(["/bin/bash", "-c", "cd ecr &&  pulumi stack select "+pulumi_stack+" -c && pulumi up -y"])
        )

        # Get ECR url
        repo_url = await (
            ecr_output
            .with_exec(["/bin/bash", "-c", "cd ecr && pulumi stack output repository_url -s "+pulumi_stack])
        ).stdout()

        # Get ECR token
        auth_token = await (
            ecr_output
            .with_exec(["/bin/bash", "-c", "cd ecr && pulumi stack output authorization_token"])
        ).stdout()

        # print(f"auth_token: {auth_token}")

        auth_token = json.loads(auth_token)
        password = client.set_secret("password", auth_token["password"])

        # Compiling the model for torchserve

        # torchserve_img_dir = (
        #     client.host()
        #     .directory(".",
        #                exclude=[
        #                    "checkpoint/",
        #                    "model_store/",
        #                    "data/",
        #                    "output/",
        #                    "ci/",
        #                    "output/",
        #                    "output_test/"
        #                ],
        #                # include=[
        #                #     # "scripts/",
        #                #     "model/",
        #                #     "*py"
        #                # ]
        #                )
        # )

        # torchserve_container = (
        #     torchserve_img_dir
        #     .docker_build(dockerfile="docker/Dockerfile", platform=dagger.Platform("linux/amd64"))
        #     .with_exec(["/bin/bash", "-c", "ls -la && ./scripts/download_checkpoints.sh"])
        # )

        # Compile the model to .mar file
        # compiled_model = (
        #     torchserve_container
        #     .with_directory(".", client.host().directory("."),
        #         exclude=["scripts/",
        #                  # "checkpoint/",
        #                  "model_store/",
        #                  "data/",
        #                  "output/",
        #                  "ci/",
        #                  "output/",
        #                  "output_test/"]
        #                     )
        #     .with_exec(["ls","-la","&&","pwd","&&","ls","-la","checkpoint/","&&","mkdir","model_store","&&","torch-model-archiver","--model-name","vToonify","--version","1.0","--serialized-file","checkpoint/arcane/vtoonify_s_d.pt","--model-file","model/vtoonify.py","--handler","main","--export-path","model_store","--extra-files","util.py,model/vtoonify.py,model/dualstylegan.py,model/stylegan/stylegan_model.py,model/stylegan/op/__init__.py,model/stylegan/op/upfirdn2d_pkg.py,model/stylegan/op/fused_act.py,model/encoder/align_all_parallel.py,model/bisenet/bisnet_model.py,model/bisenet/resnet.py,model/stylegan/op/upfirdn2d_kernel.cu,model/stylegan/op/fused_bias_act.cpp,model/stylegan/op/fused_bias_act_kernel.cu,model/stylegan/op/upfirdn2d.cpp,model/stylegan/op/conv2d_gradfix.py,model/encoder/encoders/psp_encoders.py,model/encoder/encoders/helpers.py,checkpoint/arcane/vtoonify_s_d.pt,checkpoint/faceparsing.pth,checkpoint/encoder.pt,checkpoint/arcane/exstyle_code.npy,checkpoint/shape_predictor_68_face_landmarks.dat"], insecure_root_capabilities=True).directory("model_store")
        # )

        torchserve_container = (
            client.host()
            .directory(".",
                       exclude=[
                           # "checkpoint/",
                           # "model_store/",
                           "data/",
                           "output/",
                           "ci/",
                           "output/",
                           "output_test/"
                       ]
                       )
            .docker_build(dockerfile="docker/Dockerfile.build", platform=dagger.Platform("linux/amd64"))
            # .with_exec(["/bin/bash", "-c", "ls -la && ./scripts/download_checkpoints.sh"])
        )

        repo_url = repo_url.replace("\n","").replace("\r","")

        container_publication = await (
            torchserve_container
            # .with_directory("model_store", compiled_model)
            # .with_default_args(["torchserve","--start","--model-store=/home/model-server/crypsis-delizziosa-model/model_store","--models","vToonify=vToonify.mar","--ts-config","/home/model-server/crypsis-delizziosa-model/config.properties"])
            .with_registry_auth(repo_url.split("/")[0], auth_token["user_name"], password)
            .publish(f"{repo_url}:latest")
        )

        # # Destroy ECR
        # ecr_del= await (
        #     ecr_output
        #     .with_env_variable("CACHEBUSTER", str(datetime.now()))
        #     .with_exec(["/bin/bash", "-c", "cd ecr && pulumi stack select dev -c && pulumi up -y"])
        # )

argParser = argparse.ArgumentParser()
argParser.add_argument("--pulumi_token",
                       help="PULUMI_ACCESS_TOKEN")
argParser.add_argument("--aws_id",
                       help="AWS_ACCESS_KEY_ID")
argParser.add_argument("--aws_secret",
                       help="AWS_SECRET_ACCESS_KEY")
argParser.add_argument("--aws_token",
                       help="AWS_SESSION_TOKEN")
argParser.add_argument("--stack",
                       default="dev",
                       help="Pulumi stack")

args = argParser.parse_args()
anyio.run(main, args)
