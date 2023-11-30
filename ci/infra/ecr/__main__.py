"""An AWS Python Pulumi program"""

import pulumi
import pulumi_aws as aws


# Create an AWS resource
repository = aws.ecr.Repository("repository")
# From https://github.com/dagger/examples/blob/main/nodejs/pulumi/infra/ecr/index.ts#L7C7-L7C91
authorization_token = aws.ecr.get_authorization_token(repository.registry_id)
# Export from resources
pulumi.export('repository_url', repository.repository_url)
pulumi.export('authorization_token', authorization_token)
