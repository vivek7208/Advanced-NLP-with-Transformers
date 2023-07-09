import json
import os
from pathlib import Path

from package import utils


current_folder = utils.get_current_folder(globals())
cfn_stack_outputs_filepath = Path(current_folder, "../../stack_outputs.json").resolve()
with open(cfn_stack_outputs_filepath) as f:
    cfn_stack_outputs = json.load(f)

SAGEMAKER_MODE = cfn_stack_outputs["SagemakerMode"]

AWS_ACCOUNT_ID = cfn_stack_outputs["AwsAccountId"]
AWS_REGION = cfn_stack_outputs["AwsRegion"]

STACK_NAME = cfn_stack_outputs["StackName"]
SOLUTION_PREFIX = cfn_stack_outputs["SolutionPrefix"]

S3_BUCKET = cfn_stack_outputs["S3Bucket"]
DATASETS_S3_PREFIX = "datasets"
OUTPUTS_S3_PREFIX = "outputs"

SOURCE_S3_BUCKET = cfn_stack_outputs["SolutionsS3Bucket"]
SOURCE_S3_PREFIX = os.path.join(cfn_stack_outputs["LibraryVersion"], cfn_stack_outputs["SolutionName"])
SOURCE_S3_PATH = f"s3://{SOURCE_S3_BUCKET}/{SOURCE_S3_PREFIX}"

TRAINING_INSTANCE_TYPE = cfn_stack_outputs["TrainingInstanceType"]
HOSTING_INSTANCE_TYPE = cfn_stack_outputs["HostingInstanceType"]

IAM_ROLE = cfn_stack_outputs["IamRole"]

TAG_KEY = "sagemaker-soln"

TEST_OUTPUTS_S3_BUCKET = cfn_stack_outputs.get("TestOutputsS3Bucket", "")
