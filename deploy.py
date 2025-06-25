#For Endpoint
#!/usr/bin/env python
"""
Deploy (or re-deploy) the DeepFake detector to SageMaker.

• Deletes any pre-existing endpoint / config with the same name
• Creates a fresh endpoint on an ml.m5.large instance
"""

import boto3
import botocore
import time
from sagemaker.tensorflow import TensorFlowModel
from sagemaker import Session, get_execution_role

# ------------------------------------------------------------------
# 0.  CONFIG – change if needed
# ------------------------------------------------------------------
ENDPOINT_NAME = "deepfake-detector-endpoint"
MODEL_S3_URI   = "s3://deepfake222/models/model.tar.gz"
INSTANCE_TYPE  = "ml.m5.large"
TF_VERSION     = "2.12.0"

# ------------------------------------------------------------------
# 1.  Set up SageMaker session + IAM role
# ------------------------------------------------------------------
sm_session = Session()
try:
    role = get_execution_role()  # works in SageMaker Notebook or Studio
except Exception:
    raise RuntimeError(
        "❌ Could not auto-detect execution role. Set `role` manually if running outside SageMaker."
    )

sm_client = boto3.client("sagemaker")

# ------------------------------------------------------------------
# 2.  Clean up old endpoint + config
# ------------------------------------------------------------------
def safe_delete_endpoint(name: str):
    try:
        sm_client.delete_endpoint(EndpointName=name)
        print(f"🗑️  Deleted old endpoint: {name}")
        waiter = sm_client.get_waiter("endpoint_deleted")
        waiter.wait(EndpointName=name)
    except botocore.exceptions.ClientError as e:
        if "Could not find endpoint" not in str(e):
            raise

def safe_delete_endpoint_config(name: str):
    try:
        sm_client.delete_endpoint_config(EndpointConfigName=name)
        print(f"🧹 Deleted old endpoint-config: {name}")
    except botocore.exceptions.ClientError as e:
        if "Could not find endpoint configuration" not in str(e):
            raise

safe_delete_endpoint(ENDPOINT_NAME)
safe_delete_endpoint_config(ENDPOINT_NAME)

# ------------------------------------------------------------------
# 3.  Create model object – no entry_point, just pure SavedModel
# ------------------------------------------------------------------
model = TensorFlowModel(
    model_data=MODEL_S3_URI,
    role=role,
    framework_version=TF_VERSION,
    sagemaker_session=sm_session
)

# ------------------------------------------------------------------
# 4.  Deploy the endpoint
# ------------------------------------------------------------------
print("🚀 Spinning up new endpoint… this can take a few minutes ⏳")
predictor = model.deploy(
    initial_instance_count=1,
    instance_type=INSTANCE_TYPE,
    endpoint_name=ENDPOINT_NAME
)

print("\n✅ Deployment complete!")
print("🔗 Endpoint name:", predictor.endpoint_name)
print("   (Use this name in your front-end / API calls)")
