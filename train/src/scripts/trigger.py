import argparse
import logging
import os
import sys
from datetime import datetime

import boto3
from sagemaker.pytorch import PyTorch
import sagemaker

# ---------------------------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def trigger_sagemaker_training(job_name: str, config_path: str) -> None:
    """
    Submits a training job to AWS SageMaker using Script Mode.
    """
    logger.info(f"Preparing to trigger SageMaker job: {job_name}")

    try:
        # SECURITY: Rely on boto3's default session which securely inherits 
        # temporary OIDC tokens from GitHub Actions. No hardcoded keys.
        boto_session = boto3.Session()
        sagemaker_session = sagemaker.Session(boto_session=boto_session)
        
        # IAM Role required by SageMaker to access S3 data and write logs
        role = os.environ.get('SAGEMAKER_EXECUTION_ROLE')
        if not role:
            raise ValueError("Environment variable 'SAGEMAKER_EXECUTION_ROLE' is missing.")

        # INFRASTRUCTURE AS CODE: Define the AWS Training Environment
        # We use AWS's pre-built Deep Learning Container (DLC) for PyTorch
        estimator = PyTorch(
            entry_point='run.py',               # The master script to run
            source_dir='./sagemaker_training',  # The folder to zip and send to AWS
            role=role,
            framework_version='2.0.0',          # PyTorch version
            py_version='py310',
            instance_count=1,
            instance_type='ml.g4dn.xlarge',     # T4 GPU instance (Cost-effective)
            sagemaker_session=sagemaker_session,
            hyperparameters={
                # Pass the config file path dynamically to run.py inside the container
                'config': config_path 
            },
            # DSA / Cost Efficiency: Use Managed Spot Training to save up to 70% on AWS bills
            use_spot_instances=True,
            max_run=86400,                      # 24 hour max timeout
            max_wait=86400                      # Spot instance wait time
        )

        logger.info("Submitting training job to AWS...")
        
        # COST OPTIMIZATION: wait=False is critical in CI/CD. 
        # If wait=True, GitHub Actions will stay open for hours charging you 
        # per minute while it watches AWS train. wait=False fires and forgets.
        estimator.fit(job_name=job_name, wait=False)
        
        logger.info(f"🎉 SUCCESS: Training job '{job_name}' submitted.")
        logger.info("Monitor the job via the AWS SageMaker Console.")

    except Exception as e:
        logger.error(f"Failed to trigger SageMaker training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trigger AWS SageMaker Training")
    parser.add_argument("--job-name", type=str, required=True, help="Unique name for the AWS job")
    parser.add_argument("--config", type=str, required=True, help="Path to configs.yaml")
    args = parser.parse_args()
    
    # Clean up the job name to meet AWS regex requirements
    safe_job_name = args.job_name.replace('_', '-').lower()[:63]
    trigger_sagemaker_training(safe_job_name, args.config)