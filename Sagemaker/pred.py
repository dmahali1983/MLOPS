import boto3
import json

def invoke_sagemaker():

    AWS_ACCESS_KEY_ID ="AKIAYA6YKLF27N2AWXFN"
    AWS_SECRET_ACCESS_KEY ="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"
    AWS_REGION = "us-east-1"
    # Initialize the SageMaker runtime client
    sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=AWS_REGION)

    # Define the endpoint name and the payload to send
    endpoint_name = 'predict'
    #'https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints'
    

    # Invoke the endpoint
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name, Body ="{}" ,     
        ContentType='application/json'
    )
    print(response)

    # Read the response
    result = response['Body'].read().decode("utf-8")
    print(result)

if __name__ == "__main__":
    invoke_sagemaker()
