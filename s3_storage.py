import boto3
import json
import os
import logging
from botocore.exceptions import ClientError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# S3 Bucket Configuration
S3_BUCKET_NAME = "data-forecast-chatbot"

# Define folder structure in S3
USER_DATA_FOLDER = "user_data/"
CHAT_HISTORY_FOLDER = "chat_history/"
FORECASTS_FOLDER = "forecasts/"
MODELS_FOLDER = "models/"

# Initialize S3 client
def get_s3_client():
    try:
        s3_client = boto3.client('s3')
        # Verify bucket exists
        s3_client.head_bucket(Bucket=S3_BUCKET_NAME)
        return s3_client
    except ClientError as e:
        logger.error(f"Error connecting to S3 bucket: {e}")
        raise Exception(f"S3 connection error: {e}")

# Create folders in S3 if they don't exist
def initialize_s3_folders():
    """Create necessary folders in S3 bucket"""
    s3_client = get_s3_client()
    folders = [USER_DATA_FOLDER, CHAT_HISTORY_FOLDER, FORECASTS_FOLDER, MODELS_FOLDER]
    
    try:
        for folder in folders:
            s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=folder)
        logger.info(f"S3 folders initialized in bucket {S3_BUCKET_NAME}")
    except ClientError as e:
        logger.error(f"Error creating S3 folders: {e}")
        raise

# User data functions
def save_user_data(users_dict):
    """Save user data to S3"""
    s3_client = get_s3_client()
    users_file = USER_DATA_FOLDER + "users.json"
    
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=users_file,
            Body=json.dumps(users_dict),
            ContentType='application/json'
        )
        logger.info("User data saved to S3 successfully")
        return True
    except ClientError as e:
        logger.error(f"Error saving user data to S3: {e}")
        return False

def load_user_data():
    """Load user data from S3"""
    s3_client = get_s3_client()
    users_file = USER_DATA_FOLDER + "users.json"
    
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=users_file)
        users_dict = json.loads(response['Body'].read().decode('utf-8'))
        logger.info("User data loaded from S3 successfully")
        return users_dict
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            logger.warning("Users file not found, creating new one")
            empty_users = {}
            save_user_data(empty_users)
            return empty_users
        else:
            logger.error(f"Error loading user data from S3: {e}")
            return {}

# Chat history functions
def save_chat_history(username, chat_history):
    """Save user chat history to S3"""
    s3_client = get_s3_client()
    chat_file = CHAT_HISTORY_FOLDER + f"{username}_chat_history.json"
    
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=chat_file,
            Body=json.dumps(chat_history),
            ContentType='application/json'
        )
        logger.info(f"Chat history for {username} saved to S3 successfully")
        return True
    except ClientError as e:
        logger.error(f"Error saving chat history to S3: {e}")
        return False

def load_chat_history(username):
    """Load user chat history from S3"""
    s3_client = get_s3_client()
    chat_file = CHAT_HISTORY_FOLDER + f"{username}_chat_history.json"
    
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=chat_file)
        chat_history = json.loads(response['Body'].read().decode('utf-8'))
        logger.info(f"Chat history for {username} loaded from S3 successfully")
        return chat_history
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            logger.info(f"No chat history found for {username}")
            return []
        else:
            logger.error(f"Error loading chat history from S3: {e}")
            return []

def delete_chat_history(username):
    """Delete user chat history from S3"""
    s3_client = get_s3_client()
    chat_file = CHAT_HISTORY_FOLDER + f"{username}_chat_history.json"
    
    try:
        s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=chat_file)
        logger.info(f"Chat history for {username} deleted from S3 successfully")
        return True
    except ClientError as e:
        logger.error(f"Error deleting chat history from S3: {e}")
        return False

# Forecast and model storage functions
def save_forecast(username, forecast_name, forecast_data):
    """Save forecast data to S3"""
    s3_client = get_s3_client()
    forecast_file = FORECASTS_FOLDER + f"{username}/{forecast_name}.json"
    
    try:
        # Create user folder if it doesn't exist
        user_folder = FORECASTS_FOLDER + f"{username}/"
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=user_folder)
        
        # Save forecast
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=forecast_file,
            Body=json.dumps(forecast_data),
            ContentType='application/json'
        )
        logger.info(f"Forecast {forecast_name} saved to S3 successfully")
        return True
    except ClientError as e:
        logger.error(f"Error saving forecast to S3: {e}")
        return False

def load_forecast(username, forecast_name):
    """Load forecast data from S3"""
    s3_client = get_s3_client()
    forecast_file = FORECASTS_FOLDER + f"{username}/{forecast_name}.json"
    
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=forecast_file)
        forecast_data = json.loads(response['Body'].read().decode('utf-8'))
        logger.info(f"Forecast {forecast_name} loaded from S3 successfully")
        return forecast_data
    except ClientError as e:
        logger.error(f"Error loading forecast from S3: {e}")
        return None

def save_model(username, model_name, model_binary):
    """Save model binary to S3"""
    s3_client = get_s3_client()
    model_file = MODELS_FOLDER + f"{username}/{model_name}.pkl"
    
    try:
        # Create user folder if it doesn't exist
        user_folder = MODELS_FOLDER + f"{username}/"
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=user_folder)
        
        # Save model
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=model_file,
            Body=model_binary
        )
        logger.info(f"Model {model_name} saved to S3 successfully")
        return True
    except ClientError as e:
        logger.error(f"Error saving model to S3: {e}")
        return False

# Initialize S3 folders
try:
    initialize_s3_folders()
except Exception as e:
    logger.warning(f"S3 initialization warning: {e}")
    logger.warning("App will continue but may use local storage if S3 is unavailable") 
