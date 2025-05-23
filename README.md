# Documentation: AI-Powered Multi-File Data Analysis & Forecasting

# Overview

This Streamlit application provides powerful data analysis and forecasting capabilities for multiple file formats, integrating AWS Bedrock for AI-powered chat interactions.

# Key Features

- Multi-File Support: Handles CSV, Excel, JSON, and Parquet files
- Automated Date Detection: Intelligently identifies date columns for time series analysis
- Advanced Forecasting: Uses ARIMA and Random Forest models for predictive analytics
- Interactive Visualizations: Displays forecasts vs. actual data through matplotlib
- AI-Powered Chat: Integrates with AWS Bedrock for dataset queries

# Technical Components

# File Processing

- Supports multiple file uploads with progress tracking
- Automatic encoding detection using Chardet
- Robust error handling for file loading

# Data Analysis

- Automatic detection of numeric columns suitable for forecasting
- Data validation and preprocessing
- Intelligent handling of date-based time series

# Forecasting System

- Primary: ARIMA model for time series forecasting
- Fallback: Random Forest Regressor for complex patterns
 
# Chat System

- Integration with AWS Bedrock for natural language processing
- Persistent chat history management

# Usage Requirements

- AWS Bedrock credentials configured
- Required Python packages: streamlit, pandas, numpy, matplotlib, boto3, statsmodels, scikit-learn
- Sufficient storage for chat history and model persistence

# Error Handling

- Robust file format validation
- Graceful fallback between forecasting models
- Clear error messaging for users

# This application combines modern data science techniques with user-friendly interfaces to provide comprehensive data analysis capabilities.

# AI-Powered Data Analysis and Forecasting

An interactive web application for data analysis and time-series forecasting with chat-based insights.

## Features

- User authentication with usage tracking
- Multi-file upload supporting CSV, Excel, JSON, Parquet, and PDF files
- Time-series forecasting with Prophet and Random Forest models
- AI-powered chatbot for data insights
- S3 integration for persistent storage
- **Premium subscription** with Razorpay payment integration

## AWS S3 Integration

This application uses AWS S3 for persistent storage of:
- User accounts and usage data
- Chat history
- Forecast results
- Trained models

### S3 Bucket Setup

1. **Create an S3 bucket** named "cheque-upload-akshit" (or modify the bucket name in `s3_storage.py`)
2. **Set appropriate permissions** for the bucket:
   - Enable programmatic access
   - Set up appropriate IAM roles with S3 read/write permissions

### AWS Credentials Setup

Ensure AWS credentials are properly configured on your system:

```bash
aws configure
```

Enter your AWS Access Key ID, Secret Access Key, default region (e.g., ap-south-1), and output format.

Alternatively, you can set environment variables:

```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=ap-south-1
```

## EC2 Deployment Instructions

1. **Clone the repository to your EC2 instance**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up AWS credentials** as described above.

4. **Configure environment variables**:
   - Create a `.env` file based on the provided `env_example.txt` template
   - Set your database credentials, Razorpay API keys, and other configuration variables
   - For production, set the `REDIRECT_URL` to your public ngrok or domain URL

5. **Run the application**:
   ```bash
   streamlit run main.py
   ```

5. **Access the admin panel** (optional):
   ```bash
   streamlit run admin.py --server.port=8502
   ```

### Running as a Service

To run the application as a background service:

1. Create a systemd service file:
   ```bash
   sudo nano /etc/systemd/system/streamlit-app.service
   ```

2. Add the following content (replace paths as needed):
   ```
   [Unit]
   Description=Streamlit Application Service
   After=network.target

   [Service]
   User=ec2-user
   WorkingDirectory=/path/to/app
   ExecStart=/home/ec2-user/.local/bin/streamlit run /path/to/app/main.py --server.port=8501
   Restart=on-failure
   RestartSec=5s

   [Install]
   WantedBy=multi-user.target
   ```

3. Enable and start the service:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable streamlit-app
   sudo systemctl start streamlit-app
   ```

4. Check service status:
   ```bash
   sudo systemctl status streamlit-app
   ```

## Payment System Setup

### Razorpay Integration

This application uses Razorpay for handling premium subscription payments. To set up:

1. **Create a Razorpay account** at [razorpay.com](https://razorpay.com/)
2. **Get your API keys** from the Razorpay Dashboard
3. **Add your keys to the `.env` file**:
   ```
   RAZORPAY_KEY_ID=rzp_test_your_key_id_here
   RAZORPAY_KEY_SECRET=your_key_secret_here
   ```

### Payment Flow

1. User clicks on "Upgrade to Premium" in the application
2. They're redirected to a payment form (either inline or in a new tab)
3. After successful payment, Razorpay redirects to your application with transaction data
4. The application verifies the payment and upgrades the user to premium

### Running with HTTPS

For secure payments in production, you need HTTPS. Options include:

1. **Using ngrok for local development**:
   ```bash
   ngrok http 8501
   ```
   Then set `REDIRECT_URL` in `.env` to your ngrok URL

2. **Proper SSL setup in production** with a domain name and SSL certificate

### Database Setup for Transactions

The application tracks payment transactions in the database. Ensure your database has the necessary tables by running:

```bash
python -c "from db_storage import initialize_database; initialize_database()"
   ```

## File Structure

```
├── main.py                # Main application file
├── auth.py                # Authentication module
├── chatbot.py             # Chatbot functionality
├── s3_storage.py          # S3 storage integration
├── db_storage.py          # Database storage integration
├── razorpay_payment.py    # Payment system integration
├── payment.html           # Payment interface template
├── admin.py               # Admin panel
├── requirements.txt       # Required packages
├── styles.css             # CSS styling
└── logo.png               # Application logo
```

## Required Packages

Update your requirements.txt file with:

```
streamlit==1.32.0
pandas==2.2.0
numpy==1.26.0
matplotlib==3.8.2
boto3==1.34.0
chardet==5.2.0
pdfplumber==0.10.3
statsmodels==0.14.1
scikit-learn==1.4.0
prophet==1.1.5
plotly==5.18.0
razorpay==1.4.1
python-dotenv==1.0.0
sqlalchemy==2.0.27
psycopg2-binary==2.9.9
```

