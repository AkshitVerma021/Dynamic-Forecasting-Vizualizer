# Documentation: AI-Powered Multi-File Data Analysis & Forecasting**

# Overview

This Streamlit application provides powerful data analysis and forecasting capabilities for multiple file formats, integrating AWS Bedrock for AI-powered chat interactions.

# Key Features

- Multi-File Support:** Handles CSV, Excel, JSON, and Parquet files
- Automated Date Detection:** Intelligently identifies date columns for time series analysis
- Advanced Forecasting:** Uses ARIMA and Random Forest models for predictive analytics
- Interactive Visualizations:** Displays forecasts vs. actual data through matplotlib
- AI-Powered Chat:** Integrates with AWS Bedrock for dataset queries

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
- 10-day forecast generation with visualization

# Chat System

- Integration with AWS Bedrock for natural language processing
- Persistent chat history management
- Voice input support for queries

# Usage Requirements

- AWS Bedrock credentials configured
- Required Python packages: streamlit, pandas, numpy, matplotlib, boto3, statsmodels, scikit-learn
- Sufficient storage for chat history and model persistence

# Error Handling

- Robust file format validation
- Graceful fallback between forecasting models
- Clear error messaging for users

# This application combines modern data science techniques with user-friendly interfaces to provide comprehensive data analysis capabilities.

<img width="1679" alt="Screenshot 2025-03-20 at 6 59 58 PM" src="https://github.c<img width="1680" alt="Screenshot 2025-03-20 at 7 00 54 PM" src="https://github.com/user-attachments/assets/39233a68-593f-47cd-9947-04c609e51d50" />
<img width="860" alt="Screenshot 2025-03-20 at 7 02 11 PM" src="https://github.com/user-attachments/assets/1c3d934b-8d7f-48d5-b0f2-eaa36bb81599" />

