# Claims Management System

A Streamlit application for processing and managing reimbursement claims with AI-powered document analysis using Azure Document Intelligence.

## Features

- **User Authentication**: Secure login system with admin and user roles
- **AI Document Processing**: Automatic extraction of data from certificates, invoices, and receipts
- **Cross-Verification**: Business logic validation across multiple document types
- **Admin Dashboard**: Comprehensive claim management and analytics
- **Real-time Processing**: Live document analysis with confidence scoring

## Prerequisites

- Python 3.8+
- Azure Document Intelligence resource
- Streamlit account (for deployment)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone [your-repo-url]
cd claims-management-system
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Azure Document Intelligence

1. Create an Azure Document Intelligence resource in Azure Portal
2. Train custom models with these exact names:
   - `Proofs` - for certificates and attendance proofs
   - `Invoices` - for billing and invoice documents  
   - `Receipts` - for payment receipts and transaction proofs

### 4. Setup Secrets

1. Copy the example secrets file:
   ```bash
   cp secrets.example.toml .streamlit/secrets.toml
   ```

2. Edit `.streamlit/secrets.toml` with your actual values:
   ```toml
   AZURE_DOC_INTELLIGENCE_ENDPOINT = "https://your-resource.cognitiveservices.azure.com/"
   AZURE_DOC_INTELLIGENCE_KEY = "your-actual-api-key"
   ```

### 5. Run the Application

```bash
streamlit run claims_new.py
```

## Demo Credentials

- **Admin**: admin@christ.edu / admin123
- **User**: Any email / Any password

## Deployment

### Streamlit Cloud

1. Push code to GitHub (secrets.toml is automatically ignored)
2. Connect repository to Streamlit Cloud
3. Add secrets in Streamlit Cloud dashboard under "Advanced settings"

### Other Platforms

For deployment on other platforms, ensure:
- Azure credentials are set as environment variables
- Custom models are trained and deployed
- All dependencies are installed

## Security Notes

- Never commit `secrets.toml` to version control
- Use environment variables in production
- Regularly rotate API keys
- Monitor Azure usage and costs

## File Structure

```
├── claims_new.py          # Main application file
├── .streamlit/
│   ├── secrets.toml       # Your secrets (not in git)
│   └── config.toml        # Optional Streamlit config
├── secrets.example.toml   # Template for secrets
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

[Add your license information here]

## Support

For issues and questions:
- Check the Azure Document Intelligence documentation
- Review Streamlit deployment guides
- Open an issue in this repository