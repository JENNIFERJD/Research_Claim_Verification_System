"""
CHRIST University Claims Management System - FIXED VERSION
Fixed: DNS error, full-width layout, white calendar with visible text
"""
import ssl
import certifi
import os

# Fix SSL certificate verification
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

import pandas as pd
import streamlit as st
from datetime import datetime, date, timedelta
import uuid
import re
import logging
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass

# Azure imports
try:
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.core.credentials import AzureKeyCredential
    from azure.core.exceptions import HttpResponseError
    AZURE_SDK_AVAILABLE = True
except ImportError as e:
    st.error(f"Azure SDK import error: {e}")
    AZURE_SDK_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="CHRIST University Claims",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    # CHRIST University Brand Colors
    CHRIST_BLUE = "#003366"
    CHRIST_GOLD = "#FFD700"
    CHRIST_LIGHT_BLUE = "#E8F4F8"
    CHRIST_DARK_BLUE = "#002147"
    
    CUSTOM_MODELS = {
        'proof': 'Proofs',
        'invoice': 'prebuilt-invoice',
        'receipt': 'prebuilt-receipt'
    }
    
    FIELD_MAPPINGS = {
        'proof': {
            'faculty_name': ['Faculty_Name'],
            'date': ['Date'],
            'event_name': ['Event Name'],
            'organization_name': ['Organisation name']
        },
        'invoice': {
            'customer_name': ['BillingAddressRecipient', 'CustomerName'],
            'total_amount': ['InvoiceTotal', 'AmountDue', 'TotalAmount', 'Total'],
            'invoice_date': ['InvoiceDate', 'Date'],
        },
        'receipt': {
            'merchant_name': ['MerchantName'],
            'total_amount': ['Total'],
            'transaction_date': ['TransactionDate'],
            'account_number': ['AccountNumber']
        }
    }
    
    MAX_FILE_SIZE_MB = 50
    ALLOWED_FILE_TYPES = ['pdf', 'jpg', 'jpeg', 'png']
    MAX_CLAIM_AMOUNT = 100000
    AMOUNT_VARIANCE_TOLERANCE = 0.10
    DATE_VARIANCE_DAYS = 3  # ±3 days
    MIN_CONFIDENCE_THRESHOLD = 0.6
    ADMIN_CREDENTIALS = {"admin@christ.edu": "admin123"}
    FACULTY_CREDENTIALS = {"faculty@christ.edu": "faculty123"}
    
    @staticmethod
    def get_azure_endpoint() -> Optional[str]:
        try:
            endpoint = st.secrets.get("AZURE_DOC_INTELLIGENCE_ENDPOINT")
            if endpoint:
                # Remove trailing slash to prevent double slash issue
                return endpoint.rstrip('/')
            return None
        except:
            return None
    
    @staticmethod
    def get_azure_key() -> Optional[str]:
        try:
            return st.secrets.get("AZURE_DOC_INTELLIGENCE_KEY")
        except:
            return None


# ============================================================================
# SESSION STATE
# ============================================================================

class SessionState:
    @staticmethod
    def initialize():
        defaults = {
            'authenticated': False,
            'user_email': "",
            'user_name': "",
            'user_role': "",
            'page': "login",
            'claims_data': [],
            'return_to_dashboard': False,
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    @staticmethod
    def reset_user():
        st.session_state.authenticated = False
        st.session_state.user_email = ""
        st.session_state.user_name = ""
        st.session_state.user_role = ""
        st.session_state.page = "login"


# ============================================================================
# AZURE CLIENT - FIXED DNS ERROR
# ============================================================================

class AzureDocumentClient:
    _client: Optional[DocumentIntelligenceClient] = None
    
    @classmethod
    def initialize(cls) -> Tuple[bool, str]:
        try:
            if not AZURE_SDK_AVAILABLE:
                return False, "❌ Azure SDK not installed"
            
            endpoint = Config.get_azure_endpoint()
            key = Config.get_azure_key()
            
            if not endpoint or not key:
                return False, "❌ Azure credentials not configured"
            
            # CRITICAL FIX: Ensure endpoint doesn't have trailing slash
            # The SDK will add necessary paths
            endpoint = endpoint.rstrip('/')
            
            logger.info(f"Initializing Azure client with endpoint: {endpoint}")
            
            cls._client = DocumentIntelligenceClient(
                endpoint=endpoint,
                credential=AzureKeyCredential(key)
            )
            
            logger.info("✅ Azure client initialized")
            return True, "✅ Client ready"
        except Exception as e:
            logger.error(f"Init failed: {e}")
            return False, f"❌ Failed: {str(e)}"
    
    @classmethod
    def analyze_document(cls, file_bytes: bytes, model_id: str, filename: str) -> Any:
        if not cls._client:
            success, msg = cls.initialize()
            if not success:
                raise RuntimeError(msg)
        
        try:
            logger.info(f"Analyzing {filename} with {model_id}")
            
            poller = cls._client.begin_analyze_document(
                model_id=model_id,
                body=file_bytes,
                content_type="application/octet-stream"
            )
            
            result = poller.result()
            logger.info(f"✅ Success: {filename}")
            return result
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Analysis failed for {filename}: {error_msg}")
            raise Exception(f"Azure processing failed: {error_msg}")


# ============================================================================
# DOCUMENT PROCESSOR
# ============================================================================

class DocumentProcessor:
    
    @staticmethod
    def extract_field_value(result: Any, field_names: List[str]) -> Optional[str]:
        try:
            if not hasattr(result, 'documents') or not result.documents:
                return None
            
            document = result.documents[0]
            if not hasattr(document, 'fields') or not document.fields:
                return None
            
            for field_name in field_names:
                if field_name in document.fields:
                    field = document.fields[field_name]
                    
                    if field is None:
                        continue
                    
                    if hasattr(field, 'value'):
                        value = field.value
                        if value is not None:
                            if hasattr(value, 'content'):
                                extracted = str(value.content)
                            else:
                                extracted = str(value)
                            return extracted
                    
                    if hasattr(field, 'content'):
                        return str(field.content)
                    
                    if hasattr(field, 'value_string'):
                        return str(field.value_string)
            
            return None
            
        except Exception as e:
            logger.error(f"Field extraction error: {e}")
            return None
    
    @staticmethod
    def process_proof_document(file_bytes: bytes, filename: str) -> Dict:
        try:
            result = AzureDocumentClient.analyze_document(file_bytes, 'Proofs', filename)
            
            extracted = {
                'success': True,
                'document_type': 'proof',
                'filename': filename,
                'faculty_name': DocumentProcessor.extract_field_value(
                    result, Config.FIELD_MAPPINGS['proof']['faculty_name']
                ),
                'date': DocumentProcessor.extract_field_value(
                    result, Config.FIELD_MAPPINGS['proof']['date']
                ),
                'event_name': DocumentProcessor.extract_field_value(
                    result, Config.FIELD_MAPPINGS['proof']['event_name']
                ),
                'organization_name': DocumentProcessor.extract_field_value(
                    result, Config.FIELD_MAPPINGS['proof']['organization_name']
                ),
            }
            
            return extracted
        except Exception as e:
            return {
                'success': False,
                'document_type': 'proof',
                'filename': filename,
                'error': str(e)
            }
    
    @staticmethod
    def process_invoice_document(file_bytes: bytes, filename: str) -> Dict:
        try:
            result = AzureDocumentClient.analyze_document(file_bytes, 'prebuilt-invoice', filename)
            
            extracted = {
                'success': True,
                'document_type': 'invoice',
                'filename': filename,
                'customer_name': DocumentProcessor.extract_field_value(
                    result, Config.FIELD_MAPPINGS['invoice']['customer_name']
                ),
                'total_amount': DocumentProcessor.extract_field_value(
                    result, Config.FIELD_MAPPINGS['invoice']['total_amount']
                ),
                'invoice_date': DocumentProcessor.extract_field_value(
                    result, Config.FIELD_MAPPINGS['invoice']['invoice_date']
                ),
            }
            
            return extracted
        except Exception as e:
            return {
                'success': False,
                'document_type': 'invoice',
                'filename': filename,
                'error': str(e)
            }
    
    @staticmethod
    def process_receipt_document(file_bytes: bytes, filename: str) -> Dict:
        try:
            result = AzureDocumentClient.analyze_document(file_bytes, 'prebuilt-receipt', filename)
            
            extracted = {
                'success': True,
                'document_type': 'receipt',
                'filename': filename,
                'merchant_name': DocumentProcessor.extract_field_value(
                    result, Config.FIELD_MAPPINGS['receipt']['merchant_name']
                ),
                'total_amount': DocumentProcessor.extract_field_value(
                    result, Config.FIELD_MAPPINGS['receipt']['total_amount']
                ),
                'transaction_date': DocumentProcessor.extract_field_value(
                    result, Config.FIELD_MAPPINGS['receipt']['transaction_date']
                ),
                'account_number': DocumentProcessor.extract_field_value(
                    result, Config.FIELD_MAPPINGS['receipt']['account_number']
                ),
            }
            
            return extracted
        except Exception as e:
            return {
                'success': False,
                'document_type': 'receipt',
                'filename': filename,
                'error': str(e)
            }


# ============================================================================
# VERIFICATION ENGINE
# ============================================================================

class VerificationEngine:
    
    @staticmethod
    def normalize_name(name: str) -> str:
        """Normalize name by removing titles and special characters"""
        if not name:
            return ""
        
        name = str(name).lower().strip()
        
        titles = ['dr', 'mr', 'mrs', 'ms', 'miss', 'prof', 'professor', 'dr.', 'mr.', 'mrs.', 'ms.', 'm/s']
        words = name.split()
        filtered_words = [w for w in words if w not in titles and w.rstrip('.') not in titles]
        
        name = ' '.join(filtered_words)
        name = re.sub(r'[^\w\s]', '', name)
        name = re.sub(r'\s+', ' ', name).strip()
        
        return name
    
    @staticmethod
    def names_match(name1: str, name2: str) -> bool:
        """Smart name matching that handles titles and variations"""
        if not name1 or not name2:
            return False
        
        norm1 = VerificationEngine.normalize_name(name1)
        norm2 = VerificationEngine.normalize_name(name2)
        
        if not norm1 or not norm2:
            return False
        
        if norm1 == norm2:
            return True
        
        if norm1 in norm2 or norm2 in norm1:
            return True
        
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        
        common_words = words1.intersection(words2)
        if len(common_words) >= 2:
            return True
        
        if len(words1) <= 2 and len(words2) <= 2 and len(common_words) >= 1:
            return True
        
        return False
    
    @staticmethod
    def normalize_account(acc: str) -> str:
        """Normalize account number by removing spaces and special chars"""
        if not acc:
            return ""
        return re.sub(r'[^a-zA-Z0-9]', '', str(acc)).upper()
    
    @staticmethod
    def parse_date(date_str: str) -> Optional[date]:
        """Parse date from various formats"""
        if not date_str:
            return None
        
        date_formats = [
            '%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y',
            '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y',
            '%d %B %Y', '%B %d, %Y', '%d %b %Y',
            '%d-%b-%Y', '%d-%B-%Y',
            '%d-%b-%y %I:%M:%S %p',
            '%d-%B-%y %I:%M:%S %p'
        ]
        
        date_str = str(date_str).strip()
        
        for fmt in date_formats:
            try:
                parsed = datetime.strptime(date_str, fmt)
                return parsed.date()
            except:
                continue
        
        return None
    
    @staticmethod
    def extract_amount(value: Any) -> Optional[float]:
        """Extract numeric amount from string or number"""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            clean = re.sub(r'[₹$£€,\s]', '', value)
            match = re.search(r'(\d+(?:\.\d{1,2})?)', clean)
            if match:
                try:
                    return float(match.group(1))
                except:
                    pass
        return None
    
    @staticmethod
    def perform_full_verification(form_data: Dict, documents: Dict) -> Dict:
        """
        STRICT VERIFICATION LOGIC:
        - APPROVE only if: (Name + Amount + Date) OR (Account + Amount + Date) all match
        - Date matching is MANDATORY for approval
        - Any mismatch = Pending Admin Review
        """
        issues = []
        high_severity_issues = []
        
        proof_data = documents.get('proofs', [])
        invoice_data = documents.get('invoices', [])
        receipt_data = documents.get('receipts', [])
        
        form_name = form_data.get('faculty_name', '')
        form_account = form_data.get('account_number', '')
        form_amount = form_data.get('total_amount', 0)
        form_event_date = form_data.get('event_date')
        
        extracted_any_data = False
        
        name_verified = False
        account_verified = False
        amount_verified = False
        date_verified = False
        
        # Extract from proofs
        proof_names = []
        proof_dates = []
        
        for proof in proof_data:
            if proof and proof.get('success'):
                if proof.get('faculty_name'):
                    extracted_any_data = True
                    proof_name = proof['faculty_name']
                    proof_names.append(proof_name)
                    
                    if form_name and proof_name:
                        if VerificationEngine.names_match(form_name, proof_name):
                            name_verified = True
                        else:
                            issues.append({
                                'severity': 'medium',
                                'message': f"⚠️ Name difference: Form '{form_name}' vs Certificate '{proof_name}'"
                            })
                
                if proof.get('date'):
                    extracted_any_data = True
                    proof_date = VerificationEngine.parse_date(proof['date'])
                    if proof_date:
                        proof_dates.append(proof_date)
            else:
                issues.append({
                    'severity': 'high',
                    'message': f"❌ Certificate processing failed: {proof.get('filename', 'unknown')}"
                })
                high_severity_issues.append("Processing failed")
        
        # Extract from receipts
        receipt_accounts = []
        receipt_dates = []
        receipt_amounts = []
        receipt_names = []
        
        for receipt in receipt_data:
            if receipt and receipt.get('success'):
                if receipt.get('merchant_name'):
                    extracted_any_data = True
                    merchant = receipt['merchant_name']
                    receipt_names.append(merchant)
                
                if receipt.get('account_number'):
                    extracted_any_data = True
                    receipt_acc = receipt['account_number']
                    receipt_accounts.append(receipt_acc)
                    
                    if form_account and receipt_acc:
                        norm_form = VerificationEngine.normalize_account(form_account)
                        norm_receipt = VerificationEngine.normalize_account(receipt_acc)
                        
                        if norm_form and norm_receipt:
                            if norm_form == norm_receipt:
                                account_verified = True
                            else:
                                issues.append({
                                    'severity': 'medium',
                                    'message': f"⚠️ Account difference: Form '{form_account}' vs Receipt '{receipt_acc}'"
                                })
                
                if receipt.get('transaction_date'):
                    extracted_any_data = True
                    receipt_date = VerificationEngine.parse_date(receipt['transaction_date'])
                    if receipt_date:
                        receipt_dates.append(receipt_date)
                
                if receipt.get('total_amount'):
                    extracted_any_data = True
                    amount = VerificationEngine.extract_amount(receipt['total_amount'])
                    if amount and amount > 0:
                        receipt_amounts.append(amount)
            else:
                issues.append({
                    'severity': 'high',
                    'message': f"❌ Receipt processing failed: {receipt.get('filename', 'unknown')}"
                })
                high_severity_issues.append("Processing failed")
        
        # Extract from invoices
        invoice_dates = []
        invoice_amounts = []
        invoice_names = []
        
        for invoice in invoice_data:
            if invoice and invoice.get('success'):
                if invoice.get('customer_name'):
                    extracted_any_data = True
                    cust_name = invoice['customer_name']
                    invoice_names.append(cust_name)
                
                if invoice.get('invoice_date'):
                    extracted_any_data = True
                    inv_date = VerificationEngine.parse_date(invoice['invoice_date'])
                    if inv_date:
                        invoice_dates.append(inv_date)
                
                if invoice.get('total_amount'):
                    extracted_any_data = True
                    amount = VerificationEngine.extract_amount(invoice['total_amount'])
                    if amount and amount > 0:
                        invoice_amounts.append(amount)
            else:
                issues.append({
                    'severity': 'high',
                    'message': f"❌ Invoice processing failed: {invoice.get('filename', 'unknown')}"
                })
                high_severity_issues.append("Processing failed")
        
        # Check name matching
        all_document_names = receipt_names + invoice_names
        if form_name and all_document_names and not name_verified:
            for doc_name in all_document_names:
                if VerificationEngine.names_match(form_name, doc_name):
                    name_verified = True
                    break
        
        # Check amount matching
        all_amounts = receipt_amounts + invoice_amounts
        
        if form_amount > 0 and all_amounts:
            for extracted_amt in all_amounts:
                variance = abs(form_amount - extracted_amt) / max(form_amount, extracted_amt)
                if variance <= Config.AMOUNT_VARIANCE_TOLERANCE:
                    amount_verified = True
                    break
            
            if not amount_verified:
                issues.append({
                    'severity': 'high',
                    'message': f"❌ AMOUNT MISMATCH: Form amount ₹{form_amount:.2f} does not match extracted amounts {['₹' + f'{a:.2f}' for a in all_amounts]}"
                })
                high_severity_issues.append("Amount mismatch")
        
        # Check date matching - CRITICAL FOR APPROVAL
        all_financial_dates = receipt_dates + invoice_dates
        
        if proof_dates and all_financial_dates:
            # Check if dates from proof certificates match financial documents
            for proof_date in proof_dates:
                for fin_date in all_financial_dates:
                    days_diff = abs((fin_date - proof_date).days)
                    if days_diff <= Config.DATE_VARIANCE_DAYS:
                        date_verified = True
                        break
                if date_verified:
                    break
            
            if not date_verified:
                issues.append({
                    'severity': 'high',
                    'message': f"❌ DATE MISMATCH: Certificate dates {[d.strftime('%Y-%m-%d') for d in proof_dates]} do not match transaction dates {[d.strftime('%Y-%m-%d') for d in all_financial_dates]} (±{Config.DATE_VARIANCE_DAYS} days tolerance)"
                })
                high_severity_issues.append("Date mismatch")
        elif not proof_dates or not all_financial_dates:
            # Missing date information - cannot verify
            issues.append({
                'severity': 'high',
                'message': f"❌ MISSING DATE DATA: Cannot verify dates - Proof dates: {len(proof_dates)}, Financial dates: {len(all_financial_dates)}"
            })
            high_severity_issues.append("Missing date data")
            date_verified = False
        
        # Determine final status - STRICT: Date matching is MANDATORY
        identity_verified = name_verified or account_verified
        core_verification_passed = identity_verified and amount_verified and date_verified
        
        if not identity_verified:
            if not name_verified and not account_verified:
                issues.append({
                    'severity': 'high',
                    'message': f"❌ IDENTITY NOT VERIFIED: Neither name nor account number could be matched"
                })
                high_severity_issues.append("Identity verification failed")
        
        if not date_verified and (proof_dates or all_financial_dates):
            issues.append({
                'severity': 'high',
                'message': f"❌ DATE MISMATCH: Date verification failed - this blocks automatic approval"
            })
            high_severity_issues.append("Date verification failed")
        
        # APPROVAL CRITERIA: Must have Identity + Amount + Date ALL verified
        if core_verification_passed and len(high_severity_issues) == 0:
            status = 'APPROVED'
            verification_method = []
            if name_verified:
                verification_method.append("Name")
            if account_verified:
                verification_method.append("Account")
            status_message = f'✅ Claim APPROVED - Verified by: {" + ".join(verification_method)} + Amount + Date match'
        else:
            # ANY failure = Pending Admin Review
            status = 'PENDING_ADMIN'
            if len(high_severity_issues) > 0:
                status_message = f'⚠️ {len(high_severity_issues)} critical issue(s) - Requires Admin Review'
            else:
                status_message = '⚠️ Verification incomplete - Requires Admin Review'
        
        return {
            'status': status,
            'status_message': status_message,
            'is_approved': status == 'APPROVED',
            'is_rejected': status == 'REJECTED',
            'all_issues': issues,
            'high_severity_count': len(high_severity_issues),
            'verification_details': {
                'name_verified': name_verified,
                'account_verified': account_verified,
                'amount_verified': amount_verified,
                'date_verified': date_verified
            },
            'extracted_data': {
                'names': {
                    'form': form_name,
                    'proof_certificates': proof_names,
                    'financial_documents': all_document_names
                },
                'accounts': {
                    'form': form_account,
                    'receipts': receipt_accounts
                },
                'dates': {
                    'proof_dates': [d.strftime('%Y-%m-%d') for d in proof_dates],
                    'transaction_dates': [d.strftime('%Y-%m-%d') for d in all_financial_dates]
                },
                'amounts': {
                    'form': form_amount,
                    'extracted': all_amounts
                }
            }
        }


# ============================================================================
# AUTHENTICATION
# ============================================================================

class AuthManager:
    
    @staticmethod
    def authenticate(email: str, password: str) -> bool:
        if not email or not password:
            return False
        
        if email in Config.ADMIN_CREDENTIALS:
            if Config.ADMIN_CREDENTIALS[email] == password:
                st.session_state.authenticated = True
                st.session_state.user_email = email
                st.session_state.user_name = "Admin"
                st.session_state.user_role = "admin"
                st.session_state.page = "admin_dashboard"
                return True
        
        if email in Config.FACULTY_CREDENTIALS:
            if Config.FACULTY_CREDENTIALS[email] == password:
                st.session_state.authenticated = True
                st.session_state.user_email = email
                st.session_state.user_name = email.split('@')[0].title()
                st.session_state.user_role = "faculty"
                st.session_state.page = "faculty_dashboard"
                return True
        
        return False
    
    @staticmethod
    def logout():
        SessionState.reset_user()
        st.rerun()


# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================

def process_all_documents(proof_files, invoice_files, receipt_files) -> Dict:
    if not AzureDocumentClient._client:
        success, msg = AzureDocumentClient.initialize()
        if not success:
            st.error(msg)
            return None
    
    results = {'proofs': [], 'invoices': [], 'receipts': []}
    total_files = len(proof_files or []) + len(invoice_files or []) + len(receipt_files or [])
    
    if total_files == 0:
        return results
    
    progress = st.progress(0)
    current = 0
    
    try:
        if proof_files:
            for file in proof_files:
                current += 1
                progress.progress(current / total_files)
                
                file_bytes = file.read()
                file.seek(0)
                
                result = DocumentProcessor.process_proof_document(file_bytes, file.name)
                results['proofs'].append(result)
        
        if invoice_files:
            for file in invoice_files:
                current += 1
                progress.progress(current / total_files)
                
                file_bytes = file.read()
                file.seek(0)
                
                result = DocumentProcessor.process_invoice_document(file_bytes, file.name)
                results['invoices'].append(result)
        
        if receipt_files:
            for file in receipt_files:
                current += 1
                progress.progress(current / total_files)
                
                file_bytes = file.read()
                file.seek(0)
                
                result = DocumentProcessor.process_receipt_document(file_bytes, file.name)
                results['receipts'].append(result)
        
        progress.progress(1.0)
        return results
    finally:
        progress.empty()


def generate_claim_id() -> str:
    year = datetime.now().year
    random = str(uuid.uuid4())[:6].upper()
    return f"RCL-{year}-{random}"


# ============================================================================
# STYLING - FULL WIDTH + WHITE CALENDAR WITH VISIBLE TEXT
# ============================================================================

def apply_custom_css():
    st.markdown(f"""
    <style>
        /* FULL WIDTH LAYOUT */
        .block-container {{
            max-width: 100% !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }}
        
        /* Force light mode with high contrast */
        .stApp {{
            background-color: #F5F7FA;
        }}
        
        /* All text dark and visible */
        * {{
            color: #1a1a1a !important;
        }}
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {{
            color: {Config.CHRIST_DARK_BLUE} !important;
            font-weight: 600 !important;
        }}
        
        /* Input fields - high contrast */
        .stTextInput input,
        .stTextArea textarea,
        .stNumberInput input,
        .stDateInput input,
        .stSelectbox select {{
            background-color: white !important;
            color: #000000 !important;
            border: 2px solid {Config.CHRIST_BLUE} !important;
            border-radius: 8px !important;
            padding: 10px !important;
            font-size: 16px !important;
            font-weight: 500 !important;
        }}
        
        /* CRITICAL: Date input - PURE WHITE BACKGROUND, BLACK TEXT */
        .stDateInput > div > div > input {{
            background-color: #FFFFFF !important;
            color: #000000 !important;
            font-weight: 600 !important;
            caret-color: #000000 !important;
        }}
        
        /* CRITICAL: Calendar popup - PURE WHITE BACKGROUND */
        [data-baseweb="calendar"] {{
            background-color: #FFFFFF !important;
            border: 2px solid {Config.CHRIST_BLUE} !important;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15) !important;
        }}
        
        /* All calendar internal elements - WHITE */
        [data-baseweb="calendar"] > div {{
            background-color: #FFFFFF !important;
        }}
        
        [data-baseweb="calendar"] * {{
            background-color: #FFFFFF !important;
        }}
        
        /* Calendar header - DARK background with WHITE text */
        [data-baseweb="calendar"] [data-baseweb="calendar-header"] {{
            background-color: #2C3E50 !important;
            padding: 12px !important;
        }}
        
        [data-baseweb="calendar"] [data-baseweb="calendar-header"] * {{
            background-color: #2C3E50 !important;
            color: #FFFFFF !important;
        }}
        
        /* Month/Year text */
        [data-baseweb="calendar"] button[aria-live="polite"] {{
            background-color: #2C3E50 !important;
            color: #FFFFFF !important;
        }}
        
        /* Navigation arrows - WHITE */
        [data-baseweb="calendar"] button[aria-label*="Previous"],
        [data-baseweb="calendar"] button[aria-label*="Next"] {{
            background-color: #2C3E50 !important;
        }}
        
        [data-baseweb="calendar"] button svg {{
            color: #FFFFFF !important;
            fill: #FFFFFF !important;
        }}
        
        /* Weekday names (Mon, Tue, Wed...) - WHITE BG, BLACK TEXT */
        [data-baseweb="calendar"] [role="presentation"] {{
            background-color: #FFFFFF !important;
        }}
        
        [data-baseweb="calendar"] [role="presentation"] * {{
            background-color: #FFFFFF !important;
            color: #000000 !important;
            font-weight: 700 !important;
        }}
        
        /* ALL day cells - PURE WHITE with BLACK TEXT */
        [data-baseweb="calendar"] [role="gridcell"] {{
            background-color: #FFFFFF !important;
        }}
        
        [data-baseweb="calendar"] [role="gridcell"] div {{
            background-color: #FFFFFF !important;
            color: #000000 !important;
            font-weight: 600 !important;
        }}
        
        /* Day number buttons - WHITE with BLACK text */
        [data-baseweb="calendar"] [role="button"]:not([aria-label*="Previous"]):not([aria-label*="Next"]) {{
            background-color: #FFFFFF !important;
            color: #000000 !important;
            font-weight: 600 !important;
            border: none !important;
        }}
        
        /* Selected date - RED circle like your screenshot */
        [data-baseweb="calendar"] [aria-selected="true"] {{
            background-color: #EF4444 !important;
            color: #FFFFFF !important;
            border-radius: 50% !important;
        }}
        
        [data-baseweb="calendar"] [aria-selected="true"] div {{
            background-color: #EF4444 !important;
            color: #FFFFFF !important;
        }}
        
        /* Hover state - Light gray */
        [data-baseweb="calendar"] [role="button"]:hover:not([aria-selected="true"]) {{
            background-color: #F3F4F6 !important;
            color: #000000 !important;
        }}
        
        /* Today's date - border indicator */
        [data-baseweb="calendar"] [data-highlighted="true"]:not([aria-selected="true"]) {{
            background-color: #FFFFFF !important;
            border: 2px solid {Config.CHRIST_BLUE} !important;
            border-radius: 50% !important;
            color: #000000 !important;
            font-weight: 700 !important;
        }}
        
        /* Disabled/outside month dates */
        [data-baseweb="calendar"] [disabled],
        [data-baseweb="calendar"] [aria-disabled="true"] {{
            background-color: #FFFFFF !important;
            color: #9CA3AF !important;
            opacity: 0.4 !important;
        }}
        
        /* Labels */
        label {{
            color: #1a1a1a !important;
            font-weight: 600 !important;
            font-size: 15px !important;
        }}
        
        /* Header */
        .christ-header {{
            background: linear-gradient(135deg, {Config.CHRIST_BLUE} 0%, {Config.CHRIST_DARK_BLUE} 100%);
            padding: 25px 40px;
            margin: -20px -20px 30px -20px;
            border-bottom: 5px solid {Config.CHRIST_GOLD};
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }}
        
        .christ-logo {{
            font-size: 2.5rem;
            color: white !important;
            vertical-align: middle;
            margin-right: 15px;
        }}
        
        .christ-title {{
            color: {Config.CHRIST_GOLD} !important;
            font-size: 2rem !important;
            font-weight: 700 !important;
            letter-spacing: 2px;
            margin: 0 !important;
        }}
        
        .christ-subtitle {{
            color: white !important;
            font-size: 1rem !important;
            margin: 5px 0 0 0 !important;
        }}
        
        /* Status boxes - HIGH VISIBILITY */
        .status-box {{
            padding: 25px;
            margin: 20px 0;
            border-radius: 12px;
            border: 3px solid;
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        }}
        
        .status-approved {{
            background: white;
            border-color: #10b981;
        }}
        
        .status-approved h2 {{
            color: #059669 !important;
        }}
        
        .status-rejected {{
            background: white;
            border-color: #ef4444;
        }}
        
        .status-rejected h2 {{
            color: #dc2626 !important;
        }}
        
        .status-pending {{
            background: white;
            border-color: #f59e0b;
        }}
        
        .status-pending h2 {{
            color: #d97706 !important;
        }}
        
        .status-title {{
            font-size: 1.8rem !important;
            font-weight: 700 !important;
            margin-bottom: 10px !important;
        }}
        
        .status-message {{
            font-size: 1.1rem !important;
            color: #1a1a1a !important;
            line-height: 1.6;
        }}
        
        /* Buttons */
        .stButton > button {{
            background: {Config.CHRIST_BLUE};
            color: white !important;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: 600;
            font-size: 16px;
            transition: all 0.3s;
        }}
        
        .stButton > button:hover {{
            background: {Config.CHRIST_DARK_BLUE};
            box-shadow: 0 4px 12px rgba(0,51,102,0.4);
        }}
        
        /* Metrics */
        [data-testid="stMetricValue"] {{
            color: {Config.CHRIST_BLUE} !important;
            font-size: 2rem !important;
            font-weight: 700 !important;
        }}
        
        [data-testid="stMetricLabel"] {{
            color: #1a1a1a !important;
            font-weight: 600 !important;
        }}
        
        /* Expander */
        .streamlit-expanderHeader {{
            background-color: white !important;
            border: 2px solid {Config.CHRIST_BLUE} !important;
            border-radius: 8px !important;
            color: #1a1a1a !important;
            font-weight: 600 !important;
        }}
        
        /* File uploader */
        [data-testid="stFileUploader"] {{
            background: white;
            border: 3px dashed {Config.CHRIST_BLUE};
            border-radius: 12px;
            padding: 20px;
        }}
        
        /* Alert boxes */
        .stAlert {{
            background-color: white !important;
            color: #1a1a1a !important;
            border-left: 5px solid;
            padding: 15px;
            border-radius: 8px;
        }}
        
        /* Progress bar */
        .stProgress > div > div {{
            background-color: {Config.CHRIST_BLUE} !important;
        }}
        
        /* Info/Success/Error text visibility */
        .stSuccess, .stInfo, .stWarning, .stError {{
            color: #1a1a1a !important;
        }}
        
        /* Sidebar (if used) */
        [data-testid="stSidebar"] {{
            background-color: white !important;
        }}
    </style>
    """, unsafe_allow_html=True)


def render_header(show_logout=True):
    """Render CHRIST University header"""
    st.markdown(f"""
    <div class="christ-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="display: flex; align-items: center;">
                <span class="christ-logo">🏛️</span>
                <div>
                    <div class="christ-title">CHRIST UNIVERSITY</div>
                    <div class="christ-subtitle">Claims Management System</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if show_logout and st.session_state.authenticated:
        col1, col2, col3 = st.columns([7, 1, 1])
        with col3:
            if st.button("🚪 Logout", key="logout_btn", use_container_width=True):
                AuthManager.logout()


# ============================================================================
# UI PAGES
# ============================================================================

def login_page():
    render_header(show_logout=False)
    
    st.markdown("## 🔐 Faculty & Admin Login")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.form("login_form"):
            email = st.text_input("📧 Email Address", placeholder="your.email@christ.edu")
            password = st.text_input("🔑 Password", type="password", placeholder="Enter password")
            
            submitted = st.form_submit_button("🚀 Login", use_container_width=True)
            
            if submitted:
                if AuthManager.authenticate(email, password):
                    st.success("✅ Login successful! Redirecting...")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials. Please check your email and password.")
        
        with st.expander("📋 Demo Credentials"):
            st.info("""
**Admin Access:**
- Email: `admin@christ.edu`
- Password: `admin123`

**Faculty Access:**
- Email: `faculty@christ.edu`
- Password: `faculty123`
            """)


def faculty_dashboard_page():
    render_header()
    
    st.markdown(f"## 👋 Welcome, {st.session_state.user_name}!")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("📝 Submit New Claim", type="primary", use_container_width=True):
            st.session_state.page = "claim_form"
            st.rerun()
    
    st.markdown("---")
    st.markdown("### 📋 My Claims History")
    
    my_claims = [c for c in st.session_state.claims_data if c['user_email'] == st.session_state.user_email]
    
    if not my_claims:
        st.info("🔭 No claims submitted yet. Click 'Submit New Claim' to get started.")
    else:
        for claim in my_claims:
            status_icons = {
                'APPROVED': '🟢',
                'REJECTED': '🔴',
                'PENDING_ADMIN': '🟡'
            }
            icon = status_icons.get(claim['status'], '⚪')
            
            with st.expander(f"{icon} **{claim['id']}** - {claim['status']} - ₹{claim['total_amount']:,.2f}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📅 Event Details:**")
                    st.write(f"Event: {claim['event_name']}")
                    st.write(f"Date: {claim['event_date']}")
                
                with col2:
                    st.markdown("**💰 Claim Status:**")
                    st.write(f"Amount: ₹{claim['total_amount']:,.2f}")
                    st.write(f"Submitted: {claim['submission_date']}")
                
                if 'verification_result' in claim:
                    st.markdown("---")
                    status = claim['verification_result']['status']
                    message = claim['verification_result']['status_message']
                    
                    if status == 'APPROVED':
                        st.markdown(f"""
                        <div class="status-box status-approved">
                            <h2 class="status-title">✅ Approved</h2>
                            <p class="status-message">{message}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif status == 'REJECTED':
                        st.markdown(f"""
                        <div class="status-box status-rejected">
                            <h2 class="status-title">❌ Rejected</h2>
                            <p class="status-message">{message}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="status-box status-pending">
                            <h2 class="status-title">⏳ Pending Review</h2>
                            <p class="status-message">{message}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if claim['verification_result']['all_issues']:
                        st.markdown("**⚠️ Issues Detected:**")
                        for issue in claim['verification_result']['all_issues']:
                            severity = "🔴" if issue['severity'] == 'high' else "🟡"
                            st.write(f"{severity} {issue['message']}")


def claim_form_page():
    render_header()
    
    col1, col2 = st.columns([6, 1])
    with col1:
        st.markdown("## 📝 Submit Reimbursement Claim")
    with col2:
        if st.button("← Back", use_container_width=True):
            st.session_state.page = "faculty_dashboard"
            st.rerun()
    
    st.markdown("---")
    
    with st.form("claim_form", clear_on_submit=False):
        st.markdown("### 👤 Personal Information")
        col1, col2 = st.columns(2)
        with col1:
            faculty_name = st.text_input("Full Name *", placeholder="e.g., Dr. John Smith or John Smith")
        with col2:
            emp_id = st.text_input("Employee ID *", placeholder="EMP12345")
        
        col1, col2 = st.columns(2)
        with col1:
            department = st.text_input("Department *", placeholder="e.g., Computer Science")
        with col2:
            account_number = st.text_input("Bank Account Number *", placeholder="1234567890")
        
        st.markdown("---")
        st.markdown("### 📅 Event Information")
        event_name = st.text_area("Event/Conference Name *", height=100, placeholder="International Conference on Artificial Intelligence 2025")
        event_date = st.date_input("Event Date *", value=datetime.now().date())
        
        st.markdown("---")
        st.markdown("### 💰 Financial Details")
        col1, col2, col3 = st.columns(3)
        with col1:
            registration_fee = st.number_input("Registration Fee (₹)", min_value=0.0, step=100.0, format="%.2f")
        with col2:
            travel_amount = st.number_input("Travel Expenses (₹)", min_value=0.0, step=100.0, format="%.2f")
        with col3:
            other_expenses = st.number_input("Other Expenses (₹)", min_value=0.0, step=100.0, format="%.2f")
        
        total_amount = registration_fee + travel_amount + other_expenses
        if total_amount > 0:
            st.markdown(f"### **Total Claim: ₹{total_amount:,.2f}**")
        
        st.markdown("---")
        st.markdown("### 📤 Upload Supporting Documents")
        
        st.warning("""
**📋 REQUIRED DOCUMENTS:**
- ✅ **Certificates/Proofs**: At least 1 certificate showing your participation
- ✅ **Financial Documents**: Either Invoice OR Receipt
- ⚠️ **STRICT VERIFICATION**: For automatic approval, system requires:
  - **(Name + Amount + Date) match** OR **(Account + Amount + Date) match**
  - **Date matching is MANDATORY** - claims without date match go to admin review
- 💡 **Note**: All three fields (identity + amount + date) must match for approval
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📜 Certificates** *(Required)*")
            proof_files = st.file_uploader(
                "Upload certificates",
                accept_multiple_files=True,
                type=Config.ALLOWED_FILE_TYPES,
                key="proofs"
            )
            if proof_files:
                st.success(f"✅ {len(proof_files)} file(s)")
        
        with col2:
            st.markdown("**🧾 Invoices**")
            invoice_files = st.file_uploader(
                "Upload invoices",
                accept_multiple_files=True,
                type=Config.ALLOWED_FILE_TYPES,
                key="invoices"
            )
            if invoice_files:
                st.success(f"✅ {len(invoice_files)} file(s)")
        
        with col3:
            st.markdown("**💳 Receipts**")
            receipt_files = st.file_uploader(
                "Upload receipts",
                accept_multiple_files=True,
                type=Config.ALLOWED_FILE_TYPES,
                key="receipts"
            )
            if receipt_files:
                st.success(f"✅ {len(receipt_files)} file(s)")
        
        has_certs = proof_files and len(proof_files) > 0
        has_financial = (invoice_files and len(invoice_files) > 0) or (receipt_files and len(receipt_files) > 0)
        
        st.markdown("---")
        
        if not has_certs or not has_financial:
            st.error("⚠️ **Cannot Submit**: Missing required documents!")
        
        submitted = st.form_submit_button("🚀 Submit Claim for Verification", type="primary", use_container_width=True)
        
        if submitted:
            errors = []
            
            if not faculty_name or len(faculty_name.strip()) < 3:
                errors.append("❌ Full name is required (minimum 3 characters)")
            if not emp_id:
                errors.append("❌ Employee ID is required")
            if not department:
                errors.append("❌ Department is required")
            if not account_number or len(account_number.strip()) < 4:
                errors.append("❌ Valid bank account number is required")
            if not event_name:
                errors.append("❌ Event name is required")
            if total_amount <= 0:
                errors.append("❌ Total amount must be greater than 0")
            if not has_certs:
                errors.append("❌ At least 1 certificate is required")
            if not has_financial:
                errors.append("❌ Either invoice OR receipt is required")
            
            if errors:
                st.error("### ❌ Submission Blocked - Please fix the following:")
                for error in errors:
                    st.error(error)
            else:
                claim_id = generate_claim_id()
                
                with st.spinner(f"🔄 Processing claim {claim_id}..."):
                    st.info("📄 Analyzing documents with AI...")
                    
                    processed_docs = process_all_documents(proof_files, invoice_files, receipt_files)
                    
                    if processed_docs:
                        st.info("🔍 Cross-verifying extracted data...")
                        
                        form_data = {
                            'faculty_name': faculty_name,
                            'account_number': account_number,
                            'event_name': event_name,
                            'event_date': event_date,
                            'total_amount': total_amount
                        }
                        
                        verification = VerificationEngine.perform_full_verification(form_data, processed_docs)
                        
                        new_claim = {
                            'id': claim_id,
                            'user_email': st.session_state.user_email,
                            'faculty_name': faculty_name,
                            'emp_id': emp_id,
                            'department': department,
                            'account_number': account_number,
                            'event_name': event_name,
                            'event_date': event_date.strftime('%Y-%m-%d'),
                            'total_amount': total_amount,
                            'status': verification['status'],
                            'submission_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'verification_result': verification,
                            'processed_documents': processed_docs,
                        }
                        
                        st.session_state.claims_data.append(new_claim)
                        
                        st.markdown("---")
                        st.markdown("## 📊 Verification Results")
                        
                        status = verification['status']
                        message = verification['status_message']
                        
                        if status == 'APPROVED':
                            st.markdown(f"""
                            <div class="status-box status-approved">
                                <h2 class="status-title">✅ Claim Approved!</h2>
                                <p class="status-message">{message}</p>
                                <p class="status-message"><strong>Claim ID:</strong> {claim_id}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            st.balloons()
                        elif status == 'REJECTED':
                            st.markdown(f"""
                            <div class="status-box status-rejected">
                                <h2 class="status-title">❌ Claim Rejected</h2>
                                <p class="status-message">{message}</p>
                                <p class="status-message"><strong>Claim ID:</strong> {claim_id}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="status-box status-pending">
                                <h2 class="status-title">⏳ Pending Admin Review</h2>
                                <p class="status-message">{message}</p>
                                <p class="status-message"><strong>Claim ID:</strong> {claim_id}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        if verification['all_issues']:
                            st.markdown("### ⚠️ Issues Detected During Verification:")
                            for idx, issue in enumerate(verification['all_issues'], 1):
                                severity_icon = "🔴" if issue['severity'] == 'high' else "🟡"
                                severity_text = "HIGH PRIORITY" if issue['severity'] == 'high' else "MEDIUM"
                                st.error(f"{severity_icon} **Issue #{idx} - {severity_text}**\n\n{issue['message']}")
                        
                        if 'verification_details' in verification:
                            st.markdown("### ✅ Verification Summary:")
                            details = verification['verification_details']
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                icon = "✅" if details['name_verified'] else "❌"
                                st.metric("Name", icon)
                            with col2:
                                icon = "✅" if details['account_verified'] else "❌"
                                st.metric("Account", icon)
                            with col3:
                                icon = "✅" if details['amount_verified'] else "❌"
                                st.metric("Amount", icon)
                            with col4:
                                icon = "✅" if details['date_verified'] else "❌"
                                st.metric("Date", icon)
                        
                        st.success(f"✅ Claim {claim_id} submitted successfully!")
                        st.info("💡 You will receive an email notification once admin reviews your claim.")
                        
                        st.session_state.claim_submitted = True
    
    if st.session_state.get('claim_submitted', False):
        st.markdown("---")
        if st.button("📋 Return to Dashboard", type="primary", use_container_width=True):
            st.session_state.claim_submitted = False
            st.session_state.page = "faculty_dashboard"
            st.rerun()


def admin_dashboard_page():
    render_header()
    
    st.markdown("## 👨‍💼 Admin Dashboard")
    st.markdown("---")
    
    st.markdown("### 📊 Claims Overview")
    
    total = len(st.session_state.claims_data)
    approved = len([c for c in st.session_state.claims_data if c['status'] == 'APPROVED'])
    rejected = len([c for c in st.session_state.claims_data if c['status'] == 'REJECTED'])
    pending = len([c for c in st.session_state.claims_data if c['status'] == 'PENDING_ADMIN'])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📋 Total", total)
    with col2:
        st.metric("✅ Approved", approved)
    with col3:
        st.metric("❌ Rejected", rejected)
    with col4:
        st.metric("⏳ Pending", pending)
    
    st.markdown("---")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("### 📋 Claims Management")
    with col2:
        filter_status = st.selectbox("Filter", ["All", "PENDING_ADMIN", "APPROVED", "REJECTED"])
    
    if not st.session_state.claims_data:
        st.info("🔭 No claims submitted yet.")
    else:
        filtered = st.session_state.claims_data if filter_status == "All" else [
            c for c in st.session_state.claims_data if c['status'] == filter_status
        ]
        
        if not filtered:
            st.info(f"No {filter_status} claims found.")
        else:
            for idx, claim in enumerate(filtered):
                status_icons = {'APPROVED': '🟢', 'REJECTED': '🔴', 'PENDING_ADMIN': '🟡'}
                icon = status_icons.get(claim['status'], '⚪')
                
                with st.expander(f"{icon} **{claim['id']}** | {claim['faculty_name']} | ₹{claim['total_amount']:,.2f} | {claim['status']}"):
                    
                    st.markdown("#### 👤 Faculty Information")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Name:** {claim['faculty_name']}")
                        st.write(f"**Employee ID:** {claim['emp_id']}")
                        st.write(f"**Department:** {claim['department']}")
                    with col2:
                        st.write(f"**Email:** {claim['user_email']}")
                        st.write(f"**Account Number:** {claim.get('account_number', 'N/A')}")
                    
                    st.markdown("---")
                    
                    st.markdown("#### 📅 Claim Details")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Claim ID:** {claim['id']}")
                        st.write(f"**Event:** {claim['event_name']}")
                        st.write(f"**Event Date:** {claim['event_date']}")
                    with col2:
                        st.write(f"**Amount:** ₹{claim['total_amount']:,.2f}")
                        st.write(f"**Submitted:** {claim['submission_date']}")
                        st.write(f"**Status:** {claim['status']}")
                    
                    st.markdown("---")
                    
                    if 'processed_documents' in claim:
                        st.markdown("#### 📄 Uploaded Documents & Extracted Data")
                        docs = claim['processed_documents']
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("**📜 Certificates**")
                            proofs = docs.get('proofs', [])
                            for proof in proofs:
                                if proof and proof.get('success'):
                                    st.success(f"✅ {proof['filename']}")
                                    if proof.get('faculty_name'):
                                        st.write(f"  👤 Name: {proof['faculty_name']}")
                                    if proof.get('date'):
                                        st.write(f"  📅 Date: {proof['date']}")
                                    if proof.get('event_name'):
                                        st.write(f"  🎯 Event: {proof['event_name'][:30]}...")
                                elif proof:
                                    st.error(f"❌ {proof.get('filename', 'unknown')}")
                        
                        with col2:
                            st.markdown("**🧾 Invoices**")
                            invoices = docs.get('invoices', [])
                            if invoices:
                                for inv in invoices:
                                    if inv and inv.get('success'):
                                        st.success(f"✅ {inv['filename']}")
                                        if inv.get('total_amount'):
                                            st.write(f"  💰 Amount: {inv['total_amount']}")
                                        if inv.get('invoice_date'):
                                            st.write(f"  📅 Date: {inv['invoice_date']}")
                                    elif inv:
                                        st.error(f"❌ {inv.get('filename', 'unknown')}")
                            else:
                                st.info("No invoices")
                        
                        with col3:
                            st.markdown("**💳 Receipts**")
                            receipts = docs.get('receipts', [])
                            if receipts:
                                for rec in receipts:
                                    if rec and rec.get('success'):
                                        st.success(f"✅ {rec['filename']}")
                                        if rec.get('total_amount'):
                                            st.write(f"  💰 Amount: {rec['total_amount']}")
                                        if rec.get('account_number'):
                                            st.write(f"  🏦 Account: {rec['account_number']}")
                                        else:
                                            st.write(f"  🏦 Account: Not found")
                                        if rec.get('merchant_name'):
                                            st.write(f"  👤 Name: {rec['merchant_name']}")
                                        if rec.get('transaction_date'):
                                            st.write(f"  📅 Date: {rec['transaction_date']}")
                                    elif rec:
                                        st.error(f"❌ {rec.get('filename', 'unknown')}")
                            else:
                                st.info("No receipts")
                    
                    st.markdown("---")
                    
                    if 'verification_result' in claim:
                        st.markdown("#### 🤖 AI Verification Results")
                        verification = claim['verification_result']
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Status", verification['status'])
                        with col2:
                            st.metric("Total Issues", len(verification['all_issues']))
                        with col3:
                            st.metric("Critical", verification['high_severity_count'])
                        
                        st.info(f"**Message:** {verification['status_message']}")
                        
                        if 'verification_details' in verification:
                            st.markdown("**✅ Verification Checks:**")
                            details = verification['verification_details']
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                icon = "✅" if details['name_verified'] else "❌"
                                st.write(f"{icon} **Name Match**")
                            with col2:
                                icon = "✅" if details['account_verified'] else "❌"
                                st.write(f"{icon} **Account Match**")
                            with col3:
                                icon = "✅" if details['amount_verified'] else "❌"
                                st.write(f"{icon} **Amount Match**")
                            with col4:
                                icon = "✅" if details['date_verified'] else "❌"
                                st.write(f"{icon} **Date Match**")
                        
                        if verification['all_issues']:
                            st.markdown("**⚠️ Detected Issues:**")
                            for issue in verification['all_issues']:
                                severity_icon = "🔴" if issue['severity'] == 'high' else "🟡"
                                st.write(f"{severity_icon} **{issue['severity'].upper()}:** {issue['message']}")
                        
                        st.markdown("**📊 Extracted Data:**")
                        st.json(verification['extracted_data'])
                    
                    st.markdown("---")
                    
                    st.markdown("#### 👨‍💼 Admin Actions")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("✅ Approve Claim", key=f"approve_{idx}", use_container_width=True):
                            claim['status'] = 'APPROVED'
                            claim['admin_action'] = {
                                'action': 'Approved',
                                'by': st.session_state.user_email,
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            }
                            st.success(f"✅ Claim {claim['id']} APPROVED!")
                            st.rerun()
                    
                    with col2:
                        if st.button("❌ Reject Claim", key=f"reject_{idx}", use_container_width=True):
                            claim['status'] = 'REJECTED'
                            claim['admin_action'] = {
                                'action': 'Rejected',
                                'by': st.session_state.user_email,
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            }
                            st.warning(f"❌ Claim {claim['id']} REJECTED!")
                            st.rerun()
                    
                    with col3:
                        if st.button("⏸️ Mark Pending", key=f"pending_{idx}", use_container_width=True):
                            claim['status'] = 'PENDING_ADMIN'
                            st.info(f"⏳ Claim {claim['id']} marked as PENDING")
                            st.rerun()


# ============================================================================
# MAIN
# ============================================================================

def main():
    SessionState.initialize()
    apply_custom_css()
    
    if not AZURE_SDK_AVAILABLE:
        st.error("❌ Azure SDK not installed. Please run: `pip install azure-ai-documentintelligence`")
        st.stop()
    
    if not st.session_state.authenticated:
        login_page()
    else:
        if st.session_state.user_role == "admin":
            admin_dashboard_page()
        elif st.session_state.user_role == "faculty":
            if st.session_state.page == "claim_form":
                claim_form_page()
            else:
                faculty_dashboard_page()


if __name__ == "__main__":
    main()