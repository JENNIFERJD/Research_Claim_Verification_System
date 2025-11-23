"""
Claims Management System - FINAL WORKING VERSION
Fixed: API calls, data extraction, admin actions, navigation
"""

import pandas as pd
import streamlit as st
from datetime import datetime, date
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
    page_title="Claims Management System",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
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
            'customer_name': ['CustomerName', 'CustomerAddressRecipient'],
            'total_amount': ['InvoiceTotal', 'AmountDue', 'TotalAmount', 'Total'],
            'invoice_date': ['InvoiceDate', 'Date'],
        },
        'receipt': {
            'merchant_name': ['MerchantName'],
            'total_amount': ['Total'],
            'transaction_date': ['TransactionDate'],
        }
    }
    
    MAX_FILE_SIZE_MB = 50
    ALLOWED_FILE_TYPES = ['pdf', 'jpg', 'jpeg', 'png']
    MAX_CLAIM_AMOUNT = 100000
    AMOUNT_VARIANCE_TOLERANCE = 0.10
    MIN_CONFIDENCE_THRESHOLD = 0.6
    ADMIN_CREDENTIALS = {"admin@christ.edu": "admin123"}
    FACULTY_CREDENTIALS = {"faculty@christ.edu": "faculty123"}
    
    @staticmethod
    def get_azure_endpoint() -> Optional[str]:
        try:
            return st.secrets.get("AZURE_DOC_INTELLIGENCE_ENDPOINT")
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
# AZURE CLIENT - FIXED WITH PROPER API CALL
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
            
            if not endpoint.endswith('/'):
                endpoint = endpoint + '/'
            
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
        """Analyze document with working API call"""
        if not cls._client:
            success, msg = cls.initialize()
            if not success:
                raise RuntimeError(msg)
        
        try:
            logger.info(f"Analyzing {filename} with {model_id}")
            
            # WORKING METHOD: Direct bytes with proper parameters
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
        """Extract field value from Azure result - FIXED VERSION"""
        try:
            if not hasattr(result, 'documents') or not result.documents:
                logger.warning("No documents in result")
                return None
            
            document = result.documents[0]
            if not hasattr(document, 'fields') or not document.fields:
                logger.warning("No fields in document")
                return None
            
            # Log available fields for debugging
            available_fields = list(document.fields.keys())
            logger.info(f"Available fields: {available_fields}")
            
            for field_name in field_names:
                if field_name in document.fields:
                    field = document.fields[field_name]
                    
                    # Check if field is None or has no value
                    if field is None:
                        logger.warning(f"Field '{field_name}' exists but is None")
                        continue
                    
                    # Method 1: Try direct value access
                    if hasattr(field, 'value'):
                        value = field.value
                        if value is not None:
                            # Handle complex value types
                            if hasattr(value, 'content'):
                                extracted = str(value.content)
                            else:
                                extracted = str(value)
                            
                            logger.info(f"✅ Extracted {field_name}: {extracted}")
                            return extracted
                    
                    # Method 2: Try content attribute directly
                    if hasattr(field, 'content'):
                        extracted = str(field.content)
                        logger.info(f"✅ Extracted {field_name} from content: {extracted}")
                        return extracted
                    
                    # Method 3: Try value_string for backward compatibility
                    if hasattr(field, 'value_string'):
                        extracted = str(field.value_string)
                        logger.info(f"✅ Extracted {field_name} from value_string: {extracted}")
                        return extracted
                    
                    logger.warning(f"Field '{field_name}' found but no extractable value")
            
            logger.warning(f"None of {field_names} could be extracted from fields: {available_fields}")
            return None
            
        except Exception as e:
            logger.error(f"Field extraction error: {e}", exc_info=True)
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
            
            # Log what was extracted
            logger.info(f"Certificate extraction result:")
            logger.info(f"  - faculty_name: {extracted.get('faculty_name')}")
            logger.info(f"  - event_name: {extracted.get('event_name')}")
            logger.info(f"  - date: {extracted.get('date')}")
            logger.info(f"  - organization: {extracted.get('organization_name')}")
            
            return extracted
        except Exception as e:
            logger.error(f"Certificate processing failed: {e}")
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
            
            logger.info(f"Invoice extraction result: {extracted}")
            return extracted
        except Exception as e:
            logger.error(f"Invoice processing failed: {e}")
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
            }
            
            logger.info(f"Receipt extraction result: {extracted}")
            return extracted
        except Exception as e:
            logger.error(f"Receipt processing failed: {e}")
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
    def normalize_text(text: str) -> str:
        if not text:
            return ""
        text = str(text).lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text
    
    @staticmethod
    def extract_amount(value: Any) -> Optional[float]:
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
        issues = []
        high_severity_issues = []
        
        proof_data = documents.get('proofs', [])
        invoice_data = documents.get('invoices', [])
        receipt_data = documents.get('receipts', [])
        
        # Check processing failures
        failed_docs = []
        for proof in proof_data:
            if not proof or not proof.get('success', False):
                failed_docs.append(f"Certificate: {proof.get('filename', 'unknown')}")
        for invoice in invoice_data:
            if not invoice or not invoice.get('success', False):
                failed_docs.append(f"Invoice: {invoice.get('filename', 'unknown')}")
        for receipt in receipt_data:
            if not receipt or not receipt.get('success', False):
                failed_docs.append(f"Receipt: {receipt.get('filename', 'unknown')}")
        
        if failed_docs:
            issues.append({
                'severity': 'high',
                'message': f"❌ Document processing failed: {', '.join(failed_docs)}"
            })
            high_severity_issues.append("Processing failed")
        
        extracted_any_data = False
        
        # Verify names
        form_name = VerificationEngine.normalize_text(form_data.get('faculty_name', ''))
        extracted_names = []
        
        for proof in proof_data:
            if proof and proof.get('success') and proof.get('faculty_name'):
                extracted_any_data = True
                name = VerificationEngine.normalize_text(proof['faculty_name'])
                extracted_names.append(name)
                
                if name and form_name:
                    if name != form_name and name not in form_name and form_name not in name:
                        issues.append({
                            'severity': 'high',
                            'message': f"❌ Name mismatch: Form='{form_data.get('faculty_name')}' vs Certificate='{proof['faculty_name']}'"
                        })
                        high_severity_issues.append("Name mismatch")
        
        # Verify amounts
        form_amount = form_data.get('total_amount', 0)
        invoice_amounts = []
        receipt_amounts = []
        
        for invoice in invoice_data:
            if invoice and invoice.get('success') and invoice.get('total_amount'):
                extracted_any_data = True
                amount = VerificationEngine.extract_amount(invoice['total_amount'])
                if amount and amount > 0:
                    invoice_amounts.append(amount)
        
        for receipt in receipt_data:
            if receipt and receipt.get('success') and receipt.get('total_amount'):
                extracted_any_data = True
                amount = VerificationEngine.extract_amount(receipt['total_amount'])
                if amount and amount > 0:
                    receipt_amounts.append(amount)
        
        all_amounts = invoice_amounts + receipt_amounts
        
        if form_amount > 0 and all_amounts:
            matched = False
            for extracted_amt in all_amounts:
                variance = abs(form_amount - extracted_amt) / max(form_amount, extracted_amt)
                if variance <= Config.AMOUNT_VARIANCE_TOLERANCE:
                    matched = True
                    break
            
            if not matched:
                issues.append({
                    'severity': 'high',
                    'message': f"❌ Amount mismatch: Form=₹{form_amount:.2f} vs Extracted={all_amounts}"
                })
                high_severity_issues.append("Amount mismatch")
        
        # Determine status
        if len(high_severity_issues) == 0 and extracted_any_data:
            status = 'APPROVED'
            status_message = '✅ All verifications passed - Claim APPROVED'
        elif len(high_severity_issues) > 0:
            status = 'PENDING_ADMIN'
            status_message = f'⚠️ {len(high_severity_issues)} issues detected - Requires Admin Review'
        else:
            status = 'PENDING_ADMIN'
            status_message = '⚠️ No data extracted - Requires Admin Review'
            
        return {
            'status': status,
            'status_message': status_message,
            'is_approved': status == 'APPROVED',
            'is_rejected': status == 'REJECTED',
            'all_issues': issues,
            'high_severity_count': len(high_severity_issues),
            'extracted_data': {
                'names': {'form': form_data.get('faculty_name'), 'extracted': extracted_names},
                'amounts': {'form': form_amount, 'invoices': invoice_amounts, 'receipts': receipt_amounts}
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
# PROCESSING
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
                st.info(f"📄 Processing: {file.name}")
                
                file_bytes = file.read()
                file.seek(0)
                
                result = DocumentProcessor.process_proof_document(file_bytes, file.name)
                results['proofs'].append(result)
                
                if result.get('success'):
                    st.success(f"✅ {file.name}")
                else:
                    st.error(f"❌ {file.name}: {result.get('error', 'Unknown error')}")
        
        if invoice_files:
            for file in invoice_files:
                current += 1
                progress.progress(current / total_files)
                st.info(f"📄 Processing: {file.name}")
                
                file_bytes = file.read()
                file.seek(0)
                
                result = DocumentProcessor.process_invoice_document(file_bytes, file.name)
                results['invoices'].append(result)
                
                if result.get('success'):
                    st.success(f"✅ {file.name}")
                else:
                    st.error(f"❌ {file.name}: {result.get('error', 'Unknown error')}")
        
        if receipt_files:
            for file in receipt_files:
                current += 1
                progress.progress(current / total_files)
                st.info(f"📄 Processing: {file.name}")
                
                file_bytes = file.read()
                file.seek(0)
                
                result = DocumentProcessor.process_receipt_document(file_bytes, file.name)
                results['receipts'].append(result)
                
                if result.get('success'):
                    st.success(f"✅ {file.name}")
                else:
                    st.error(f"❌ {file.name}: {result.get('error', 'Unknown error')}")
        
        progress.progress(1.0)
        return results
    finally:
        progress.empty()


def generate_claim_id() -> str:
    year = datetime.now().year
    random = str(uuid.uuid4())[:6].upper()
    return f"RCL-{year}-{random}"


# ============================================================================
# STYLING
# ============================================================================

def apply_custom_css():
    st.markdown("""
    <style>
        .approved-box {
            background: #d1fae5; border-left: 4px solid #10b981;
            padding: 20px; margin: 10px 0; border-radius: 5px;
        }
        .rejected-box {
            background: #fee2e2; border-left: 4px solid #ef4444;
            padding: 20px; margin: 10px 0; border-radius: 5px;
        }
        .pending-box {
            background: #fef3c7; border-left: 4px solid #f59e0b;
            padding: 20px; margin: 10px 0; border-radius: 5px;
        }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# UI PAGES
# ============================================================================

def login_page():
    st.markdown("# 🏛️ Claims Management System")
    st.markdown("### CHRIST University")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)
            
            if submitted:
                if AuthManager.authenticate(email, password):
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials")
        
        with st.expander("Demo Credentials"):
            st.info("**Admin:** admin@christ.edu / admin123\n\n**Faculty:** faculty@christ.edu / faculty123")


def faculty_dashboard_page():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"# Welcome, {st.session_state.user_name}!")
    with col2:
        if st.button("Logout"):
            AuthManager.logout()
    
    if st.button("📝 Submit New Claim", type="primary", use_container_width=True):
        st.session_state.page = "claim_form"
        st.rerun()
    
    st.markdown("---")
    st.markdown("### My Claims")
    
    my_claims = [c for c in st.session_state.claims_data if c['user_email'] == st.session_state.user_email]
    
    if not my_claims:
        st.info("No claims submitted yet")
    else:
        for claim in my_claims:
            status_icon = {'APPROVED': '🟢', 'REJECTED': '🔴', 'PENDING_ADMIN': '🟡'}.get(claim['status'], '⚪')
            
            with st.expander(f"{status_icon} {claim['id']} - {claim['status']} - ₹{claim['total_amount']:,.2f}"):
                st.write(f"**Event:** {claim['event_name']}")
                st.write(f"**Submitted:** {claim['submission_date']}")
                if 'verification_result' in claim:
                    st.write(f"**Message:** {claim['verification_result']['status_message']}")


def claim_form_page():
    if st.session_state.get('return_to_dashboard', False):
        st.session_state.return_to_dashboard = False
        st.session_state.page = "faculty_dashboard"
        st.rerun()
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("## Submit Reimbursement Claim")
    with col2:
        if st.button("← Back"):
            st.session_state.page = "faculty_dashboard"
            st.rerun()
    
    with st.form("claim_form"):
        st.markdown("### Personal Details")
        col1, col2 = st.columns(2)
        with col1:
            faculty_name = st.text_input("Faculty Name *")
        with col2:
            emp_id = st.text_input("Employee ID *")
        
        department = st.text_input("Department *")
        
        st.markdown("### Event Details")
        event_name = st.text_area("Event/Conference Name *", height=80)
        event_date = st.date_input("Event Date")
        
        st.markdown("### Financial Details")
        col1, col2 = st.columns(2)
        with col1:
            registration_fee = st.number_input("Registration Fee (₹)", min_value=0.0, format="%.2f")
        with col2:
            travel_amount = st.number_input("Travel Amount (₹)", min_value=0.0, format="%.2f")
        
        total_amount = registration_fee + travel_amount
        if total_amount > 0:
            st.markdown(f"**Total: ₹{total_amount:,.2f}**")
        
        st.markdown("---")
        st.markdown("### 📤 Upload Documents")
        
        st.warning("""
        **REQUIRED:**
        - ✅ At least 1 Certificate
        - ✅ Either Invoice OR Receipt
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**📜 Certificates***")
            proof_files = st.file_uploader(
                "Required",
                accept_multiple_files=True,
                type=Config.ALLOWED_FILE_TYPES,
                key="proofs"
            )
            if proof_files:
                st.success(f"✅ {len(proof_files)} uploaded")
        
        with col2:
            st.markdown("**🧾 Invoices**")
            invoice_files = st.file_uploader(
                "Optional",
                accept_multiple_files=True,
                type=Config.ALLOWED_FILE_TYPES,
                key="invoices"
            )
            if invoice_files:
                st.success(f"✅ {len(invoice_files)} uploaded")
        
        with col3:
            st.markdown("**💳 Receipts**")
            receipt_files = st.file_uploader(
                "Optional",
                accept_multiple_files=True,
                type=Config.ALLOWED_FILE_TYPES,
                key="receipts"
            )
            if receipt_files:
                st.success(f"✅ {len(receipt_files)} uploaded")
        
        has_certs = proof_files and len(proof_files) > 0
        has_financial = (invoice_files and len(invoice_files) > 0) or (receipt_files and len(receipt_files) > 0)
        
        if not has_certs or not has_financial:
            st.error("⚠️ Cannot submit: Missing required documents")
        
        submitted = st.form_submit_button("🚀 Submit Claim", type="primary", use_container_width=True)
        
        if submitted:
            errors = []
            
            if not faculty_name or len(faculty_name.strip()) < 2:
                errors.append("❌ Faculty name required")
            if not emp_id:
                errors.append("❌ Employee ID required")
            if not department:
                errors.append("❌ Department required")
            if not event_name:
                errors.append("❌ Event name required")
            if total_amount <= 0:
                errors.append("❌ Total amount must be > 0")
            if not proof_files or len(proof_files) == 0:
                errors.append("❌ At least 1 certificate required")
            if not has_financial:
                errors.append("❌ Either invoice OR receipt required")
            
            if errors:
                st.error("### ❌ SUBMISSION BLOCKED")
                for error in errors:
                    st.error(error)
                st.stop()
            
            else:
                claim_id = generate_claim_id()
                st.info(f"📋 Processing {claim_id}...")
                
                st.markdown("### 🤖 AI Processing")
                processed_docs = process_all_documents(proof_files, invoice_files, receipt_files)
                
                if processed_docs:
                    st.markdown("### 🔍 Verification")
                    
                    form_data = {
                        'faculty_name': faculty_name,
                        'event_name': event_name,
                        'total_amount': total_amount
                    }
                    
                    verification = VerificationEngine.perform_full_verification(form_data, processed_docs)
                    
                    if verification['status'] == 'APPROVED':
                        st.markdown(f"""
                        <div class="approved-box">
                            <h2>✅ CLAIM APPROVED</h2>
                            <p>{verification['status_message']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.balloons()
                    elif verification['status'] == 'REJECTED':
                        st.markdown(f"""
                        <div class="rejected-box">
                            <h2>❌ CLAIM REJECTED</h2>
                            <p>{verification['status_message']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="pending-box">
                            <h2>⏳ PENDING REVIEW</h2>
                            <p>{verification['status_message']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if verification['all_issues']:
                        with st.expander(f"⚠️ Issues ({len(verification['all_issues'])})"):
                            for issue in verification['all_issues']:
                                icon = "🔴" if issue['severity'] == 'high' else "🟡"
                                st.write(f"{icon} {issue['message']}")
                    
                    with st.expander("📊 Extracted Data"):
                        st.json(verification['extracted_data'])
                    
                    # Save claim
                    new_claim = {
                        'id': claim_id,
                        'user_email': st.session_state.user_email,
                        'faculty_name': faculty_name,
                        'emp_id': emp_id,
                        'department': department,
                        'event_name': event_name,
                        'event_date': event_date.strftime('%Y-%m-%d'),
                        'total_amount': total_amount,
                        'status': verification['status'],
                        'submission_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'verification_result': verification,
                        'processed_documents': processed_docs,
                    }
                    
                    st.session_state.claims_data.append(new_claim)
                    st.success(f"✅ Claim {claim_id} submitted!")
                    st.info("💡 You'll receive email notification after admin review")
                    
                    st.session_state.claim_submitted = True
    
    if st.session_state.get('claim_submitted', False):
        st.markdown("---")
        if st.button("Return to Dashboard", type="primary", use_container_width=True):
            st.session_state.claim_submitted = False
            st.session_state.page = "faculty_dashboard"
            st.rerun()


def admin_dashboard_page():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("# Admin Dashboard")
    with col2:
        if st.button("Logout"):
            AuthManager.logout()
    
    st.markdown("### 📊 Overview")
    
    total = len(st.session_state.claims_data)
    approved = len([c for c in st.session_state.claims_data if c['status'] == 'APPROVED'])
    rejected = len([c for c in st.session_state.claims_data if c['status'] == 'REJECTED'])
    pending = len([c for c in st.session_state.claims_data if c['status'] == 'PENDING_ADMIN'])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total", total)
    with col2:
        st.metric("✅ Approved", approved)
    with col3:
        st.metric("❌ Rejected", rejected)
    with col4:
        st.metric("⏳ Pending", pending)
    
    st.markdown("---")
    
    filter_status = st.selectbox("Filter", ["All", "APPROVED", "REJECTED", "PENDING_ADMIN"])
    
    st.markdown("### 📋 Claims")
    
    if not st.session_state.claims_data:
        st.info("No claims submitted yet")
    else:
        filtered = st.session_state.claims_data if filter_status == "All" else [
            c for c in st.session_state.claims_data if c['status'] == filter_status
        ]
        
        if not filtered:
            st.info(f"No {filter_status} claims")
        else:
            for idx, claim in enumerate(filtered):
                status_icon = {
                    'APPROVED': '🟢',
                    'REJECTED': '🔴',
                    'PENDING_ADMIN': '🟡'
                }.get(claim['status'], '⚪')
                
                with st.expander(f"{status_icon} {claim['id']} | {claim['faculty_name']} | ₹{claim['total_amount']:,.2f}"):
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 👤 Faculty")
                        st.write(f"**Name:** {claim['faculty_name']}")
                        st.write(f"**ID:** {claim['emp_id']}")
                        st.write(f"**Dept:** {claim['department']}")
                        st.write(f"**Email:** {claim['user_email']}")
                    
                    with col2:
                        st.markdown("#### 📅 Claim")
                        st.write(f"**Event:** {claim['event_name']}")
                        st.write(f"**Date:** {claim['event_date']}")
                        st.write(f"**Amount:** ₹{claim['total_amount']:,.2f}")
                        st.write(f"**Submitted:** {claim['submission_date']}")
                    
                    if 'processed_documents' in claim:
                        st.markdown("#### 📄 Documents")
                        docs = claim['processed_documents']
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            proofs = docs.get('proofs', [])
                            success_count = len([p for p in proofs if p and p.get('success')])
                            st.write(f"Certificates: {success_count}/{len(proofs)}")
                            
                            for proof in proofs:
                                if proof:
                                    if proof.get('success'):
                                        st.success(f"✅ {proof.get('filename', 'unknown')}")
                                        if proof.get('faculty_name'):
                                            st.write(f"  • Name: {proof['faculty_name']}")
                                        if proof.get('event_name'):
                                            st.write(f"  • Event: {proof['event_name'][:30]}...")
                                    else:
                                        st.error(f"❌ {proof.get('filename', 'unknown')}")
                        
                        with col2:
                            invoices = docs.get('invoices', [])
                            success_count = len([i for i in invoices if i and i.get('success')])
                            st.write(f"Invoices: {success_count}/{len(invoices)}")
                            
                            for invoice in invoices:
                                if invoice:
                                    if invoice.get('success'):
                                        st.success(f"✅ {invoice.get('filename', 'unknown')}")
                                        if invoice.get('total_amount'):
                                            st.write(f"  • Amount: {invoice['total_amount']}")
                                    else:
                                        st.error(f"❌ {invoice.get('filename', 'unknown')}")
                        
                        with col3:
                            receipts = docs.get('receipts', [])
                            success_count = len([r for r in receipts if r and r.get('success')])
                            st.write(f"Receipts: {success_count}/{len(receipts)}")
                            
                            for receipt in receipts:
                                if receipt:
                                    if receipt.get('success'):
                                        st.success(f"✅ {receipt.get('filename', 'unknown')}")
                                        if receipt.get('total_amount'):
                                            st.write(f"  • Amount: {receipt['total_amount']}")
                                    else:
                                        st.error(f"❌ {receipt.get('filename', 'unknown')}")
                    
                    if 'verification_result' in claim:
                        st.markdown("---")
                        st.markdown("#### 🤖 AI Verification")
                        
                        verification = claim['verification_result']
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Status", verification['status'])
                        with col2:
                            st.metric("Issues", verification['high_severity_count'])
                        
                        st.info(verification['status_message'])
                        
                        if verification['all_issues']:
                            st.markdown("**Issues:**")
                            for issue in verification['all_issues']:
                                icon = "🔴" if issue['severity'] == 'high' else "🟡"
                                st.write(f"{icon} {issue['message']}")
                            
                            st.markdown("##### 📊 Extracted Data")
                            st.json(verification['extracted_data'])
                    
                    st.markdown("---")
                    st.markdown("#### 👨‍💼 Actions")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("✅ Approve", key=f"approve_{idx}", use_container_width=True):
                            claim['status'] = 'APPROVED'
                            claim['admin_action'] = {
                                'action': 'Approved',
                                'by': st.session_state.user_email,
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            }
                            st.success(f"✅ Approved {claim['id']}")
                            st.rerun()
                    
                    with col2:
                        if st.button("❌ Reject", key=f"reject_{idx}", use_container_width=True):
                            claim['status'] = 'REJECTED'
                            claim['admin_action'] = {
                                'action': 'Rejected',
                                'by': st.session_state.user_email,
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            }
                            st.warning(f"❌ Rejected {claim['id']}")
                            st.rerun()
                    
                    with col3:
                        if st.button("⏸️ Pending", key=f"pending_{idx}", use_container_width=True):
                            claim['status'] = 'PENDING_ADMIN'
                            st.info("Marked as pending")
                            st.rerun()


# ============================================================================
# MAIN
# ============================================================================

def main():
    SessionState.initialize()
    apply_custom_css()
    
    if not AZURE_SDK_AVAILABLE:
        st.error("❌ Azure SDK not installed. Run: pip install azure-ai-documentintelligence")
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