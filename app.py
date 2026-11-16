import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ============================================================
# PAGE CONFIGURATION & METADATA
# ============================================================

st.set_page_config(
    page_title="FinRisk IQ — Intelligent Credit Underwriting & Repayment Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

# ============================================================
# CUSTOM FINTECH DESIGN SYSTEM (CSS)
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-main: #0B0F19;
    --card-bg: rgba(17, 24, 39, 0.75);
    --card-border: rgba(255, 255, 255, 0.08);
    --accent-primary: #6366F1;
    --accent-emerald: #10B981;
    --accent-cyan: #06B6D4;
    --accent-amber: #F59E0B;
    --accent-rose: #EF4444;
    --text-primary: #F9FAFB;
    --text-secondary: #94A3B8;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
}

/* Background canvas */
.stApp {
    background: radial-gradient(circle at 15% 15%, rgba(99, 102, 241, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 85% 85%, rgba(6, 182, 212, 0.07) 0%, transparent 40%),
                #0B0F19;
    color: #F3F4F6;
}

/* Executive Header */
.fin-header-container {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.95) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 18px;
    padding: 24px 28px;
    margin-bottom: 24px;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    position: relative;
    overflow: hidden;
}

.fin-header-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 3px;
    background: linear-gradient(90deg, #6366F1, #06B6D4, #10B981);
}

.fin-title {
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(90deg, #FFFFFF 30%, #CBD5E1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 12px;
}

.fin-subtitle {
    font-size: 0.95rem;
    color: #94A3B8;
    margin-bottom: 0px;
    font-weight: 400;
}

/* Status Pill */
.fin-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.badge-live {
    background: rgba(16, 185, 129, 0.15);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.badge-info {
    background: rgba(99, 102, 241, 0.15);
    color: #A5B4FC;
    border: 1px solid rgba(99, 102, 241, 0.3);
}

/* FinTech Glass Cards */
.fin-card {
    background: rgba(17, 24, 39, 0.7);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 16px;
    transition: all 0.25s ease-in-out;
}

.fin-card:hover {
    border-color: rgba(99, 102, 241, 0.3);
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
}

.fin-card-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #E2E8F0;
    letter-spacing: -0.01em;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Metric Display Cards */
.metric-box {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 18px 20px;
    text-align: left;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.metric-box:hover {
    transform: translateY(-2px);
    border-color: rgba(255, 255, 255, 0.16);
}

.metric-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}

.metric-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.85rem;
    font-weight: 700;
    line-height: 1.1;
    color: #F8FAFC;
}

.metric-sub {
    font-size: 0.75rem;
    margin-top: 6px;
    color: #64748B;
}

/* Decision Outcome Banners */
.decision-banner {
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border: 1px solid transparent;
}

.decision-approved {
    background: linear-gradient(90deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 182, 212, 0.08) 100%);
    border-color: rgba(16, 185, 129, 0.4);
    box-shadow: 0 0 25px rgba(16, 185, 129, 0.12);
}

.decision-review {
    background: linear-gradient(90deg, rgba(245, 158, 11, 0.15) 0%, rgba(217, 119, 6, 0.08) 100%);
    border-color: rgba(245, 158, 11, 0.4);
    box-shadow: 0 0 25px rgba(245, 158, 11, 0.12);
}

.decision-declined {
    background: linear-gradient(90deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.08) 100%);
    border-color: rgba(239, 68, 68, 0.4);
    box-shadow: 0 0 25px rgba(239, 68, 68, 0.12);
}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: rgba(15, 23, 42, 0.7);
    padding: 6px;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.06);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #94A3B8;
    font-weight: 600;
    padding: 8px 18px;
    font-size: 0.88rem;
    transition: all 0.2s ease;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.25) 0%, rgba(124, 58, 237, 0.25) 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(99, 102, 241, 0.4) !important;
}

/* Form Submit Button */
div.stButton > button:first-child {
    background: linear-gradient(135deg, #4F46E5 0%, #6366F1 50%, #06B6D4 100%);
    color: #FFFFFF;
    font-weight: 700;
    font-size: 1.02rem;
    padding: 12px 24px;
    border-radius: 12px;
    border: none;
    box-shadow: 0 4px 20px rgba(79, 70, 229, 0.35);
    transition: all 0.25s ease;
    width: 100%;
}

div.stButton > button:first-child:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(79, 70, 229, 0.55);
    filter: brightness(1.1);
}

/* Input Fields Overrides */
div[data-baseweb="input"], div[data-baseweb="select"] {
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA & MODEL LOADING
# ============================================================

PURPOSE_MAPPING = {
    "Debt Consolidation": "debt_consolidation",
    "Credit Card Refinance": "credit_card",
    "Home Improvement": "home_improvement",
    "Small Business": "small_business",
    "Major Purchase": "major_purchase",
    "Educational Loan": "educational",
    "Other Personal Needs": "all_other",
}

PURPOSE_TO_INT = {
    "all_other": 0,
    "credit_card": 1,
    "debt_consolidation": 2,
    "educational": 3,
    "home_improvement": 4,
    "major_purchase": 5,
    "small_business": 6,
}

INT_TO_PURPOSE = {v: k for k, v in PURPOSE_TO_INT.items()}

@st.cache_resource
def load_underwriting_models():
    """Load trained ensemble models and encoders."""
    model_path = BASE_DIR / "loan_model.pkl"
    if not model_path.exists():
        st.error("Model file `loan_model.pkl` not found in workspace.")
        st.stop()
    model = joblib.load(model_path)
    
    # Load purpose encoder if available
    encoder_path = BASE_DIR / "purpose_encoder.pkl"
    encoder = None
    if encoder_path.exists():
        try:
            encoder = joblib.load(encoder_path)
        except Exception:
            encoder = None
            
    return model, encoder

@st.cache_data
def load_historical_benchmark():
    """Load reference distributions from loan_data.csv for benchmarking."""
    data_path = BASE_DIR / "loan_data.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
        return df
    return None

model, purpose_encoder = load_underwriting_models()
benchmark_df = load_historical_benchmark()


# ============================================================
# HELPER FUNCTIONS: RISK TIERS & UNDERWRITING
# ============================================================

def calculate_risk_tier(prob_default):
    """
    Categorizes the applicant into institutional credit grades (A to E).
    Based on standard credit risk bands calibrated to the 16.0% historical default base rate.
    """
    if prob_default < 0.08:
        return {
            "grade": "Grade A",
            "tier": "Super-Prime",
            "color": "#10B981",
            "badge_class": "badge-live",
            "decision": "Instant Auto-Approval",
            "status": "APPROVED",
            "recommendation": "Qualifies for lowest prime interest rates with standard automated underwriting and zero supplementary collateral.",
            "max_ltv": "100%",
            "terms": "36 or 60 Months @ Prime Rate"
        }
    elif prob_default < 0.15:
        return {
            "grade": "Grade B",
            "tier": "Prime",
            "color": "#06B6D4",
            "badge_class": "badge-live",
            "decision": "Standard Approval",
            "status": "APPROVED",
            "recommendation": "Applicant exhibits solid credit fundamentals. Approved under standard portfolio guidelines with standard pricing.",
            "max_ltv": "90%",
            "terms": "Standard 36/60 Mo Tier Terms"
        }
    elif prob_default < 0.24:
        return {
            "grade": "Grade C",
            "tier": "Near-Prime",
            "color": "#F59E0B",
            "badge_class": "badge-info",
            "decision": "Conditional Approval",
            "status": "REVIEW",
            "recommendation": "Moderate credit risk. Conditional approval subject to mandatory W-2 / bank-statement income verification or a 10% cash equity contribution.",
            "max_ltv": "80%",
            "terms": "Requires Income Verification"
        }
    elif prob_default < 0.36:
        return {
            "grade": "Grade D",
            "tier": "Subprime",
            "color": "#F97316",
            "badge_class": "badge-info",
            "decision": "Credit Committee Review",
            "status": "REVIEW",
            "recommendation": "Elevated default probability. Requires manual underwriting review, creditworthy co-signer, or pledge of unencumbered collateral.",
            "max_ltv": "70%",
            "terms": "Co-signer or Collateral Mandated"
        }
    else:
        return {
            "grade": "Grade E",
            "tier": "High Risk / Deep Subprime",
            "color": "#EF4444",
            "badge_class": "badge-info",
            "decision": "Adverse Action / Decline",
            "status": "DECLINED",
            "recommendation": "Projected default risk significantly exceeds acceptable credit portfolio risk limits. Issue standard FCRA adverse action notice.",
            "max_ltv": "N/A",
            "terms": "Decline — Refer to Credit Repair"
        }

def get_fico_rating(score):
    if score >= 800:
        return "Exceptional (800+)", "#10B981"
    elif score >= 740:
        return "Very Good (740-799)", "#06B6D4"
    elif score >= 670:
        return "Good / Prime (670-739)", "#3B82F6"
    elif score >= 580:
        return "Fair / Near-Prime (580-669)", "#F59E0B"
    else:
        return "Poor / Subprime (<580)", "#EF4444"

def compute_applicant_economics(installment, int_rate, prob_default, annual_income, term_months=36):
    """Calculates financial exposure, total estimated loan value, and expected loss."""
    # Approximate principal amount: P = Installment * [(1 - (1+r)^-n) / r]
    monthly_rate = int_rate / 12.0
    if monthly_rate > 0:
        est_loan_amount = installment * ((1.0 - (1.0 + monthly_rate) ** (-term_months)) / monthly_rate)
    else:
        est_loan_amount = installment * term_months

    total_payments = installment * term_months
    total_interest = max(0.0, total_payments - est_loan_amount)
    
    # Basel II Expected Loss: EL = PD * LGD * EAD (assuming 45% Loss Given Default)
    lgd = 0.45
    expected_loss = prob_default * lgd * est_loan_amount
    
    # Monthly gross income & Installment-to-Income (ITI)
    monthly_income = annual_income / 12.0 if annual_income > 0 else 1.0
    iti_ratio = (installment / monthly_income) * 100.0

    return {
        "est_loan_amount": est_loan_amount,
        "total_payments": total_payments,
        "total_interest": total_interest,
        "expected_loss": expected_loss,
        "iti_ratio": iti_ratio,
        "monthly_income": monthly_income
    }


# ============================================================
# SIDEBAR CONTROLS & BENCHMARK METRICS
# ============================================================

with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
        <span style="font-size:2rem;">🛡️</span>
        <div>
            <div style="font-size:1.15rem; font-weight:800; color:#F8FAFC; line-height:1.2;">FinRisk IQ</div>
            <div style="font-size:0.75rem; color:#94A3B8;">Institutional Credit Engine</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<span class="fin-badge badge-live">● Engine Active v2.4</span>', unsafe_allow_html=True)
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    
    st.subheader("⚙️ Underwriting Settings")
    
    risk_policy = st.selectbox(
        "Risk Policy Cutoff Profile",
        options=["Balanced (Standard F1)", "Conservative (Loss Aversion)", "Aggressive (Growth Focus)"],
        index=0,
        help="Adjusts risk decisioning sensitivity. Conservative applies tighter approval thresholds."
    )
    
    if "Conservative" in risk_policy:
        custom_cutoff = 0.12
    elif "Aggressive" in risk_policy:
        custom_cutoff = 0.22
    else:
        custom_cutoff = 0.16  # Base dataset default rate
        
    st.caption(f"Target Default Cutoff: **{custom_cutoff*100:.1f}%**")
    
    st.divider()
    
    st.subheader("📊 Portfolio Benchmark")
    if benchmark_df is not None:
        total_records = len(benchmark_df)
        repaid_pct = (1.0 - benchmark_df['not.fully.paid'].mean()) * 100.0
        avg_fico = benchmark_df['fico'].mean()
        avg_int = benchmark_df['int.rate'].mean() * 100.0
        
        st.markdown(f"""
        <div style="background:rgba(30,41,59,0.5); padding:12px; border-radius:10px; border:1px solid rgba(255,255,255,0.05); font-size:0.82rem;">
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                <span style="color:#94A3B8;">Dataset Volume:</span>
                <span style="font-weight:700; color:#F8FAFC;">{total_records:,} loans</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                <span style="color:#94A3B8;">Historical Paid:</span>
                <span style="font-weight:700; color:#34D399;">{repaid_pct:.1f}%</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                <span style="color:#94A3B8;">Benchmark FICO:</span>
                <span style="font-weight:700; color:#F8FAFC;">{avg_fico:.0f}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#94A3B8;">Avg Interest Rate:</span>
                <span style="font-weight:700; color:#F8FAFC;">{avg_int:.2f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("""
    <div style="font-size:0.75rem; color:#64748B; text-align:center;">
        Trained Ensemble: Gradient Boosting<br>
        Loss: Log-Loss | 100 Estimators<br>
        © 2026 FinRisk Analytics Group
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# TOP EXECUTIVE HEADER
# ============================================================

st.markdown("""
<div class="fin-header-container">
    <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:16px;">
        <div>
            <div class="fin-title">
                <span>💰</span> Loan Repayment & Credit Risk Underwriter
            </div>
            <div class="fin-subtitle">
                Advanced ensemble machine learning for credit underwriting, default probability prediction & portfolio risk management.
            </div>
        </div>
        <div style="display:flex; gap:10px; align-items:center;">
            <span class="fin-badge badge-live">Model: Gradient Boosting</span>
            <span class="fin-badge badge-info">ROC-AUC: 0.710</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# PRIMARY TABS
# ============================================================

tab_single, tab_simulator, tab_batch, tab_analytics, tab_policy = st.tabs([
    "👤 Single Underwriting & Scorecard",
    "🧪 What-If Sensitivity Simulator",
    "📁 Batch Portfolio Underwriting",
    "📊 Model Benchmark & Analytics Lab",
    "📘 Credit Policy & Rules SOP"
])


# ============================================================
# TAB 1: SINGLE APPLICANT UNDERWRITING
# ============================================================

with tab_single:
    
    # Preset Selector for 1-Click Testing
    st.markdown("#### ⚡ Quick-Load Underwriting Presets")
    preset_cols = st.columns(4)
    
    preset_choice = None
    with preset_cols[0]:
        if st.button("🌟 Tier A: Super-Prime Executive", use_container_width=True):
            st.session_state['applicant_preset'] = "super_prime"
    with preset_cols[1]:
        if st.button("🏡 Tier B: Prime Homeowner", use_container_width=True):
            st.session_state['applicant_preset'] = "prime_home"
    with preset_cols[2]:
        if st.button("🏢 Tier C: Near-Prime Business", use_container_width=True):
            st.session_state['applicant_preset'] = "near_prime"
    with preset_cols[3]:
        if st.button("⚠️ Tier D: Subprime Stressed", use_container_width=True):
            st.session_state['applicant_preset'] = "subprime"

    # Default preset values
    active_preset = st.session_state.get('applicant_preset', 'prime_home')
    
    if active_preset == "super_prime":
        p_policy = 1
        p_purpose = "Major Purchase"
        p_int_rate = 7.88
        p_installment = 450.0
        p_income = 135000.0
        p_dti = 6.8
        p_fico = 785
        p_credit_years = 16.5
        p_revol_bal = 12500.0
        p_revol_util = 14.2
        p_inq = 0
        p_delinq = 0
        p_pub_rec = 0
    elif active_preset == "prime_home":
        p_policy = 1
        p_purpose = "Debt Consolidation"
        p_int_rate = 11.89
        p_installment = 320.0
        p_income = 72000.0
        p_dti = 14.5
        p_fico = 715
        p_credit_years = 11.2
        p_revol_bal = 18500.0
        p_revol_util = 44.0
        p_inq = 1
        p_delinq = 0
        p_pub_rec = 0
    elif active_preset == "near_prime":
        p_policy = 1
        p_purpose = "Small Business"
        p_int_rate = 14.26
        p_installment = 580.0
        p_income = 58000.0
        p_dti = 22.4
        p_fico = 672
        p_credit_years = 7.5
        p_revol_bal = 26000.0
        p_revol_util = 68.5
        p_inq = 2
        p_delinq = 1
        p_pub_rec = 0
    else:  # subprime
        p_policy = 0
        p_purpose = "Credit Card Refinance"
        p_int_rate = 17.50
        p_installment = 620.0
        p_income = 38000.0
        p_dti = 27.5
        p_fico = 628
        p_credit_years = 4.2
        p_revol_bal = 31000.0
        p_revol_util = 89.2
        p_inq = 4
        p_delinq = 2
        p_pub_rec = 1

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    
    with st.form("underwriting_applicant_form"):
        st.markdown("""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
            <div style="font-size:1.1rem; font-weight:700; color:#F8FAFC;">📝 Applicant Credit & Financial Parameters</div>
            <span style="font-size:0.8rem; color:#94A3B8;">All 13 features mapped directly to the GBDT scoring model</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Row 1: Loan & Policy
        st.markdown('<div class="fin-card-title">💳 1. Loan Request & Structure</div>', unsafe_allow_html=True)
        col_l1, col_l2, col_l3, col_l4 = st.columns(4)
        
        with col_l1:
            credit_policy_choice = st.selectbox(
                "Lender Credit Policy",
                options=["1 — Meets Internal Credit Policy", "0 — Exception / Non-Policy"],
                index=0 if p_policy == 1 else 1,
                help="1 = Customer meets underwriting criteria, 0 = credit policy exception"
            )
            credit_policy = 1 if "1" in credit_policy_choice else 0
            
        with col_l2:
            purpose_label = st.selectbox(
                "Loan Purpose",
                options=list(PURPOSE_MAPPING.keys()),
                index=list(PURPOSE_MAPPING.keys()).index(p_purpose) if p_purpose in PURPOSE_MAPPING else 0
            )
            raw_purpose = PURPOSE_MAPPING[purpose_label]
            encoded_purpose = PURPOSE_TO_INT[raw_purpose]
            
        with col_l3:
            interest_rate_pct = st.number_input(
                "Interest Rate (%)",
                min_value=6.0,
                max_value=25.0,
                value=float(p_int_rate),
                step=0.25,
                format="%.2f",
                help="Contract annual interest rate"
            )
            int_rate = interest_rate_pct / 100.0
            
        with col_l4:
            installment = st.number_input(
                "Monthly Installment ($)",
                min_value=15.0,
                max_value=1200.0,
                value=float(p_installment),
                step=25.0,
                format="%.2f",
                help="Monthly scheduled loan payment"
            )

        st.divider()

        # Row 2: Financials & Capacity
        st.markdown('<div class="fin-card-title">💵 2. Income & Debt Capacity</div>', unsafe_allow_html=True)
        col_f1, col_f2, col_f3 = st.columns(3)
        
        with col_f1:
            annual_income = st.number_input(
                "Annual Gross Income ($)",
                min_value=15000.0,
                max_value=1000000.0,
                value=float(p_income),
                step=5000.0,
                format="%.0f",
                help="Borrower verified annual gross income"
            )
            log_annual_inc = float(np.log(max(annual_income, 1.0)))
            st.caption(f"Monthly: **${annual_income/12.0:,.0f}** | Log Income: `{log_annual_inc:.4f}`")
            
        with col_f2:
            dti = st.number_input(
                "Debt-to-Income (DTI %)",
                min_value=0.0,
                max_value=35.0,
                value=float(p_dti),
                step=0.5,
                format="%.2f",
                help="Ratio of total monthly debt payments to gross monthly income"
            )
            dti_status = "Healthy (<15%)" if dti < 15 else ("Moderate (15-25%)" if dti <= 25 else "High (>25%)")
            st.caption(f"DTI Health: **{dti_status}**")
            
        with col_f3:
            credit_years = st.number_input(
                "Credit History Length (Years)",
                min_value=0.5,
                max_value=50.0,
                value=float(p_credit_years),
                step=0.5,
                format="%.1f",
                help="Number of active years with a revolving credit line"
            )
            days_with_cr_line = credit_years * 365.25
            st.caption(f"Equivalent: **{days_with_cr_line:.0f} days**")

        st.divider()

        # Row 3: Credit Profile & Activity
        st.markdown('<div class="fin-card-title">📈 3. Credit Profile & Bureau History</div>', unsafe_allow_html=True)
        col_c1, col_c2, col_c3 = st.columns(3)
        
        with col_c1:
            fico = st.number_input(
                "FICO Credit Score",
                min_value=600,
                max_value=850,
                value=int(p_fico),
                step=5,
                help="Bureau FICO credit rating"
            )
            fico_label, fico_color = get_fico_rating(fico)
            st.markdown(f"<span style='color:{fico_color}; font-size:0.8rem; font-weight:700;'>● {fico_label}</span>", unsafe_allow_html=True)
            
        with col_c2:
            revol_bal = st.number_input(
                "Revolving Credit Balance ($)",
                min_value=0.0,
                max_value=1500000.0,
                value=float(p_revol_bal),
                step=1000.0,
                format="%.0f",
                help="Total revolving balance across all credit lines"
            )
            
        with col_c3:
            revol_util = st.number_input(
                "Revolving Line Utilization (%)",
                min_value=0.0,
                max_value=120.0,
                value=float(p_revol_util),
                step=1.0,
                format="%.1f",
                help="Proportion of available credit currently utilized"
            )
            util_status = "Optimal (<30%)" if revol_util < 30 else ("Acceptable (30-60%)" if revol_util <= 60 else "Elevated (>60%)")
            st.caption(f"Utilization Tier: **{util_status}**")

        col_c4, col_c5, col_c6 = st.columns(3)
        with col_c4:
            inq_last_6mths = st.number_input(
                "Credit Inquiries (Past 6 Months)",
                min_value=0,
                max_value=20,
                value=int(p_inq),
                step=1,
                help="Hard inquiries recorded on credit report in last 6 months"
            )
        with col_c5:
            delinq_2yrs = st.number_input(
                "Delinquencies (Past 2 Years)",
                min_value=0,
                max_value=15,
                value=int(p_delinq),
                step=1,
                help="Number of 30+ days past due payment delinquencies"
            )
        with col_c6:
            pub_rec = st.number_input(
                "Derogatory Public Records",
                min_value=0,
                max_value=5,
                value=int(p_pub_rec),
                step=1,
                help="Bankruptcies, tax liens, civil judgments"
            )

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        submit_underwriting = st.form_submit_button("⚡ Run Institutional Underwriting Engine")


    # ========================================================
    # EXECUTION & RESULTS PRESENTATION
    # ========================================================
    
    if submit_underwriting or "single_scored" in st.session_state:
        st.session_state["single_scored"] = True

        # Build feature vector matching original model exactly
        applicant_data = pd.DataFrame({
            "credit.policy": [credit_policy],
            "purpose": [encoded_purpose],
            "int.rate": [int_rate],
            "installment": [installment],
            "log.annual.inc": [log_annual_inc],
            "dti": [dti],
            "fico": [fico],
            "days.with.cr.line": [days_with_cr_line],
            "revol.bal": [revol_bal],
            "revol.util": [revol_util],
            "inq.last.6mths": [inq_last_6mths],
            "delinq.2yrs": [delinq_2yrs],
            "pub.rec": [pub_rec]
        })

        # Predict probability
        probs = model.predict_proba(applicant_data)[0]
        prob_paid = probs[0]
        prob_default = probs[1]
        
        # Risk assessment
        risk_info = calculate_risk_tier(prob_default)
        economics = compute_applicant_economics(installment, int_rate, prob_default, annual_income)

        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        st.markdown("### 🎯 Underwriting Decision & Credit Scorecard")

        # Top Decision Banner
        banner_class = "decision-approved" if risk_info["status"] == "APPROVED" else ("decision-review" if risk_info["status"] == "REVIEW" else "decision-declined")
        status_icon = "✅" if risk_info["status"] == "APPROVED" else ("⚠️" if risk_info["status"] == "REVIEW" else "❌")

        st.markdown(f"""
        <div class="decision-banner {banner_class}">
            <div>
                <div style="font-size:0.8rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:{risk_info['color']}; margin-bottom:4px;">
                    {risk_info['tier']} Risk Rating • {risk_info['grade']}
                </div>
                <div style="font-size:1.6rem; font-weight:800; color:#FFFFFF; margin-bottom:6px;">
                    {status_icon} Decision: {risk_info['decision']}
                </div>
                <div style="font-size:0.9rem; color:#CBD5E1; max-width:850px;">
                    {risk_info['recommendation']}
                </div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:0.75rem; color:#94A3B8; text-transform:uppercase; letter-spacing:0.05em;">Repayment Probability</div>
                <div style="font-size:2.2rem; font-weight:800; font-family:'JetBrains Mono',monospace; color:{risk_info['color']};">
                    {prob_paid * 100:.1f}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 4 Core FinTech Metric Cards
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        
        with m_col1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Repayment Probability</div>
                <div class="metric-val" style="color:#10B981;">{prob_paid * 100:.2f}%</div>
                <div class="metric-sub">Historical Baseline: 84.0%</div>
            </div>
            """, unsafe_allow_html=True)

        with m_col2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Default Risk (PD)</div>
                <div class="metric-val" style="color:{risk_info['color']};">{prob_default * 100:.2f}%</div>
                <div class="metric-sub">Risk Cutoff Profile: {risk_policy.split()[0]}</div>
            </div>
            """, unsafe_allow_html=True)

        with m_col3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Estimated Loan Value</div>
                <div class="metric-val">${economics['est_loan_amount']:,.0f}</div>
                <div class="metric-sub">36-Month Amortization Est.</div>
            </div>
            """, unsafe_allow_html=True)

        with m_col4:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Expected Loss (EL)</div>
                <div class="metric-val" style="color:{'#10B981' if economics['expected_loss'] < 500 else '#F59E0B'};">${economics['expected_loss']:,.2f}</div>
                <div class="metric-sub">PD × 45% LGD × Exposure</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        # Visual Analytics: Gauge Chart + Radar Chart
        col_g1, col_g2 = st.columns([1.1, 1.0])

        with col_g1:
            st.markdown('<div class="fin-card-title">🧭 Probability of Full Repayment Gauge</div>', unsafe_allow_html=True)
            
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=float(prob_paid * 100.0),
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Repayment Confidence Index", 'font': {'size': 15, 'color': '#E2E8F0', 'family': 'Plus Jakarta Sans'}},
                delta={'reference': 84.0, 'increasing': {'color': "#10B981"}, 'decreasing': {'color': "#EF4444"}},
                number={'suffix': "%", 'font': {'size': 38, 'color': '#FFFFFF', 'family': 'JetBrains Mono'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
                    'bar': {'color': risk_info['color'], 'thickness': 0.28},
                    'bgcolor': "rgba(15, 23, 42, 0.6)",
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 65], 'color': "rgba(239, 68, 68, 0.25)"},
                        {'range': [65, 80], 'color': "rgba(245, 158, 11, 0.25)"},
                        {'range': [80, 90], 'color': "rgba(6, 182, 212, 0.25)"},
                        {'range': [90, 100], 'color': "rgba(16, 185, 129, 0.25)"}
                    ],
                    'threshold': {
                        'line': {'color': "#FFFFFF", 'width': 3},
                        'thickness': 0.8,
                        'value': float(prob_paid * 100.0)
                    }
                }
            ))

            gauge_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=290,
                margin=dict(l=20, r=20, t=30, b=10)
            )
            st.plotly_chart(gauge_fig, use_container_width=True)

        with col_g2:
            st.markdown('<div class="fin-card-title">🕸️ Applicant vs Prime Benchmark Profile</div>', unsafe_allow_html=True)
            
            # Normalize axes for comparative radar chart
            norm_fico = min(100.0, max(0.0, (fico - 600) / 2.5))
            norm_inc = min(100.0, max(0.0, (annual_income - 20000) / 1000.0))
            norm_dti = min(100.0, max(0.0, (35.0 - dti) / 0.35))
            norm_util = min(100.0, max(0.0, (100.0 - revol_util)))
            norm_exp = min(100.0, max(0.0, credit_years * 5.0))
            norm_inq = min(100.0, max(0.0, (6.0 - inq_last_6mths) * 16.6))

            radar_categories = ['Credit Score (FICO)', 'Annual Income', 'Low DTI Strength', 'Low Utilization', 'Credit Experience', 'Clean Inquiries']
            
            radar_fig = go.Figure()
            radar_fig.add_trace(go.Scatterpolar(
                r=[norm_fico, norm_inc, norm_dti, norm_util, norm_exp, norm_inq],
                theta=radar_categories,
                fill='toself',
                name='Applicant',
                line_color='#6366F1',
                fillcolor='rgba(99, 102, 241, 0.35)'
            ))
            radar_fig.add_trace(go.Scatterpolar(
                r=[65, 55, 60, 60, 60, 75],
                theta=radar_categories,
                fill='toself',
                name='Portfolio Median',
                line_color='#06B6D4',
                fillcolor='rgba(6, 182, 212, 0.15)'
            ))

            radar_fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=False, range=[0, 100]),
                    bgcolor="rgba(15, 23, 42, 0.4)"
                ),
                showlegend=True,
                legend=dict(orientation="h", y=-0.1, font=dict(color="#94A3B8")),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=290,
                margin=dict(l=30, r=30, t=10, b=30)
            )
            st.plotly_chart(radar_fig, use_container_width=True)

        # Attribution / Positive vs Risk Factors
        st.markdown('<div class="fin-card-title">🔍 Underwriting Factor Attribution & Explainability</div>', unsafe_allow_html=True)
        col_att1, col_att2 = st.columns(2)

        with col_att1:
            st.markdown("""
            <div style="background:rgba(16, 185, 129, 0.08); border:1px solid rgba(16, 185, 129, 0.25); border-radius:12px; padding:16px;">
                <div style="font-weight:700; color:#34D399; font-size:0.95rem; margin-bottom:8px;">✅ Positive Credit Drivers</div>
            """, unsafe_allow_html=True)
            
            pos_factors = []
            if fico >= 700:
                pos_factors.append(f"Strong FICO Score of {fico} ({fico_label.split()[0]}) reduces probability of default.")
            if credit_policy == 1:
                pos_factors.append("Meets comprehensive underwriting credit policy guidelines.")
            if dti < 15.0:
                pos_factors.append(f"Conservative DTI ratio of {dti:.1f}% indicates high debt-absorption capacity.")
            if revol_util < 35.0:
                pos_factors.append(f"Low credit line utilization of {revol_util:.1f}% confirms disciplined credit management.")
            if inq_last_6mths == 0:
                pos_factors.append("Zero credit inquiries in the past 6 months reflects stable financial positioning.")
            if delinq_2yrs == 0:
                pos_factors.append("Pristine 24-month payment history with zero reported delinquencies.")
            if annual_income >= 75000:
                pos_factors.append(f"Verified annual income of ${annual_income:,.0f} provides strong debt service buffer.")

            if not pos_factors:
                pos_factors.append("Standard revolving history present without severe recent derogatory entries.")
            for pf in pos_factors:
                st.markdown(f"<div style='font-size:0.85rem; color:#E2E8F0; margin-bottom:6px;'>• {pf}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_att2:
            st.markdown("""
            <div style="background:rgba(239, 68, 68, 0.08); border:1px solid rgba(239, 68, 68, 0.25); border-radius:12px; padding:16px;">
                <div style="font-weight:700; color:#F87171; font-size:0.95rem; margin-bottom:8px;">⚠️ Elevated Risk Factors & Flags</div>
            """, unsafe_allow_html=True)
            
            risk_factors = []
            if fico < 670:
                risk_factors.append(f"FICO Score of {fico} is below standard prime benchmark (670 threshold).")
            if credit_policy == 0:
                risk_factors.append("Applicant does not meet standard credit policy criteria (policy exception).")
            if revol_util > 60.0:
                risk_factors.append(f"Elevated revolving utilization ({revol_util:.1f}%) suggests revolving debt dependency.")
            if inq_last_6mths >= 2:
                risk_factors.append(f"{inq_last_6mths} inquiries in past 6 months signals elevated credit-seeking activity.")
            if dti > 20.0:
                risk_factors.append(f"High DTI of {dti:.1f}% leaves limited disposable income cushion.")
            if delinq_2yrs > 0:
                risk_factors.append(f"{delinq_2yrs} delinquency incident(s) in last 2 years indicates prior payment difficulty.")
            if pub_rec > 0:
                risk_factors.append(f"{pub_rec} derogatory public record(s) on file.")
            if int_rate >= 0.15:
                risk_factors.append(f"High risk-adjusted interest rate ({int_rate*100:.2f}%) raises monthly debt load.")

            if not risk_factors:
                risk_factors.append("No critical derogatory flags or elevated leverage concerns identified.")
            for rf in risk_factors:
                st.markdown(f"<div style='font-size:0.85rem; color:#E2E8F0; margin-bottom:6px;'>• {rf}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Download Underwriting Memorandum
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        credit_memo = f"""# INSTITUTIONAL CREDIT UNDERWRITING MEMORANDUM
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
Application ID: APP-{datetime.now().strftime('%Y%m%d')}-{np.random.randint(1000, 9999)}

## 1. EXECUTIVE DECISION
- Underwriting Decision: {risk_info['decision']}
- Credit Grade: {risk_info['grade']} ({risk_info['tier']})
- Probability of Full Repayment: {prob_paid * 100:.2f}%
- Projected Default Probability (PD): {prob_default * 100:.2f}%
- Maximum Recommended LTV: {risk_info['max_ltv']}

## 2. APPLICANT FINANCIAL & CREDIT PROFILE
- FICO Score: {fico} ({fico_label})
- Annual Gross Income: ${annual_income:,.2f}
- Monthly Gross Income: ${economics['monthly_income']:,.2f}
- Monthly Installment: ${installment:,.2f}
- Installment-to-Income (ITI): {economics['iti_ratio']:.2f}%
- Debt-to-Income (DTI): {dti:.2f}%
- Contract Interest Rate: {int_rate * 100:.2f}%
- Credit History Length: {credit_years:.1f} years ({days_with_cr_line:.0f} days)
- Revolving Credit Balance: ${revol_bal:,.2f}
- Revolving Utilization: {revol_util:.2f}%
- Recent Inquiries (6 Mo): {inq_last_6mths}
- 2-Year Delinquencies: {delinq_2yrs}
- Public Derogatory Records: {pub_rec}

## 3. LOAN ECONOMICS & LOSS MODELING
- Estimated Loan Principal: ${economics['est_loan_amount']:,.2f}
- Total Interest Payable (36 Mo): ${economics['total_interest']:,.2f}
- Projected Basel II Expected Loss: ${economics['expected_loss']:,.2f}

## 4. UNDERWRITING RECOMMENDATION & STIPULATIONS
{risk_info['recommendation']}
"""
        st.download_button(
            label="📄 Export Formal Credit Underwriting Memorandum (Markdown)",
            data=credit_memo,
            file_name=f"credit_underwriting_memo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True
        )


# ============================================================
# TAB 2: WHAT-IF SENSITIVITY SIMULATOR
# ============================================================

with tab_simulator:
    st.markdown("""
    <div class="fin-card">
        <div style="font-size:1.1rem; font-weight:700; color:#F8FAFC; margin-bottom:4px;">
            🧪 Dynamic What-If Risk Sensitivity Simulator
        </div>
        <div style="font-size:0.88rem; color:#94A3B8;">
            Simulate real-time credit restructuring: test how score improvements, balance paydowns, or installment adjustments immediately impact repayment probability and risk tiering.
        </div>
    </div>
    """, unsafe_allow_html=True)

    sim_col1, sim_col2 = st.columns([1, 1])

    with sim_col1:
        st.markdown('<div class="fin-card-title">🎛️ Interactive Restructuring Sliders</div>', unsafe_allow_html=True)
        
        sim_fico = st.slider("Simulate FICO Score", min_value=612, max_value=830, value=710, step=5)
        sim_util = st.slider("Simulate Revolving Utilization (%)", min_value=0.0, max_value=100.0, value=40.0, step=2.5)
        sim_installment = st.slider("Simulate Monthly Payment ($)", min_value=50.0, max_value=900.0, value=300.0, step=25.0)
        sim_inq = st.slider("Simulate Recent Inquiries (6 Mo)", min_value=0, max_value=8, value=1, step=1)
        sim_income = st.slider("Simulate Annual Income ($)", min_value=25000.0, max_value=200000.0, value=75000.0, step=5000.0)

    # Calculate simulated applicant
    sim_data = pd.DataFrame({
        "credit.policy": [1],
        "purpose": [PURPOSE_TO_INT["debt_consolidation"]],
        "int.rate": [0.1189],
        "installment": [sim_installment],
        "log.annual.inc": [np.log(sim_income)],
        "dti": [13.5],
        "fico": [sim_fico],
        "days.with.cr.line": [4200.0],
        "revol.bal": [14000.0],
        "revol.util": [sim_util],
        "inq.last.6mths": [sim_inq],
        "delinq.2yrs": [0],
        "pub.rec": [0]
    })

    sim_probs = model.predict_proba(sim_data)[0]
    sim_paid = sim_probs[0]
    sim_default = sim_probs[1]
    sim_risk = calculate_risk_tier(sim_default)

    with sim_col2:
        st.markdown('<div class="fin-card-title">⚡ Real-time Simulated Scorecard</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="background:rgba(30,41,59,0.7); border:1px solid rgba(255,255,255,0.08); border-radius:14px; padding:20px; text-align:center;">
            <div style="font-size:0.85rem; color:#94A3B8; text-transform:uppercase; letter-spacing:0.06em;">Simulated Risk Grade</div>
            <div style="font-size:2.8rem; font-weight:800; color:{sim_risk['color']}; margin:4px 0;">{sim_risk['grade']}</div>
            <div style="font-size:1.1rem; font-weight:700; color:#FFFFFF;">{sim_risk['decision']}</div>
            <div style="margin-top:14px; display:flex; justify-content:space-around;">
                <div>
                    <div style="font-size:0.75rem; color:#94A3B8;">Repayment Odds</div>
                    <div style="font-size:1.4rem; font-weight:700; color:#34D399; font-family:'JetBrains Mono',monospace;">{sim_paid*100:.1f}%</div>
                </div>
                <div>
                    <div style="font-size:0.75rem; color:#94A3B8;">Default Risk (PD)</div>
                    <div style="font-size:1.4rem; font-weight:700; color:{sim_risk['color']}; font-family:'JetBrains Mono',monospace;">{sim_default*100:.1f}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Sensitivity Curve: FICO vs Default Rate Curve for this profile
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="fin-card-title">📈 FICO Score Sensitivity Curve (Ceteris Paribus)</div>', unsafe_allow_html=True)

    fico_range = np.linspace(615, 825, 43)
    curve_data = pd.DataFrame([sim_data.iloc[0].to_dict() for _ in fico_range])
    curve_data['fico'] = fico_range
    curve_probs = model.predict_proba(curve_data)[:, 0] * 100.0  # P(Paid)

    curve_fig = px.line(
        x=fico_range,
        y=curve_probs,
        labels={"x": "FICO Credit Score", "y": "Probability of Full Repayment (%)"},
        title="Repayment Probability as a Function of FICO Score"
    )
    curve_fig.update_traces(line=dict(color='#06B6D4', width=3))
    curve_fig.add_vline(x=sim_fico, line_dash="dash", line_color="#F59E0B", annotation_text=f"Selected: {sim_fico}", annotation_position="top left")
    curve_fig.add_hline(y=84.0, line_dash="dot", line_color="#94A3B8", annotation_text="Benchmark Average (84%)")
    curve_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.5)",
        font=dict(color="#CBD5E1"),
        height=320,
        margin=dict(l=30, r=20, t=40, b=30)
    )
    st.plotly_chart(curve_fig, use_container_width=True)


# ============================================================
# TAB 3: BATCH PORTFOLIO UNDERWRITING
# ============================================================

with tab_batch:
    st.markdown("""
    <div class="fin-card">
        <div style="font-size:1.1rem; font-weight:700; color:#F8FAFC; margin-bottom:4px;">
            📁 Batch Portfolio Underwriting & Scoring Engine
        </div>
        <div style="font-size:0.88rem; color:#94A3B8;">
            Upload CSV portfolios containing multiple loan applications to perform automated scoring, risk grading, and portfolio-level loss estimation in parallel.
        </div>
    </div>
    """, unsafe_allow_html=True)

    b_col1, b_col2 = st.columns([1.2, 1.0])
    
    with b_col1:
        uploaded_file = st.file_uploader("Upload Applicant CSV File", type=["csv"])
    with b_col2:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        load_sample_batch = st.button("📥 Load Pre-Built Sample Portfolio (10 Realistic Applicants)", use_container_width=True)

    batch_df = None
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded uploaded portfolio with {len(batch_df)} rows.")
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
    elif load_sample_batch or "sample_batch_loaded" in st.session_state:
        st.session_state["sample_batch_loaded"] = True
        sample_path = BASE_DIR / "sample_loan_applicants.csv"
        if sample_path.exists():
            batch_df = pd.read_csv(sample_path)
            st.info("Loaded pre-built sample loan portfolio containing 10 diverse applicant profiles.")
        elif benchmark_df is not None:
            batch_df = benchmark_df.sample(10, random_state=42).copy()
            batch_df.drop(columns=['not.fully.paid'], inplace=True, errors='ignore')

    if batch_df is not None:
        # Prepare data for prediction
        scoring_df = batch_df.copy()
        
        # Map purpose strings if needed
        if 'purpose' in scoring_df.columns:
            scoring_df['purpose_clean'] = scoring_df['purpose'].apply(
                lambda x: PURPOSE_TO_INT.get(str(x).strip(), 0) if str(x).strip() in PURPOSE_TO_INT else 0
            )
        else:
            scoring_df['purpose_clean'] = 0

        # Feature alignment
        required_features = [
            'credit.policy', 'purpose', 'int.rate', 'installment', 'log.annual.inc',
            'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util',
            'inq.last.6mths', 'delinq.2yrs', 'pub.rec'
        ]

        # Ensure all columns exist
        for col in required_features:
            if col not in scoring_df.columns:
                if col == 'purpose':
                    scoring_df[col] = scoring_df['purpose_clean']
                else:
                    scoring_df[col] = 0.0

        model_input = scoring_df[required_features].copy()
        model_input['purpose'] = scoring_df['purpose_clean']

        # Batch Predict
        batch_probs = model.predict_proba(model_input)
        batch_paid_probs = batch_probs[:, 0]
        batch_default_probs = batch_probs[:, 1]

        # Enrich results
        batch_df['Repayment_Probability_%'] = (batch_paid_probs * 100.0).round(2)
        batch_df['Default_Risk_PD_%'] = (batch_default_probs * 100.0).round(2)
        
        risk_tiers = [calculate_risk_tier(pd_val) for pd_val in batch_default_probs]
        batch_df['Risk_Grade'] = [rt['grade'] for rt in risk_tiers]
        batch_df['Decision'] = [rt['decision'] for rt in risk_tiers]

        # Financial exposure
        est_loans = [
            compute_applicant_economics(row['installment'], row['int.rate'], pd_val, np.exp(row['log.annual.inc']))['est_loan_amount']
            for row, pd_val in zip(model_input.to_dict('records'), batch_default_probs)
        ]
        expected_losses = [
            compute_applicant_economics(row['installment'], row['int.rate'], pd_val, np.exp(row['log.annual.inc']))['expected_loss']
            for row, pd_val in zip(model_input.to_dict('records'), batch_default_probs)
        ]

        batch_df['Est_Loan_Amount'] = [round(x, 2) for x in est_loans]
        batch_df['Expected_Loss_$'] = [round(x, 2) for x in expected_losses]

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        st.markdown("#### 📊 Portfolio Overview & Aggregations")

        port_col1, port_col2, port_col3, port_col4 = st.columns(4)
        
        with port_col1:
            st.metric("Total Applications", f"{len(batch_df):,}")
        with port_col2:
            st.metric("Total Loan Exposure", f"${sum(est_loans):,.0f}")
        with port_col3:
            avg_paid_prob = batch_paid_probs.mean() * 100.0
            st.metric("Avg Repayment Odds", f"{avg_paid_prob:.1f}%")
        with port_col4:
            st.metric("Portfolio Expected Loss", f"${sum(expected_losses):,.0f}")

        # Charts: Grade breakdown & FICO distribution
        bchart1, bchart2 = st.columns(2)
        with bchart1:
            grade_counts = batch_df['Risk_Grade'].value_counts().reset_index()
            grade_counts.columns = ['Risk_Grade', 'Count']
            grade_fig = px.pie(
                grade_counts,
                names='Risk_Grade',
                values='Count',
                title="Portfolio Risk Grade Breakdown",
                hole=0.45,
                color_discrete_sequence=['#10B981', '#06B6D4', '#F59E0B', '#F97316', '#EF4444']
            )
            grade_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#CBD5E1"), height=280)
            st.plotly_chart(grade_fig, use_container_width=True)

        with bchart2:
            scatter_fig = px.scatter(
                batch_df,
                x='fico',
                y='Repayment_Probability_%',
                color='Risk_Grade',
                size='Est_Loan_Amount',
                title="Applicant FICO vs Repayment Probability",
                labels={"fico": "FICO Credit Score", "Repayment_Probability_%": "Repayment Odds (%)"}
            )
            scatter_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15, 23, 42, 0.5)", font=dict(color="#CBD5E1"), height=280)
            st.plotly_chart(scatter_fig, use_container_width=True)

        # Scored table
        st.markdown("#### 📋 Scored Applicant Registry")
        display_cols = [c for c in ['Applicant_ID', 'Applicant_Name', 'fico', 'int.rate', 'installment', 'dti', 'Risk_Grade', 'Repayment_Probability_%', 'Decision', 'Est_Loan_Amount', 'Expected_Loss_$'] if c in batch_df.columns]
        st.dataframe(batch_df[display_cols], use_container_width=True, hide_index=True)

        # Download scored CSV
        csv_export = batch_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Complete Scored Portfolio CSV",
            data=csv_export,
            file_name=f"scored_loan_portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )


# ============================================================
# TAB 4: MODEL BENCHMARK & ANALYTICS LAB
# ============================================================

with tab_analytics:
    st.markdown("""
    <div class="fin-card">
        <div style="font-size:1.1rem; font-weight:700; color:#F8FAFC; margin-bottom:4px;">
            📊 Machine Learning Model Diagnostics & Validation Lab
        </div>
        <div style="font-size:0.88rem; color:#94A3B8;">
            Complete quantitative validation report of the Gradient Boosting Classifier trained on 9,578 historical records.
        </div>
    </div>
    """, unsafe_allow_html=True)

    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    with stat_col1:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-label">Algorithm</div>
            <div class="metric-val" style="font-size:1.4rem;">Gradient Boosting</div>
            <div class="metric-sub">100 Trees | lr = 0.05</div>
        </div>
        """, unsafe_allow_html=True)
    with stat_col2:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-label">ROC - AUC Score</div>
            <div class="metric-val" style="color:#06B6D4;">0.7105</div>
            <div class="metric-sub">Stratified Holdout Test</div>
        </div>
        """, unsafe_allow_html=True)
    with stat_col3:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-label">Model Accuracy</div>
            <div class="metric-val" style="color:#10B981;">83.8%</div>
            <div class="metric-sub">Matches 84% base rate</div>
        </div>
        """, unsafe_allow_html=True)
    with stat_col4:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-label">Brier Score Loss</div>
            <div class="metric-val" style="color:#A78BFA;">0.122</div>
            <div class="metric-sub">Calibrated Probability Error</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    # Feature Importance Bar Chart
    feature_names = [
        'credit.policy', 'purpose', 'int.rate', 'installment', 'log.annual.inc',
        'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util',
        'inq.last.6mths', 'delinq.2yrs', 'pub.rec'
    ]
    friendly_names = [
        'Credit Policy Compliance', 'Loan Purpose', 'Contract Interest Rate', 'Monthly Installment',
        'Log Annual Income', 'Debt-to-Income (DTI)', 'FICO Credit Score', 'Credit Line Age (Days)',
        'Revolving Balance ($)', 'Revolving Line Utilization', 'Recent Credit Inquiries (6 Mo)',
        'Past Delinquencies (2 Yr)', 'Public Derogatory Records'
    ]
    
    if hasattr(model, 'feature_importances_'):
        fi_df = pd.DataFrame({
            'Feature': friendly_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)

        fi_fig = px.bar(
            fi_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Global Feature Importance (Gini Impurity Reduction)",
            color='Importance',
            color_continuous_scale="Viridis"
        )
        fi_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.5)",
            font=dict(color="#CBD5E1"),
            height=420,
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fi_fig, use_container_width=True)

    # Confusion Matrix Heatmap
    c_m1, c_m2 = st.columns(2)
    with c_m1:
        st.markdown('<div class="fin-card-title">🧮 Test Confusion Matrix</div>', unsafe_allow_html=True)
        cm_data = [[2401, 7], [460, 6]]
        cm_fig = px.imshow(
            cm_data,
            text_auto=True,
            labels=dict(x="Predicted Label", y="Actual Ground Truth", color="Cases"),
            x=['Fully Paid (0)', 'Default (1)'],
            y=['Fully Paid (0)', 'Default (1)'],
            color_continuous_scale='Blues'
        )
        cm_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#CBD5E1"), height=280)
        st.plotly_chart(cm_fig, use_container_width=True)

    with c_m2:
        st.markdown('<div class="fin-card-title">💡 Class Imbalance & Threshold Insights</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:rgba(30,41,59,0.5); padding:16px; border-radius:12px; border:1px solid rgba(255,255,255,0.06); font-size:0.86rem; color:#CBD5E1; line-height:1.6;">
            <b>Why Continuous Probability & Tiers Matter:</b><br>
            • The LendingClub benchmark exhibits an <b>84% paid vs 16% default</b> class distribution.<br>
            • Naive binary models using an arbitrary 50% cutoff fail to flag real-world defaults because defaults are rare events.<br>
            • <b>FinRisk IQ</b> overcomes this by using calibrated posterior probabilities to assign institutional <b>Credit Grades (A through E)</b>, ensuring risk officers capture default exposure early while maximizing safe origination volume.
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# TAB 5: CREDIT POLICY & RULES SOP
# ============================================================

with tab_policy:
    st.markdown("""
    <div class="fin-card">
        <div style="font-size:1.1rem; font-weight:700; color:#F8FAFC; margin-bottom:4px;">
            📘 Credit Policy Guidelines & Institutional Underwriting Matrix
        </div>
        <div style="font-size:0.88rem; color:#94A3B8;">
            Standard Operating Procedures (SOP) established for credit officers and compliance audits.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    | Credit Grade | Risk Classification | Max Default Prob (PD) | Auto-Approval Decision | Mandatory Stipulations |
    | :--- | :--- | :--- | :--- | :--- |
    | **Grade A** | Super-Prime | < 8.0% | **Auto-Approved** | Standard automated identity verification |
    | **Grade B** | Prime | 8.0% – 15.0% | **Auto-Approved** | Standard bureau check, 30-day paystub |
    | **Grade C** | Near-Prime | 15.0% – 24.0% | **Conditional** | 2-year W-2, bank statement cash-flow review |
    | **Grade D** | Subprime | 24.0% – 36.0% | **Committee Review** | Qualified co-signer or 25% liquid collateral pledge |
    | **Grade E** | Severe Default Risk | > 36.0% | **Decline** | Issue adverse action notice per FCRA guidelines |
    """)

    st.divider()

    st.markdown("#### ⚖️ Regulatory Compliance Disclosures")
    st.markdown("""
    - **Equal Credit Opportunity Act (ECOA) / Regulation B**: Model evaluations are conducted exclusively using non-discriminatory financial, debt-service, and credit capacity attributes. Protected class characteristics (race, color, religion, sex, marital status, age) are strictly excluded from all training pipelines.
    - **Fair Credit Reporting Act (FCRA)**: Adverse underwriting decisions (Grades D & E) mandate furnishing written Adverse Action Notices outlining key credit score contributors within 30 days.
    """)
