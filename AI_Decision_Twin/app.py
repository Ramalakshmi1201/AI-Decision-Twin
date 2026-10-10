import streamlit as st

import pandas as pd

import joblib
from pathlib import Path

import plotly.graph_objects as go

import plotly.express as px



# --------------------------------

# PAGE CONFIG

# --------------------------------



st.set_page_config(

    page_title="AI Decision Twin",

    page_icon="🏦",

    layout="wide"

)



# --------------------------------

# CUSTOM CSS

# --------------------------------



st.markdown("""

<style>



.main {

    background-color: #0E1117;

}



h1 {

    color: #00FFAA;

}



.block-container {

    padding-top: 2rem;

}



</style>

""", unsafe_allow_html=True)



# --------------------------------

# LOAD MODEL

# --------------------------------



BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "loan_model.pkl"
DATA_PATH = BASE_DIR / "loan_data.csv"

if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)
else:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    df = pd.read_csv(DATA_PATH)
    X = df.drop("approved", axis=1)
    y = df["approved"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH, compress=3)





# --------------------------------

# HEADER

# --------------------------------



st.title("🏦 AI Decision Twin")



st.caption(

    "Virtual Financial Twin for Explainable Loan Approval Decisions"

)



st.info(

    """

Simulate financial decisions before they happen.



AI Decision Twin predicts loan approval probability and shows how changes in income, debt and credit score affect future outcomes.

"""

)

# --------------------------------

# INPUT SECTION

# --------------------------------



st.markdown("---")

st.subheader("📝 Applicant Profile")



col1, col2 = st.columns(2)



with col1:



    age = st.slider(

        "Age",

        21,

        60,

        30

    )



    income = st.slider(

        "Monthly Income (₹)",

        20000,

        150000,

        70000,

        step=1000

    )



    credit_score = st.slider(

        "Credit Score",

        300,

        900,

        700

    )



    loan_amount = st.slider(

        "Loan Amount (₹)",

        50000,

        1000000,

        300000,

        step=10000

    )



with col2:



    employment = st.selectbox(

        "Employment Status",

        [

            "Employed",

            "Unemployed"

        ]

    )



    savings = st.slider(

        "Savings (₹)",

        10000,

        500000,

        100000,

        step=5000

    )



    debt = st.slider(

        "Existing Debt (₹)",

        0,

        800000,

        100000,

        step=10000

    )



    employment_years = st.slider(

        "Employment Years",

        0,

        25,

        5

    )



    dependents = st.slider(

        "Dependents",

        0,

        5,

        1

    )



# Convert Employment to Numerical



employment_num = 1 if employment == "Employed" else 0

# --------------------------------

# PREDICTION DATA

# --------------------------------



current_data = pd.DataFrame(

    [[

        age,

        income,

        credit_score,

        loan_amount,

        employment_num,

        savings,

        debt,

        employment_years,

        dependents

    ]],

    columns=[

        "age",

        "income",

        "credit_score",

        "loan_amount",

        "employment",

        "savings",

        "debt",

        "employment_years",

        "dependents"

    ]

)



# --------------------------------

# PREDICTION

# --------------------------------



current_probability = (

    model.predict_proba(current_data)[0][1] * 100

)



# --------------------------------

# DECISION LOGIC

# --------------------------------



if current_probability >= 70:

    decision = "APPROVED"

    risk = "LOW RISK"



elif current_probability >= 35:

    decision = "REVIEW REQUIRED"

    risk = "MEDIUM RISK"



else:

    decision = "HIGH RISK"

    risk = "HIGH RISK"



# --------------------------------

# DECISION RESULT

# --------------------------------



st.markdown("---")

st.subheader("📊 Decision Result")



# Status Card



if current_probability >= 70:

    st.success(

        f"""

✅ DECISION: {decision}



Approval Probability: {current_probability:.2f}%



Risk Level: {risk}

"""

    )



elif current_probability >= 35:

    st.warning(

        f"""

🟡 DECISION: {decision}



Approval Probability: {current_probability:.2f}%



Risk Level: {risk}

"""

    )



else:

    st.error(

        f"""

🔴 DECISION: {decision}



Approval Probability: {current_probability:.2f}%



Risk Level: {risk}

"""

    )







# --------------------------------

# GAUGE

# --------------------------------



st.markdown("---")

st.subheader("🎯 Approval Gauge")



gauge_fig = go.Figure(

    go.Indicator(

        mode="gauge+number",

        value=current_probability,

        title={"text": "Approval Score"},

        gauge={

            "axis": {"range": [0, 100]}

        }

    )

)



st.plotly_chart(

    gauge_fig,

    use_container_width=True

)

# --------------------------------

# FINANCIAL HEALTH SCORE

# --------------------------------



health_score = 0



# Credit Score Contribution (30 points)



health_score += min(

    (credit_score / 900) * 30,

    30

)



# Income Contribution (25 points)



health_score += min(

    (income / 150000) * 25,

    25

)



# Savings Contribution (20 points)



health_score += min(

    (savings / 500000) * 20,

    20

)



# Employment Stability (15 points)



health_score += min(

    (employment_years / 25) * 15,

    15

)



# Debt Contribution (10 points)



if debt < 300000:

    health_score += 10



health_score = round(health_score)



st.markdown("---")

st.subheader("💎 Financial Health Score")

col1, col2 = st.columns(2)



with col1:



    st.metric(

        "Health Score",

        f"{health_score}/100"

    )



with col2:



    if health_score >= 80:

        st.success("Excellent Financial Health")



    elif health_score >= 60:

        st.warning("Moderate Financial Health")



    else:

        st.error("Financial Health Needs Improvement")



# --------------------------------

# AI DECISION TWIN SCENARIOS

# --------------------------------



st.markdown("---")

st.subheader("🔮 AI Decision Twin Scenarios")



# --------------------------------

# RULE BASED DECISION TWIN

# --------------------------------



# CREDIT IMPROVEMENT



if credit_score < 600:

    credit_gain = 15



elif credit_score < 700:

    credit_gain = 10



elif credit_score < 800:

    credit_gain = 5



else:

    credit_gain = 2



credit_prob = min(

    current_probability + credit_gain,

    99

)



# INCOME IMPROVEMENT



income_gain = min(

    (20000 / income) * 50,

    20

)



income_prob = min(

    current_probability + income_gain,

    99

)



# DEBT INCREASE



debt_penalty = 8



debt_prob = max(

    current_probability - debt_penalty,

    1

)



gain_credit = (

    credit_prob - current_probability

)



gain_income = (

    income_prob - current_probability

)



gain_debt = (

    debt_prob - current_probability

)





# --------------------------------

# CREDIT CARD

# --------------------------------



st.info(

    f"""

📈 CREDIT SCORE IMPROVEMENT



Current Credit Score:

{credit_score}



Improved Credit Score:

{min(credit_score + 50, 900)}



Approval Probability:



{current_probability:.1f}% → {credit_prob:.1f}%



Impact:



{credit_gain:+.1f}%

"""

)



# --------------------------------

# INCOME CARD

# --------------------------------



st.success(

    f"""

💰 INCOME GROWTH



Current Income:

₹{income:,}



Improved Income:

₹{income + 20000:,}



Approval Probability:



{current_probability:.1f}% → {income_prob:.1f}%



Impact:



{income_gain:+.1f}%

"""

)



# --------------------------------

# DEBT CARD

# --------------------------------



st.error(

    f"""

⚠ DEBT INCREASE



Current Debt:

₹{debt:,}



New Debt:

₹{min(debt + 200000, 800000):,}



Approval Probability:



{current_probability:.1f}% → {debt_prob:.1f}%



Impact:



{gain_debt:+.1f}%

"""

)



# --------------------------------

# SMART AI RECOMMENDATION ENGINE

# --------------------------------



st.markdown("---")

st.subheader("🤖 AI Recommendation Engine")



improvements = {

    "Increase Credit Score (+50)": gain_credit,

    "Increase Income (+₹20,000)": gain_income,

    "Avoid Additional Debt (+₹200,000 Debt Impact)": abs(gain_debt)

}



best_action = max(

    improvements,

    key=improvements.get

)



best_gain = improvements[best_action]



# Strong Profile



if current_probability >= 70:



    st.success(

        f"""

Strong Financial Profile Detected



Current Approval Probability:

{current_probability:.2f}%



Recommendation:



Maintain current financial behaviour.



No major improvements are currently required.

"""

    )



# Medium Profile



elif current_probability >= 35:



    st.warning(

        f"""

Medium Risk Profile Detected



Current Approval Probability:

{current_probability:.2f}%



Best Improvement Opportunity:



{best_action}



Potential Improvement:



+{best_gain:.1f}%

"""

    )



# High Risk Profile



else:



    st.error(

        f"""

High Risk Profile Detected



Current Approval Probability:

{current_probability:.2f}%



Most Important Action:



{best_action}



Potential Improvement:



+{best_gain:.1f}%

"""

    )



# Detailed Recommendations



st.markdown("### Suggested Actions")



if credit_score < 700:

    st.write("✔ Improve credit score by at least 50 points")



if income < 80000:

    st.write("✔ Increase monthly income")



if savings < 150000:

    st.write("✔ Build higher savings reserves")



if debt > 200000:

    st.write("✔ Reduce existing debt obligations")



if employment_years < 5:

    st.write("✔ Maintain longer employment history")



if dependents > 2:

    st.write("✔ Manage financial commitments carefully")



# --------------------------------

# EXPLAINABLE AI

# --------------------------------



st.markdown("---")

st.subheader("🧠 Why Did The AI Make This Decision?")



positive_factors = []

risk_factors = []



# Positive Factors



if credit_score >= 700:

    positive_factors.append(

        f"Strong Credit Score ({credit_score})"

    )



if income >= 80000:

    positive_factors.append(

        f"Healthy Income (₹{income:,})"

    )



if savings >= 150000:

    positive_factors.append(

        f"Strong Savings (₹{savings:,})"

    )



if debt <= 200000:

    positive_factors.append(

        f"Low Existing Debt (₹{debt:,})"

    )



if employment_years >= 5:

    positive_factors.append(

        f"Stable Employment History ({employment_years} years)"

    )



if employment == "Employed":

    positive_factors.append(

        "Currently Employed"

    )



# Risk Factors



if credit_score < 700:

    risk_factors.append(

        f"Credit Score Needs Improvement ({credit_score})"

    )



if income < 80000:

    risk_factors.append(

        f"Income Below Preferred Level (₹{income:,})"

    )



if savings < 150000:

    risk_factors.append(

        f"Savings Could Be Higher (₹{savings:,})"

    )



if debt > 200000:

    risk_factors.append(

        f"High Existing Debt (₹{debt:,})"

    )



if employment_years < 5:

    risk_factors.append(

        f"Short Employment History ({employment_years} years)"

    )



if employment == "Unemployed":

    risk_factors.append(

        "Currently Unemployed"

    )



if dependents > 2:

    risk_factors.append(

        f"Higher Number Of Dependents ({dependents})"

    )



# DISPLAY



col1, col2 = st.columns(2)



with col1:



    st.success("✅ Positive Factors")



    if len(positive_factors) == 0:

        st.write("No major positive factors identified.")



    for item in positive_factors:

        st.write("✔", item)



with col2:



    st.error("⚠ Risk Factors")



    if len(risk_factors) == 0:

        st.write("No major risk factors identified.")



    for item in risk_factors:

        st.write("⚠", item)



# --------------------------------

# DOWNLOAD REPORT

# --------------------------------



st.markdown("---")

st.subheader("📄 Generate Decision Report")



report_text = f"""

AI DECISION TWIN REPORT

========================



APPLICANT DETAILS

\-----------------



Age: {age}



Monthly Income: ₹{income:,}



Credit Score: {credit_score}



Loan Amount: ₹{loan_amount:,}



Employment Status: {employment}



Savings: ₹{savings:,}



Debt: ₹{debt:,}



Employment Years: {employment_years}



Dependents: {dependents}



\--------------------------------



AI DECISION RESULT



Approval Probability:

{current_probability:.2f}%



Decision:

{decision}



Risk Level:

{risk}



\--------------------------------



AI RECOMMENDATIONS



"""



for item in positive_factors:

    report_text += f"\n+ {item}"



for item in risk_factors:

    report_text += f"\n- {item}"



report_text += f"""



\--------------------------------



Generated By:

AI Decision Twin



Powered By:

Random Forest Machine Learning

"""



st.download_button(

    label="📥 Download Full Decision Report",

    data=report_text,

    file_name="AI_Decision_Twin_Report.txt",

    mime="text/plain"

)
