import pandas as pd
import numpy as np

np.random.seed(42)

data = []

for _ in range(15000):

    age = np.random.randint(21, 60)

    income = np.random.randint(20000, 150000)

    credit_score = np.random.randint(300, 900)

    loan_amount = np.random.randint(50000, 1000000)

    employment = np.random.randint(0, 2)

    savings = np.random.randint(10000, 500000)

    debt = np.random.randint(0, 800000)

    employment_years = np.random.randint(0, 25)

    dependents = np.random.randint(0, 6)

    score = 0

    # Income
    score += np.interp(
        income,
        [20000, 150000],
        [0, 20]
    )

    # Credit Score
    score += np.interp(
        credit_score,
        [300, 900],
        [0, 25]
    )

    # Savings
    score += np.interp(
        savings,
        [10000, 500000],
        [0, 15]
    )

    # Debt Penalty
    score -= np.interp(
        debt,
        [0, 800000],
        [0, 20]
    )

    # Employment Years
    score += np.interp(
        employment_years,
        [0, 25],
        [0, 10]
    )

    # Employment Status
    if employment == 1:
        score += 5

    # Dependents Penalty
    score -= dependents

    # Loan affordability
    affordability = loan_amount / income

    if affordability < 5:
        score += 10
    elif affordability < 10:
        score += 5
    else:
        score -= 5

    # Convert score to probability
    probability = 1 / (1 + np.exp(-(score - 25) / 8))

    approved = np.random.choice(
        [0, 1],
        p=[1 - probability, probability]
    )

    data.append([
        age,
        income,
        credit_score,
        loan_amount,
        employment,
        savings,
        debt,
        employment_years,
        dependents,
        approved
    ])

df = pd.DataFrame(
    data,
    columns=[
        "age",
        "income",
        "credit_score",
        "loan_amount",
        "employment",
        "savings",
        "debt",
        "employment_years",
        "dependents",
        "approved"
    ]
)

df.to_csv(
    "loan_data.csv",
    index=False
)

print("New Realistic Dataset Created Successfully")