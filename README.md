# Complete Pipeline: Adaptive LRI with Uncertainty Quantification σᵢ

## Objective

Evaluate the reliability of imputed values using:

- a **local uncertainty estimate (σᵢ)**
- an **adaptive mask (Mᵢ)**
- a **Local Reliability Index (LRI)** based on SHAP values

---

## Step 1 — Data Loading

**Dataset:** Diabetes dataset from **scikit-learn**

---

## Step 2 — MCAR Simulation

Introduce **random missing values** (MCAR) with rates ranging from **5% to 50%**.

---

## Step 3 — Random Forest Imputation + σᵢ Extraction

For each variable:

1. Train a **RandomForestRegressor**
2. Extract predictions from each tree in the forest
3. Compute:

\[
\sigma_i = \text{std}(\text{tree predictions})
\]

- **Final imputed value:** mean of the tree predictions

---

## Step 4 — Construction of the Adaptive Mask

For each imputed value:

\[
M_i = 1 - \left(\frac{\sigma_i}{\max(\sigma)}\right)
\]

**Interpretation:**

- **High uncertainty → low Mᵢ**
- **Low uncertainty → Mᵢ close to 1**

---

## Step 5 — SHAP Modeling

For each variable:

- Train a **Random Forest model**
- Compute **SHAP values**

---

## Step 6 — Adaptive LRI Computation

\[
LRI = \frac{\sum |SHAP_i| \times M_i}{\sum |SHAP_i|}
\]

**Interpretation:**

- **High LRI → reliable prediction**
- **Low LRI → uncertain prediction**

---

## Step 7 — Threshold Optimization

Automatic search for the **optimal threshold** using the **elbow method**.

---

## Step 8 — Multi-Scenario Analysis

Missing rates evaluated:

**5%, 10%, 15%, 20%, 25%, 30%, 35%, 40%, 45%, 50%**
