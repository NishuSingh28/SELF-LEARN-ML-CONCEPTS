---
layout: default
title: "REGRESSION IN TIME SERIES ANALYSIS"
date: 2025-09-28
---

# Time Series Regression: Residuals, Degrees of Freedom, and R²

## 1. What is Residual Standard Error (RSE)?

In regression or time series modeling, the **residuals** are the differences between the observed values and the model’s predicted values:

$$
e_t = y_t - \hat{y}_t
$$

The **Residual Standard Error (RSE)** is essentially the standard deviation of these residuals.

**Formula:**

$$
\hat{\sigma}^2 = \frac{\sum e_t^2}{n-k}
$$

Where:  
- $n$ = number of observations  
- $k$ = number of estimated parameters in the model  

### 1.1 Why do we look at RSE in Time Series?

1. **Measure of model fit**  
   - RSE indicates how much the actual data points deviate, on average, from the model’s fitted values.  
   - Smaller RSE → predictions are closer to actual data.

2. **Scale-aware error**  
   - Unlike $R^2$, RSE is in the same units as the dependent variable $y_t$.  
   - Example: If modeling daily sales, RSE of 120 means the model is off by ~120 units on average.

3. **Model comparison**  
   - Comparing two models (e.g., AR(2) vs ARIMA(1,1,1)), the one with the lower RSE is generally a better fit.

4. **Check adequacy of residuals**  
   - Large RSE relative to mean of $y_t$ → poor predictive power.  
   - RSE close to measurement noise → model captures most of the signal.

5. **Diagnostic for overfitting/underfitting**  
   - RSE decreases too much with more parameters → potential overfitting.  
   - Stable RSE across validation sets → model generalizes well.

---

## 2. What are Degrees of Freedom (DF)?

Degrees of freedom measure how many independent pieces of information are available to estimate variability.

**General formula (in regression/time series):**

$$
\text{DF} = n - k
$$

Where:  
- $n$ = number of observations  
- $k$ = number of estimated parameters (including intercept, lags, seasonal terms, etc.)

### 2.1 Why Degrees of Freedom Matter in Time Series

1. **Corrects for model complexity**  
   - Residual variance (RSE) is calculated as:

   $$
   \hat{\sigma}^2 = \frac{\sum e_t^2}{n-k}
   $$

   - Not $\sum e_t^2 / n$  
   - DF correction prevents underestimating error when fitting many parameters.

2. **Prevents overconfidence in estimates**  
   - Lower DF → fewer “free” data points → higher uncertainty in estimates.  
   - Encourages avoiding overfitting.

3. **Influences statistical tests**  
   - t-tests, F-tests, Ljung–Box tests all depend on DF for significance.  
   - Ignoring DF → incorrect p-values.

4. **Model comparison**  
   - Criteria like AIC/BIC penalize models with fewer DF (too many parameters).  
   - Models with too few DF may overfit the training data.

5. **Interpretation of residuals**  
   - DF adjustment ensures residual variance is unbiased.

### 2.2 What happens if you ignore DF?

- Fitting too many parameters reduces residuals artificially.  
- Residual variance $\sum e_t^2 / n$ looks smaller than reality → misleadingly good fit.  
- DF correction exposes artificial improvements.

### 2.3 Why DF correction matters

- Without DF correction, complex models always look “better” (smaller residual variance).  
- DF adjustment balances **fit vs parsimony**.

---

## 3. What is Multiple R-squared ($R^2$)?

In regression and time series models, **$R^2$** measures the proportion of variability in the dependent variable $y_t$ explained by the model.

**Formula:**

$$
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
$$

Where:  
- $SS_{res} = \sum (y_t - \hat{y}_t)^2$ → residual sum of squares  
- $SS_{tot} = \sum (y_t - \bar{y})^2$ → total variation in data  

**Interpretation:**  
- $R^2 = 0$ → model explains nothing  
- $R^2 = 1$ → perfect fit

### 3.1 Why “Multiple” R²?

- **Simple regression** (one predictor) → $R^2$ = squared correlation between predictor & response  
- **Multiple regression** (several predictors, e.g., AR/MA terms in time series) → generalized → “Multiple $R^2$”

### 3.2 Why do we look at Multiple R² in Time Series?

1. **Measure of goodness-of-fit**  
   - Indicates how well the model explains variation in the observed series.  
   - Example: $R^2 = 0.75$ → 75% of variance in $y_t$ explained by the model.

2. **Diagnostic for model adequacy**  
   - Low $R^2$ → model fails to capture signal  
   - High $R^2$ → model captures most variation (beware overfitting)

3. **Model comparison**  
   - Compare AR(1) vs AR(5), ARIMA vs regression  
   - $R^2$ alone increases with more parameters → check Adjusted R² or AIC/BIC as well

4. **Intuitive communication**  
   - $R^2$ provides percentage-based interpretation:  
     > “Our ARIMA model explains 83% of the variability in daily demand.”

---

## 4. Problem with R²

- Adding more predictors always increases (or at least never decreases) $R^2$, even if the predictor is irrelevant.  
- Example: Adding many random AR lags can increase $R^2$, but the model is overfitting.  
- $R^2$ can be misleading in complex models (ARIMA with too many AR/MA terms).

---

## 5. Adjusted R²

Adjusted R² penalizes extra parameters to prevent overfitting.

**Formula:**

$$
R^2_{adj} = 1 - \frac{SS_{res}/(n-k-1)}{SS_{tot}/(n-1)}
$$

Where:  
- $n$ = number of observations  
- $k$ = number of predictors (not counting intercept)  

**Key difference:** Adjusted R² divides by **degrees of freedom** instead of just using sums of squares.

### 5.1 Intuition

- If a new predictor improves the model → Adjusted R² goes up.  
- If the predictor adds noise (overfitting) → Adjusted R² goes down.  
- Adjusted R² balances **fit vs complexity**.

---

## 6. What are AIC and BIC?

Both are **information criteria** that balance:

1. **Goodness of fit** → how well the model explains the data  
2. **Model complexity** → number of parameters used  

### 6.1 Formulas

**Akaike Information Criterion (AIC):**

$$
\text{AIC} = -2 \ln(\hat{L}) + 2k
$$

**Bayesian Information Criterion (BIC):**

$$
\text{BIC} = -2 \ln(\hat{L}) + k \ln(n)
$$

Where:  
- $\hat{L}$ = maximum likelihood of the model  
- $k$ = number of estimated parameters  
- $n$ = number of observations  

### 6.2 Key differences

- Both start with a fit term $-2 \ln(\hat{L})$ → smaller if fit is good.  
- Both add a penalty for complexity:  
  - AIC penalty = $2k$  
  - BIC penalty = $k \ln(n)$  
- Since $\ln(n) > 2$ when $n > 7$, BIC penalizes complexity more heavily than AIC for large datasets.

### 6.3 Why do we look at them in Time Series?

- **Model selection** among ARIMA candidates (AR(1), AR(2), ARMA(1,1), ARIMA(2,1,2), etc.)  
- Compute AIC/BIC for each and choose the model with the lowest value  
- Balance **fit and parsimony** → prevents overfitting  
- Models with lower AIC/BIC usually generalize better out-of-sample

### 6.4 Intuition

- AIC/BIC = “Fit score” + “Complexity penalty”  
- Best model = lowest total cost

### 6.5 Choosing between AIC vs BIC

- **AIC:** more forgiving, favors better fit → good for prediction  
- **BIC:** harsher penalty, favors simpler models → good for identifying true underlying process

---

## 7. What are residuals in Time Series?

Residuals = difference between observed values and model’s fitted values:

$$
e_t = y_t - \hat{y}_t
$$

After fitting a time series model (AR, ARMA, ARIMA, SARIMA, etc.), residuals should behave like **white noise**:  
- Mean ≈ 0  
- Constant variance  
- No autocorrelation  

### 7.1 Why residual diagnostics matter

Even if the model has:  
- High $R^2$  
- Low AIC/BIC  
- Small residual standard error  

> The model is **not adequate** if residuals still contain structure (patterns, autocorrelation, seasonality).  

Structured residuals indicate the model has left “signal” unexplained that could have been captured.

---
# Time Series Residual Diagnostics, Ljung–Box Test, F-Statistic, Multicollinearity, AIC/BIC

## 8. Tools for Residual Diagnostics

### 8.1 Plot Residuals
- Residuals should look like a random scatter around 0.  
- Any visible pattern (trends, cycles) indicates the model hasn’t captured all structure.

### 8.2 Autocorrelation Function (ACF) of Residuals
- Compute sample autocorrelations of $e_t$.  
- For a correct model:
  - All autocorrelations should be close to 0  
  - None should be statistically significant (outside confidence bands)  
- Significant lags remaining indicate underfitting (missed AR/MA terms or seasonality)

### 8.3 Ljung–Box Test
- Formal statistical test for autocorrelation in residuals  
- **Null hypothesis:** residuals are independently distributed (white noise)  

**Test statistic:**

$$
Q = n(n+2) \sum_{k=1}^h \frac{\hat{\rho}_k^2}{n-k}
$$

Where:  
- $\hat{\rho}_k$ = sample autocorrelation at lag $k$  
- $h$ = number of lags tested  
- $n$ = number of observations  

- If p-value < 0.05 → reject null → residuals are not white noise → model inadequate

### 8.4 Histogram / Q-Q Plot of Residuals
- Check if residuals are approximately normal (useful for prediction intervals)  
- Not strictly necessary for point forecasts but helps with inference

### 8.5 Workflow
1. Fit an ARIMA model  
2. Check residual plots  
3. Plot ACF of residuals  
4. Run Ljung–Box test at multiple lags (e.g., 10, 20)  
5. If residuals fail → refine the model (add terms, difference, or seasonal component)

### 8.6 Example Intuition
- Fit AR(1) on data that’s actually AR(2)  
- Model R² = 0.80 looks okay  
- Residual ACF shows significant correlation at lag 2  
- Ljung–Box p-value < 0.01 → AR(1) is inadequate → try AR(2)

---

## 9. Ljung–Box Test

### 9.1 What is it?
- Checks whether autocorrelations in residuals are jointly zero  
- Determines if residuals of a model are white noise

### 9.2 Why do we need it?
- A good model explains all predictable structure; leftover residuals should be random  
- Residual autocorrelation indicates missing structure (e.g., wrong AR/MA order or seasonality)

### 9.3 Hypotheses
- **Null ($H_0$):** residuals are independently distributed (white noise)  
- **Alternative ($H_1$):** residuals are autocorrelated (model inadequate)

### 9.4 Test Statistic
$$
Q = n(n+2) \sum_{k=1}^{h} \frac{\hat{\rho}_k^2}{n-k}, \quad Q \sim \chi^2(h - p - q)
$$

Where:  
- $p, q$ = AR and MA orders of the model  
- $h$ = number of lags tested

### 9.5 Interpretation
- p-value < 0.05 → reject $H_0$, residuals autocorrelated → model inadequate  
- p-value > 0.05 → fail to reject $H_0$, residuals look like white noise → model adequate

### 9.6 Example
- Fit AR(1) on data that’s actually AR(2)  
- Residuals show significant correlation at lag 2  
- Ljung–Box p-value = 0.01 → reject $H_0$ → AR(1) inadequate → try AR(2)

### 9.7 Practical Implementation

**In R:**
```r
# Ljung-Box test for residuals at lag 20
Box.test(residuals, lag=20, type="Ljung-Box")
```

**In Python (statsmodels):**
```python
from statsmodels.stats.diagnostic import acorr_ljungbox

# Ljung-Box test for residuals at lags 10 and 20
acorr_ljungbox(residuals, lags=[10, 20], return_df=True)
```

- This checks if residuals are uncorrelated across multiple lags

### 9.8 Summary
- Confirms whether ARIMA/ARMA model has captured all structure  
- Guides model refinement if residuals fail the test  
- Ensures residuals behave like white noise

---

## 10. F-Statistic in Regression

- Tests if predictors collectively explain variation in the dependent variable  
- Formula:

$$
F = \frac{\text{MSR}}{\text{MSE}} = \frac{\text{SSR}/p}{\text{SSE}/(n-p-1)}
$$

Where:  
- MSR = Mean Square Regression = SSR / p  
- MSE = Mean Square Error = SSE / (n - p - 1)  
- SSR = Regression sum of squares  
- SSE = Error sum of squares  
- n = number of observations  
- p = number of predictors  

- Large F → predictors likely significant  
- Small F → predictors may not be useful

### 10.1 Hypotheses
- $H_0$: $\beta_1 = \beta_2 = \dots = \beta_p = 0$ (no predictive power)  
- $H_a$: at least one $\beta_j \neq 0$ (model useful)  
- Reject $H_0$ if F > critical value or p-value < significance level

---

## 11. Multicollinearity

### 11.1 What is it?
- Two or more predictors are highly correlated  
- Makes it difficult to determine which predictor is responsible for changes in response

### 11.2 Why it inflates standard errors
$$
\text{Var}(\hat{\beta}) = \sigma^2 (X^\top X)^{-1}
$$

- Nearly linear dependence → $X^\top X$ nearly singular → $(X^\top X)^{-1}$ large → variances of coefficients inflate

### 11.3 Effect on p-values
- Inflated SE → t-statistics shrink → p-values increase → harder to reject $H_0: \beta_j = 0$  
- Consequences:
  - Type II errors  
  - Wide confidence intervals  
  - Unstable inference  
  - Potential underfitting

---

## 12. AIC and BIC

### 12.1 Why we need AIC
- Multiple candidate models; more complex models may overfit  
- AIC balances fit and complexity to select the most parsimonious predictive model

### 12.2 Formulas
- **AIC:**
$$
AIC = -2 \ln(L) + 2k
$$
- **Corrected AIC (AICc):**
$$
AIC_c = AIC + \frac{2k(k+1)}{n-k-1}
$$
- **BIC:**
$$
BIC = -2 \ln(L) + k \ln(n)
$$

Where:  
- $L$ = likelihood  
- $k$ = number of parameters  
- $n$ = number of observations

### 12.3 Intuition
- AIC → focus on predictive accuracy  
- BIC → focus on selecting the true model (consistency)  
- Small datasets → AICc penalizes extra parameters more heavily

### 12.4 Rule of Thumb
- Prediction goal → prefer AIC  
- Model selection/inference → prefer BIC

## 13. Ex-Ante vs Ex-Post Forecasting

### 13.1 What is an Ex-Ante Forecast?

**Q:** What does “ex-ante” mean?  
**A:** Literal: “Before the fact.”

**Q:** How is it done in practice?  
You forecast the dependent variable (e.g., ice cream sales) for the future.  
Problem: Future predictor values (e.g., temperature) are unknown.  
Solution: Use forecasts of predictors (e.g., weather forecast for Tuesday).

**Key feature:**  
- Realistic, implementable forecast  
- Includes uncertainty from both your model and predictor forecasts

**Example:**  
- Monday night: predict Tuesday’s ice cream sales  
- Temperature forecast: \(30^\circ C \pm 2^\circ C\)  
- Model predicts: \(500 \pm \text{some error}\)  
- Ex-ante forecast: \(500 \pm \text{combined uncertainty}\)

---

### 13.2 What is an Ex-Post Forecast?

**Q:** What does “ex-post” mean?  
**A:** Literal: “After the fact.”

**Q:** How is it done?  
- Wait until the actual future is known  
- Plug actual observed values of predictors into your model

**Key feature:**  
- Not a true forecast — assumes perfect knowledge of predictors  
- Isolates model performance itself, ignoring errors from forecasting predictors

**Example:**  
- Wednesday: Tuesday’s actual temperature was \(32^\circ C\)  
- Plug \(32^\circ C\) into sales model → predicted sales = 520 cones  
- Compare with actual sales → measure model accuracy without predictor uncertainty

---

### 13.3 Why Distinguish Ex-Ante vs Ex-Post?

| Scenario | Ex-Ante Error | Ex-Post Error | Interpretation |
|----------|---------------|---------------|----------------|
| Low ex-post, high ex-ante | High real-world error | Low model error | Model good; predictor forecasts are bad |
| High ex-post, high ex-ante | High real-world error | High model error | Model itself is flawed; even perfect predictor values give bad predictions |
| Low ex-post, low ex-ante | Low real-world error | Low model error | Ideal: both model and predictor forecasts are accurate |

**Intuition:**  
- Ex-post = perfect information diagnostic  
- Ex-ante = real-world actionable forecast  
- Comparing both diagnoses errors: predictor forecasts vs model specification

---

### 13.4 Practical Strategies for Ex-Ante Forecasting

1. **Forecast your predictors first**  
   - Use ARIMA, ML, or other time series models to predict predictors themselves  
   - Example: forecast temperature before predicting ice cream sales

2. **Use scenario analysis**  
   - Produce forecasts under different plausible predictor values  
   - Example: sales forecast if temperature is 28°C, 30°C, or 32°C

3. **Incorporate uncertainty**  
   - Use prediction intervals to reflect both model error and predictor uncertainty  
   - Example: “We expect 500 ± 50 cones”

4. **Regularly update forecasts**  
   - As new predictor data arrives, update the ex-ante forecast

---

### 13.5 Scenario-Based Forecasting

**Q:** What is it?  
- Instead of a single “best guess” for predictor values, create plausible future scenarios for predictors and see how the dependent variable responds

**Q:** Why useful?  
- Future is uncertain  
- Provides range of possible outcomes  
- Helps decision-makers plan for optimistic, pessimistic, and baseline cases

**Example (conceptual):**  
- Predict consumption based on income  
  - Optimistic: Income ↑ 1% → predicted consumption under this scenario  
  - Pessimistic: Income ↓ 1% → predicted consumption under this scenario

---

### 13.6 Using Lagged Predictors

* **Problem:** A key challenge in **ex-ante forecasting** is that future values of predictor variables are unknown.

* **Solution:** To address this, we use **lagged predictors**, which are the known past values of these variables.

---

#### Mathematical Transformation

**Original contemporaneous model:**
$$y_t = \beta_0 + \beta_1 x_t + \epsilon_t$$

**Lagged model:**
$$y_t = \beta_0 + \beta_1 x_{t-1} + \epsilon_t$$

**Prediction for $t+1$:**
$$\hat{y}_{T+1} = \hat{\beta}_0 + \hat{\beta}_1 x_T$$

---

#### Intuition 

* At the current time $T$, the value of the predictor $x_T$ is already known, eliminating the need to forecast the predictors.
* This makes ex-ante forecasting more **feasible 

# Prediction Intervals, Nonlinearity, and Regression Transformations

## 14. Understanding Prediction Intervals (PIs)

### 14.1 What is a Prediction Interval?
- A **prediction interval (PI)** is a range of plausible future values for the dependent variable, accounting for uncertainty.
- Example: A 95% PI means the true value is expected to fall within this range 95% of the time.
- Analogy:
  - Point forecast: “It will take exactly 25 minutes.”
  - Prediction interval: “It will take between 20–30 minutes with 95% confidence.”

### 14.2 Sources of Uncertainty in a Regression PI
The PI formula for a new observation is:

$$
\hat y_{\text{new}} \pm 1.96 \cdot \hat{\sigma}_e \sqrt{ 1 + \frac{1}{T} + \frac{(x_{\text{new}} - \bar x)^2}{(T-1)s_x^2} }
$$

Sources of uncertainty:  
1. **Irreducible error** ($\epsilon_t$): inherent randomness; cannot be predicted.  
2. **Intercept uncertainty** ($1/T$): only estimated $\beta_0$; more data reduces this.  
3. **Slope uncertainty** $\frac{(x_{\text{new}} - \bar x)^2}{(T-1)s_x^2}$: larger when $x_{\text{new}}$ is far from historical mean $\bar x$.

**Intuition:**  
- Extrapolating far from historical data → wider PI  
- Extreme predictor values → more risk → PI expands

---

### 14.3 Regression Setup

**Simple regression model:**

$$
y_i = \beta_0 + \beta_1 x_i + \epsilon_i
$$

Where:  
- $y_i$ = dependent variable  
- $x_i$ = predictor  
- $\beta_0, \beta_1$ = regression coefficients  
- $\epsilon_i$ = random error  

Estimate $\beta_0$ and $\beta_1$ from data.

**Predicted value for a new $x_0$:**

$$
\hat y_0 = \hat \beta_0 + \hat \beta_1 x_0
$$

---

### 14.4 Confidence Interval (CI) for the Mean Response

Formula:

$$
CI: \hat y_0 \pm t_{n-2,0.975} \cdot SE(\hat y_0)
$$

Where:

$$
SE(\hat y_0) = \hat \sigma \sqrt{\frac{1}{n} + \frac{(x_0 - \bar x)^2}{\sum (x_i - \bar x)^2}}
$$

**Key points:**  
- CI predicts where the **mean response** is likely.  
- Does **not** include individual observation variability.  
- Narrower than PI.  
- Analogy: “Where is the regression line likely to be?”

---

### 14.5 Prediction Interval (PI) for a Single Observation

Formula:

$$
PI: \hat y_0 \pm t_{n-2,0.975} \cdot \sqrt{SE(\hat y_0)^2 + \hat \sigma^2}
$$

**Key points:**  
- Includes both:
  1. Uncertainty in the regression line  
  2. Random variation of individual observations  
- Wider than CI.  
- Analogy: “Where will a new observation fall around the line?”

---

### 14.6 Comparison Table

| Step | Formula | Concept |
|------|---------|---------|
| 1 | $\hat y_0 = \hat \beta_0 + \hat \beta_1 x_0$ | Predicted mean response |
| 2 | $CI: \hat y_0 \pm t \cdot SE(\hat y_0)$ | Uncertainty in mean estimate |
| 3 | $PI: \hat y_0 \pm t \cdot \sqrt{SE(\hat y_0)^2 + \hat \sigma^2}$ | Uncertainty in mean + individual variation |
| 4 | Width comparison | PI > CI |
| 5 | Interval assignment | Narrow → CI, Wide → PI |

**Example:**  
- Interval A: [105, 115] → narrower → CI for mean  
- Interval B: [100, 120] → wider → PI for single observation  

**Takeaway:**  
- CI → average outcome → narrower  
- PI → single future outcome → wider  

---

## 15. Why a Straight Line Doesn’t Always Fit

### 15.1 Nonlinear Relationships
- Some relationships are nonlinear.  
- Example: Sales vs. Advertising:
  - Low ad spend → large sales increase  
  - High ad spend → diminishing returns  

Linear model:

$$
y = \beta_0 + \beta_1 x + \epsilon
$$

- Won’t capture curved trends → biased predictions

---

### 15.2 Approach 1: Data Transformations (“Lens” Method)

**Goal:** Make curved relationships linear.  

**Common transformations:**

1. **Log-Log Model:**

$$
\log(y) = \beta_0 + \beta_1 \log(x)
$$`

- % change in $x$ → % change in $y$  
- Example: Price vs. quantity (elasticity)

2. **Log-Linear Model:**

$$
\log(y) = \beta_0 + \beta_1 x
$$`

- Unit change in $x$ → % change in $y$  
- Example: Investment growth over time

3. **Linear-Log Model:**

$$
y = \beta_0 + \beta_1 \log(x)
$$`

- % change in $x$ → unit change in $y$  
- Example: Crop yield vs. fertilizer  

> **Tip:** If $x=0$ occurs, use $\log(x+1)$ to avoid undefined values.

---

### 15.3 Approach 2: Piecewise Regression (“Bendy Ruler” Method)

- Useful when slope changes at certain points (knots).  

**Mathematics (knot at $c$):**

$$
x_2 = (x-c)^+ =
\begin{cases} 
0 & x < c \\ 
x-c & x \geq c 
\end{cases}
$$

Model:

$$
y = \beta_0 + \beta_1 x + \beta_2 x_2
$$

- Before knot ($x < c$): slope = $\beta_1$  
- After knot ($x \ge c$): slope = $\beta_1 + \beta_2$  
- $\beta_2$ = change in slope at knot

**Why not polynomials?**  
- Quadratic/cubic can predict extreme values outside data → unrealistic  
- Piecewise linear is more stable

---

### 15.4 Example: Boston Marathon Times

- Linear trend: straight line  
- Exponential trend: $\log(\text{Minutes}) \sim \text{trend()}$  
- Piecewise trend: slope changes at knots (e.g., 1950, 1980)

**R code (fpp3):**

```r
fit_trends <- boston_men |>
  model(
    linear = TSLM(Minutes ~ trend()),
    exponential = TSLM(log(Minutes) ~ trend()),
    piecewise = TSLM(Minutes ~ trend(knots = c(1950, 1980)))
  )
```

**Mathematical interpretation:**

- Before 1950: slope = $\beta_1$  
- Between 1950–1980: slope = $\beta_1 + \beta_2$  
- After 1980: slope = $\beta_1 + \beta_2 + \beta_3$  

**Benefit:** Captures different eras realistically without extreme extrapolation.

# 16. Exercises: Piecewise Trends, Correlation, and Multicollinearity

## 16.1 Exercise: Piecewise Linear Trend

**Model:**  
$$
\text{Sales} \sim \text{trend(knots = 2022)}
$$  

**Coefficients:**

- $\beta_0 = 500$  
- $\beta_1 = 20$  
- $\beta_2 = -15$  

**Growth rate calculation:**

- **Before 2022:** slope = $\beta_1 = 20$ → +20 units/month  
- **After 2022:** slope = $\beta_1 + \beta_2 = 20 - 15 = 5$ → growth slowed  

**Math check:** piecewise regression adds the change in slope to the original slope after the knot.

---

## 16.2 Exercise: Australian Air Passengers

**Task:** Model 1970–2011 data with a knot at 1989 (pilot strike)

**R Code:**

```r
fit_air <- aus_airpassengers |>
  model(piecewise_trend = TSLM(Passengers ~ trend(knots = 1989)))
fc_air <- forecast(fit_air, h = 10)
```

**Interpretation:**

- Before 1989: slope = $\beta_1$  
- After 1989: slope = $\beta_1 + \beta_2$ (change due to strike)  
- Forecast extends the last linear piece → realistic future prediction  

**Benefit:** Piecewise linear trend captures sudden structural changes better than a single straight line.

---

## 16.3 Key Takeaways for Trend Transformations

| Topic         | Formula / Concept                    | Intuition |
|---------------|------------------------------------|-----------|
| Log-Log       | $\log(y) = \beta_0 + \beta_1 \log(x)$ | % change in $x$ → % change in $y$ |
| Log-Linear    | $\log(y) = \beta_0 + \beta_1 x$       | Unit change in $x$ → % change in $y$ |
| Linear-Log    | $y = \beta_0 + \beta_1 \log(x)$       | % change in $x$ → unit change in $y$ |
| Piecewise     | $y = \beta_0 + \beta_1 x + \beta_2 (x-c)^+$ | Slope changes at knot $c$ |
| Polynomial danger | $y = \beta_0 + \beta_1 x + \beta_2 x^2$ | Can explode in forecasts → unrealistic |
| Forecast      | Extend last linear piece              | Stable extrapolation |

---

## 16.4 Part 1: Correlation ≠ Causation

**Correlation:**  
$$
\rho_{X,Y} = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y}
$$  

- $\rho \approx +1$ → variables move together  
- $\rho \approx -1$ → variables move oppositely  

**Important:** Correlation does **not imply causation**.  

### Examples

**1. Ice Cream & Drownings:**  

- Hot weather → more ice cream  
- Hot weather → more swimming → more drownings  
- Correlation $\text{Corr}(\text{Ice Cream}, \text{Drownings}) > 0$  
- Insight: Useful for forecasting even if not causal

**2. Cyclists & Rain:**  

- Fewer morning cyclists → predictor of afternoon rain  
- Causal arrow: Forecast of rain → fewer cyclists  
- Regression: Cyclists (x) → Rain (y)  
- Insight: Reverse causality does not affect forecast usefulness

---

## 16.5 Part 2: Multicollinearity

**Definition:**  
- Predictors strongly correlated with each other  
- Formally:  
$$
\text{Corr}(x_1, x_2) \approx \pm 1
$$  
- Makes regression coefficients unstable

**Analogy:** Two students submit almost identical book reports → high overall fit, but individual credit is ambiguous.

**Effect on coefficients:**  

- Predictions $\hat y$ may still be accurate  
- Coefficients $\beta_1, \beta_2$ unstable → small data changes → big swings in estimates

**Problematic Scenarios:**

| Situation                               | Problem? |
|-----------------------------------------|-----------|
| Interpret coefficients (e.g., GDP effect) | 🔴 YES |
| Scenario forecasting with correlated predictors | 🔴 YES |
| Extrapolating outside historical ranges  | 🔴 YES |
| Accurate predictions inside known ranges | 🟢 NO  |

---

### 16.6 Detecting Multicollinearity

**Method 1: Correlation Matrix (R)**

```r
predictors <- longley[, c("GNP.deflator", "GNP", "Unemployed", "Armed.Forces", "Population", "Year")]
cor_matrix <- round(cor(predictors), 2)
print(cor_matrix)
```

- Example: GNP vs Year = 1.00, GNP.deflator vs GNP = 0.99 → severe multicollinearity

**Method 2: Variance Inflation Factor (VIF)**

$$
VIF_j = \frac{1}{1 - R_j^2}
$$  

- $R_j^2$ = R² from regressing predictor $x_j$ on other predictors  
- VIF = 1 → no multicollinearity  
- VIF > 5 → concerning  
- VIF > 10 → serious problem  

**R Code:**

```r
fit <- lm(Employed ~ ., data = longley)
library(car)
vif(fit)
```

- Example output: GNP = 1788, Population = 399, Year = 759 → extremely high

---

### 16.7 Conceptual Exercise

**Scenario:** Manager sees coefficient for Advertising = -500 with high VIF  

**Answer:**  
- High VIF → unstable coefficient  
- Advertising and Salesforce_Size move together → regression cannot separate effect  
- Negative sign is misleading → cutting ads based on this is wrong

---

### 16.8 Coding Exercise (mtcars)

**Test:** mpg ~ disp + hp + wt

```r
fit_cars <- lm(mpg ~ disp + hp + wt, data = mtcars)

car_predictors <- mtcars[, c("disp", "hp", "wt")]
print(round(cor(car_predictors), 2))

library(car)
print(vif(fit_cars))
```

**Observations:**  

- Correlations: disp vs wt = 0.89, disp vs hp = 0.79 → high  
- VIF: disp = 10.5, wt = 5.3, hp = 4.4  

**Interpretation:**  

- Multicollinearity present  
- Predictions okay  
- Cannot interpret individual coefficients reliably (e.g., hp)  

## 16.8 Correlation vs Causation

- **Correlation ≠ Causation**  
- But correlation can still help **forecasting**.

### Forecasting vs Causal Explanation

- **Forecasting:** detective (look for clues)  
- **Causal inference:** scientist (prove cause)

### Multicollinearity

- Doesn’t harm forecasting accuracy (usually)  
- But destroys coefficient interpretability  
- Always check correlations and VIF

---

## 16.9 Variance Inflation Factor (VIF)

### Step 1: Why do we care?

In linear regression:

$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p + \varepsilon
$$

- Coefficient estimates $\hat{\beta}_j$ have variance  
- Highly correlated predictors → variance inflates → unstable  
- **VIF** measures this inflation

### Step 2: Formula

For predictor $x_j$:

$$
\text{VIF}_j = \frac{1}{1 - R_j^2}
$$

- $R_j^2$ = R-squared from regressing $x_j$ on all other predictors  
- $R_j^2 = 0 \Rightarrow \text{VIF}_j = 1$  
- $R_j^2 = 1 \Rightarrow \text{VIF}_j = \infty$

### Step 3: Connection to Variance

Variance of OLS coefficient:

$$
\text{Var}(\hat{\beta}_j) = \frac{\sigma^2}{SST_j (1 - R_j^2)}
$$

- $\sigma^2$ = error variance  
- $SST_j = \sum (x_j - \bar{x}_j)^2$ = total variation of $x_j$  
- $(1 - R_j^2)$ penalizes variance due to multicollinearity

### Step 4: Intuition with Numbers

- If $R_1^2 = 0.95$, then  
$$
\text{VIF}_1 = \frac{1}{1 - 0.95} = 20
$$
- Variance of $\hat{\beta}_1$ is 20× larger than if independent  
- **Interpretation:** sign/size of $\hat{\beta}_1$ unreliable

### Step 5: Rule of Thumb

- VIF = 1 → no correlation  
- VIF = 2–5 → moderate  
- VIF > 10 → severe multicollinearity

### Step 6: Analogy

- One suspect → easy (VIF=1)  
- Two suspects always together → impossible to separate responsibility (VIF → ∞)  
- Higher VIF → harder to assign credit

---

## 17. Matrix Algebra in Regression

### Q1: Why bother?

- Ordinary regression:  
$$
y_t = \beta_0 + \beta_1 x_{1,t} + \dots + \beta_k x_{k,t} + \varepsilon_t, \quad t = 1, \dots, T
$$  
- T equations, k+1 unknowns → writing individually is tedious  
- **Matrix form:**  
$$
y = X \beta + \varepsilon
$$

Where:  
- $y$ = outcomes vector $(T \times 1)$  
- $X$ = design matrix with intercept + predictors $(T \times (k+1))$  
- $\beta$ = coefficient vector $((k+1) \times 1)$  
- $\varepsilon$ = error vector $(T \times 1)$

- Compact representation enables variance analysis, calculus, and predictions

### Q2: Building Blocks

- **Response vector:**  
$$
y = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_T \end{bmatrix}, \quad (T \times 1)
$$

- **Design matrix:**  
$$
X = 
\begin{bmatrix} 
1 & x_{1,1} & \dots & x_{k,1} \\ 
1 & x_{1,2} & \dots & x_{k,2} \\ 
\vdots & \vdots & \ddots & \vdots \\ 
1 & x_{1,T} & \dots & x_{k,T} 
\end{bmatrix}, \quad (T \times (k+1))
$$

- **Coefficient vector:**  
$$
\beta = \begin{bmatrix} \beta_0 \\ \beta_1 \\ \vdots \\ \beta_k \end{bmatrix}, \quad ((k+1) \times 1)
$$

- **Error vector:**  
$$
\varepsilon = \begin{bmatrix} \varepsilon_1 \\ \varepsilon_2 \\ \vdots \\ \varepsilon_T \end{bmatrix}, \quad (T \times 1)
$$

### Q3: Estimate $\beta$ (OLS)

Minimize squared errors:

$$
\hat{\beta} = (X^\top X)^{-1} X^\top y
$$

- Normal Equation  
- $X^\top y$ = correlation of predictors with outcome  
- $(X^\top X)^{-1}$ adjusts for overlap → best linear unbiased estimate

### Q4: Why multicollinearity breaks regression

- Perfect collinearity → $X^\top X$ singular → inverse does not exist  
- High correlation → $(X^\top X)^{-1}$ has large entries  
- Coefficient variances blow up → unstable signs/magnitudes

### Q5: Fitted Values

$$
\hat{y} = X \hat{\beta} = H y, \quad H = X (X^\top X)^{-1} X^\top
$$

- Hat matrix $H$: symmetric, idempotent  
- Diagonal entries $h_t$ = leverage

### Q6: Leverage

- $h_t$ measures distance of $x_t$ from predictor center  
- High leverage → observation heavily influences its fitted value  
- Rule of thumb:  
$$
h_t > \frac{2(k+1)}{T} \Rightarrow \text{high leverage}
$$

### Q7: Connection to Cross-Validation

- LOOCV shortcut via hat matrix:

$$
\hat{y}_{(-t)} = \hat{y}_t - \frac{e_t}{1 - h_t}
$$

- CV error:  
$$
CV = \frac{1}{T} \sum_{t=1}^T \left( \frac{e_t}{1 - h_t} \right)^2
$$
