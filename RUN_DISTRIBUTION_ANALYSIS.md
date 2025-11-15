# 🚀 Quick Start: Distribution & Bias Analysis

## For Class Presentation

This guide helps you run the comprehensive distribution and bias analysis during your lecture.

---

## Pre-Class Setup (2 minutes)

### 1. Navigate to project folder
```bash
cd /home/kossiso-royce/CascadeProjects/loan_approval_analysis
```

### 2. Verify your dataset is in place
```bash
ls -lh data/loan_approval.csv
```
Should show: **58,644 rows** of data

### 3. Install dependencies (if not already done)
```bash
pip3 install -r requirements.txt
```

---

## Running the Analysis

### Option 1: Run Complete Analysis (Recommended)

```bash
cd scripts
python3 4_distribution_and_bias_analysis.py
```

**What happens:**
- ✅ Loads your 58K+ loan applications
- ✅ Analyzes class balance (Approved vs Rejected)
- ✅ Studies probability distributions for all numerical features
- ✅ Detects bias in categorical features (Gender, Home Ownership, etc.)
- ✅ Creates professional Matplotlib visualizations
- ✅ Generates comprehensive Markdown report

**Time:** ~2-3 minutes (depends on your computer)

**Output:**
- All visualizations saved in `outputs/distribution_bias_analysis/`
- Report saved as `DISTRIBUTION_BIAS_ANALYSIS_REPORT.md`

---

### Option 2: Run Previous Scripts First (Full Workflow)

If you want to show the complete data science pipeline:

```bash
cd scripts

# Step 1: Load and assess (5 min)
python3 1_load_and_assess.py

# Step 2: Basic distributions (5 min)
python3 2_distribution_analysis.py

# Step 3: Data cleaning (3 min)
python3 3_data_cleaning.py

# Step 4: Advanced distribution & bias analysis (3 min)
python3 4_distribution_and_bias_analysis.py
```

**Total time:** ~15-20 minutes

---

## During Class: What to Show

### Part 1: Introduction (5 min)

**Open and discuss:**
```bash
# Show the beginner's guide first
cat DISTRIBUTION_AND_BIAS_GUIDE.md
# Or open in text editor for better viewing
```

**Key concepts to explain:**
- What is a distribution?
- Types of distributions (Normal, Skewed, Uniform)
- What is class balance?
- What is bias and why it matters?

---

### Part 2: Run the Analysis (3 min)

```bash
cd scripts
python3 4_distribution_and_bias_analysis.py
```

**Watch the console output together** - it shows:
- Dataset size
- Class balance calculations
- Distribution analysis for each feature
- Bias detection results

---

### Part 3: Review Results (15-20 min)

#### 3.1 Open the generated report

```bash
cd ../outputs/distribution_bias_analysis
cat DISTRIBUTION_BIAS_ANALYSIS_REPORT.md
```

**Discuss with students:**
- Class balance ratio
- Which features are normally distributed?
- Which features show bias?

---

#### 3.2 Show visualizations one by one

**Start with class balance:**
```bash
# View the class balance plot
xdg-open 01_class_balance.png
# Or use your preferred image viewer
```

**Talk about:**
- How many approved vs rejected?
- Is the dataset balanced?
- Why does this matter for ML?

---

**Show distribution plots for key features:**

```bash
# Income distribution
xdg-open dist_income.png

# Age distribution
xdg-open dist_age.png

# Loan amount distribution
xdg-open dist_loan_amount.png
```

**For each, point out:**
- The histogram shape (normal? skewed?)
- The Q-Q plot (does it follow the line?)
- The box plot (any outliers?)
- The statistics table (mean, median, skewness)

---

**Show bias analysis plots:**

```bash
# Gender bias
xdg-open bias_Sex.png

# Home ownership bias
xdg-open bias_home_ownership.png

# Loan intent bias
xdg-open bias_loan_intent.png
```

**For each, discuss:**
- Overall distribution (top-left)
- Approval rates by category (top-right)
- Are differences significant (>15%)?
- What could cause this bias?
- How should we address it?

---

### Part 4: Key Concepts to Emphasize

#### 📊 Distribution Types

**Normal Distribution:**
- Bell-shaped, symmetric
- Mean ≈ Median
- Most common in nature
- Examples: Height, test scores

**Right-Skewed:**
- Long tail to the right
- Mean > Median
- Common in financial data
- Examples: Income, loan amounts

**Left-Skewed:**
- Long tail to the left
- Mean < Median
- Less common
- Example: Age at retirement

---

#### ⚖️ Class Balance

**Balanced (Good):**
- Ratio close to 1:1
- Model learns both classes equally

**Imbalanced (Problem):**
- Ratio > 3:1
- Model may ignore minority class
- Need special techniques

**Solutions:**
1. Oversampling (SMOTE)
2. Undersampling
3. Class weights
4. Ensemble methods

---

#### 🎯 Bias Detection

**What to look for:**
- Approval rate differences > 15%
- Systematic patterns
- Protected characteristics (gender, age)

**Red flags:**
- Gender: 70% vs 40% approval ⚠️
- Age groups: Young rejected more ⚠️
- Location: Urban favored over rural ⚠️

**Why it matters:**
- **Ethical**: Discrimination is wrong
- **Legal**: Can violate fair lending laws
- **Business**: Damages reputation
- **Technical**: Creates biased AI

---

## Teaching Tips

### ✅ Do This

1. **Show live coding**
   - Run the script while students watch
   - Let them see the output in real-time

2. **Pause for questions**
   - After each major section
   - Encourage discussion

3. **Use real examples**
   - "Imagine you're the loan officer..."
   - "How would you feel if..."

4. **Interactive elements**
   - Ask students to predict what they'll see
   - "What do you think this distribution will look like?"

5. **Connect to concepts**
   - Link visualizations to theory
   - "Remember when we talked about skewness? Here it is!"

---

### ❌ Avoid This

1. **Don't rush through plots**
   - Take time to explain each element
   - Point out what to look for

2. **Don't skip the "why"**
   - Not just "what is it" but "why does it matter"

3. **Don't ignore difficult topics**
   - Address bias head-on
   - Discuss ethical implications

4. **Don't assume prior knowledge**
   - Review concepts before diving in
   - Use the beginner's guide

---

## Discussion Questions

### For Class Discussion

**On Distributions:**
1. "Why might income be right-skewed instead of normal?"
2. "What does it mean if the Q-Q plot deviates from the line?"
3. "How would you transform a skewed distribution?"

**On Balance:**
1. "If 90% of loans are approved, what problem might this cause?"
2. "How would you fix a 10:1 class imbalance?"
3. "Can you think of real-world scenarios with imbalanced data?"

**On Bias:**
1. "Is it always wrong to have different approval rates by group?"
2. "How can we tell if bias is in the data or the real world?"
3. "What are the consequences of deploying a biased model?"
4. "How would you explain bias to a non-technical stakeholder?"

---

## Quick Reference: Files Generated

### In `outputs/distribution_bias_analysis/`:

**Reports:**
- `DISTRIBUTION_BIAS_ANALYSIS_REPORT.md` - Complete analysis summary

**Visualizations:**
- `01_class_balance.png` - Target variable balance
- `dist_[feature].png` - Distribution plots for each numerical feature
- `bias_[feature].png` - Bias analysis for each categorical feature

**Example files you'll see:**
```
01_class_balance.png
dist_age.png
dist_income.png
dist_loan_amount.png
dist_loan_interest_rate.png
dist_loan_income_ratio.png
dist_employment_length.png
dist_credit_history_length.png
bias_Sex.png
bias_Education_Qualifications.png
bias_home_ownership.png
bias_loan_intent.png
bias_payment_default_on_file.png
```

---

## Troubleshooting

### Problem: "File not found"
**Solution:**
```bash
# Make sure you're in the right directory
pwd
# Should show: .../loan_approval_analysis

# Check if data file exists
ls -l data/loan_approval.csv
```

---

### Problem: "Module not found"
**Solution:**
```bash
# Install missing packages
pip3 install pandas numpy matplotlib seaborn scipy
```

---

### Problem: "No display" (for plots)
**Solution:**
- Plots are saved as PNG files automatically
- No need to display them during script execution
- Open them manually from `outputs/` folder

---

### Problem: Script takes too long
**Solution:**
- Normal for 58K+ rows
- Should finish in 2-5 minutes
- If longer, check CPU usage

---

## After Class

### Student Exercises

**Beginner:**
1. Open each plot and identify the distribution type
2. List which features show bias
3. Calculate the class imbalance ratio manually

**Intermediate:**
1. Modify the script to analyze different features
2. Try different visualization styles
3. Add your own bias detection thresholds

**Advanced:**
1. Implement bias mitigation techniques
2. Create a model with fairness constraints
3. Compare biased vs debiased model performance

---

## Additional Resources

**In this project:**
- `DISTRIBUTION_AND_BIAS_GUIDE.md` - Comprehensive beginner's guide
- `DATA_DICTIONARY.md` - Dataset column descriptions
- `QUICK_START.md` - General project guide

**External:**
- Matplotlib gallery: https://matplotlib.org/stable/gallery/
- Seaborn tutorial: https://seaborn.pydata.org/tutorial.html
- Fairness in ML: https://fairmlbook.org/

---

## Summary Checklist

Before class:
- [ ] Dataset in place (`data/loan_approval.csv`)
- [ ] Dependencies installed
- [ ] Test run completed successfully
- [ ] All visualizations generated
- [ ] Read through beginner's guide

During class:
- [ ] Explain key concepts first
- [ ] Run analysis live
- [ ] Review report together
- [ ] Show and discuss each visualization type
- [ ] Address questions about bias and fairness

After class:
- [ ] Share generated report with students
- [ ] Provide access to visualization files
- [ ] Assign relevant exercises
- [ ] Encourage exploration

---

**Good luck with your lecture! 🎓📊**

*Remember: Take your time with each concept. Understanding is more important than speed!*
