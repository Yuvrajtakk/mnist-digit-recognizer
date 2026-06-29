# MNIST Digit Recognition — 7 Algorithm Comparison

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Yuvrajtakk/mnist-digit-recognizer/blob/main/DigiReco.ipynb)

**Yuvraj Tak** | B.Tech CSE-AI, Anand International College of Engineering
**Internship:** AI/ML at Watsoo Express Pvt. Ltd. | Mentor: Ankit Gupta | June – August 2026

---

## About This Project

I built this to prove a single idea: **algorithm-problem fit matters more than implementation quality.**

The notebook trains seven fundamentally different algorithms on the same dataset, evaluates them on the identical test set, and ranks them. The 73.89% accuracy gap between Linear Regression and Logistic Regression — same data, same pipeline, same preprocessing — is that proof. One is the wrong tool. One is the right tool.

This is Phase 2 of my internship project, assigned by Ankit sir on **June 25, 2026** and completed on **June 29, 2026**. Phase 1 was four days of studying all seven algorithms from scratch — the math, the code, the visualization — before touching MNIST. This repo is the implementation that followed.

---

## The Journey

Ankit sir assigned this on June 25 with a clear brief: learn 7 ML algorithms, build a digit recognizer, evaluate everything.

I didn't write MNIST code first. I spent four days (June 25–28) building the foundation in a separate repo — working through each algorithm's theory with manual walkthroughs and Watsoo Express truck health data. Linear Regression, Logistic Regression, KNN, Decision Tree, Random Forest, Naive Bayes, SVM — each one from scratch, with the math worked out before touching sklearn. By June 28, I could explain what each algorithm assumes, where it excels, and where it fails.

On June 28, before writing any MNIST code, I designed the pipeline decisions deliberately:

**Dataset choice:** I chose sklearn's 8×8 MNIST (1,797 images) over the full 28×28 version (60,000 images). Not because it was easier, but because it was right. KNN computing Euclidean distance across 784 features on 60,000 samples would run for hours on CPU. SVM's optimization problem becomes impractical without a GPU. The curse of dimensionality makes all pairwise distances nearly uniform at that scale, which breaks nearest-neighbor reasoning anyway. The pipeline is identical at either resolution. I chose the resolution where the engineering made sense.

**Train/test split:** 70/30 with `random_state=42` locked in. The fixed seed is non-negotiable — every algorithm must see the exact same 540 test images or the comparison table is meaningless noise.

**Normalization:** StandardScaler applied once before all seven algorithms. KNN and SVM need it (distance-based). Decision Tree, Random Forest, and Naive Bayes are unaffected (threshold-based). Applied universally to keep the pipeline clean with no branching logic.

**Primary metric:** Recall. In real truck breakdown detection, a missed failure puts a truck on a highway where it breaks. A false alarm triggers an unnecessary maintenance check. These are not equally bad. I chose to measure what actually matters in the problem context.

**Linear Regression inclusion:** Deliberate. Not because it would perform well, but because it wouldn't. The gap between it and the classifiers proves the argument — fit matters more than execution.

On June 29, I built and completed the full MNIST pipeline in a single session in Google Colab, trained all seven algorithms, ran the full evaluation, and generated every visualization. The notebook uses a cooking analogy thread across all sections — Imports = laying tools on counter, Load data = opening the ingredient box, Preprocess = washing and chopping, Train = cooking 7 ways, Evaluate = food critic's scorecard. A deliberate design choice that makes the notebook readable beyond just ML practitioners.

---

## Pipeline Design Decisions

| Decision | Choice | Reasoning |
|---|---|---|
| **Dataset** | 8×8 MNIST (sklearn) | 1,797 images, 64 features. KNN and SVM become computationally impractical at 784 features on CPU. |
| **Split** | 70/30, `random_state=42` | Every algorithm must see the exact same test set. Fixed seed is mandatory for fair comparison. |
| **Normalization** | StandardScaler (universal) | Required for KNN and SVM (distance-based). Harmless for trees (threshold-based). Keeps pipeline clean. |
| **Primary Metric** | Recall | Missed failures cost more than false alarms. Recall measures what matters. |
| **Linear Regression** | Included deliberately | Proves algorithm-problem fit > implementation quality. Wrong tool on right problem = predictable failure. |

---

## Dataset

**1,797 handwritten digit images**
**64 features per image** (8×8 pixels, grayscale values 0–16)
**10 classes** (digits 0–9)
**Balanced:** 174–183 images per digit

### Sample Digits

![25 sample digits from the MNIST 8×8 dataset](evaluation%20images/01_sample_digits.png)
*25 sample images from the MNIST 8×8 dataset. Each image is 64 pixel values rendered as an 8×8 grayscale grid. Notice the variation in handwriting — the model learns from all of it.*

### Class Distribution

![Class distribution across all 10 digit classes](evaluation%20images/02_class_distribution.png)
*Class distribution across all 10 digit classes. Range: 174–183 images per digit. Balanced dataset — accuracy is a trustworthy metric here.*

---

## Results Summary

| Rank | Algorithm | Accuracy | Precision | Recall | F1 Score |
|:---:|---|---:|---:|---:|---:|
| 7 | Linear Regression | 23.15% | 26.33% | 23.15% | 22.36% |
| 6 | Naive Bayes | 78.70% | 83.25% | 78.70% | 78.42% |
| 5 | Decision Tree | 85.74% | 86.02% | 85.74% | 85.76% |
| 4 | Logistic Regression | 97.04% | 97.15% | 97.04% | 97.05% |
| 3 | Random Forest | 97.59% | 97.62% | 97.59% | 97.59% |
| 2 | KNN | 97.78% | 97.79% | 97.78% | 97.76% |
| **1** | **SVM** | **98.15%** | **98.17%** | **98.15%** | **98.15%** |

Hyperparameters tuned during build:
- Random Forest: 150 estimators (up from sklearn default 100)
- KNN: K=5 with StandardScaler applied
- SVM: RBF kernel, C=10, gamma='scale'

![Accuracy comparison across all 7 algorithms](evaluation%20images/05_accuracy_comparison.png)
*Accuracy comparison across all 7 algorithms. The gap between Linear Regression and everything else is the algorithm-problem fit argument made visual.*

---

## Algorithm Breakdown

### 7. Linear Regression — 23.15% | The Wrong Tool
A regression algorithm applied to a classification problem. Digit labels are categories, not measurements. A 3 is not mathematically halfway between 1 and 5. Raw output is forced through `clip(0, 9).astype(int)` to produce valid predictions — mechanically functional, fundamentally mismatched. This is here to prove the argument, not to compete.

### 6. Naive Bayes — 78.70% | Independence Assumption Breaks
Calculates probability assuming each pixel is independent. In images, this is false — neighboring pixels are always spatially correlated. Precision 83.25% but Recall 78.70% — a 4.55% gap. When it predicts, it's usually right. But it misses many real digits entirely. The violated assumption is the cost.

### 5. Decision Tree — 85.74% | Shallow Boundaries
Builds a yes/no flowchart from pixel thresholds up to `max_depth=10`. One tree finds real patterns but not all of them. Confusion matrix shows mistakes scattered across multiple digit pairs — not a focused failure, just limited capacity in a single tree.

### 4. Logistic Regression — 97.04% | Right Tool, Proper Foundation
Sigmoid function produces class probabilities. Built for classification from the ground up. The jump from Linear Regression (23.15%) to here (97.04%) is entirely explained by using the right tool. Same data, same preprocessing — the 73.89% gap is the algorithm difference.

### 3. Random Forest — 97.59% | Ensembles Win
150 independent decision trees, each trained on a random data and feature subset, all voting. Diversity of predictions cancels individual errors. Tuned from 100 to 150 estimators during build — gained 0.18%. Small but measurable. First hands-on hyperparameter experiment.

### 2. KNN — 97.78% | Neighbor Voting
Finds the 5 most similar training images and returns the majority digit label. No learned weights — just distance. Scaling is critical. Only 0.37% behind SVM (2 predictions out of 540). Two completely different mathematical approaches converging to nearly identical results — the most interesting finding in this project.

### 1. SVM — 98.15% | Winner
RBF kernel projects 64-dimensional pixel data into higher-dimensional space where complex digit boundaries become linearly separable. 10 mistakes out of 540. Built for exactly this problem — high-dimensional data, non-linear boundaries, maximum margin separation.

---

## Confusion Matrix Analysis

### SVM — Detailed View

![SVM confusion matrix](evaluation%20images/03_confusion_matrix_svm.png)
*SVM confusion matrix — 540 test images, 10 total mistakes. The 3–8–9 cluster accounts for most errors — digits that share curved strokes at 8×8 resolution.*

Of SVM's 10 mistakes, most fall in the 3–8–9 cluster. These digits share curved shapes in the lower half of the 8×8 grid. At this resolution, those curves blur together. This is a data resolution limitation, not a model failure.

### All 7 Algorithms — Side by Side

![All 7 confusion matrices](evaluation%20images/04_confusion_matrices_all_7.png)
*All 7 confusion matrices. Linear Regression (top-left): color scattered everywhere, no diagonal. SVM (bottom-right): razor-sharp diagonal, near-invisible off-diagonal. The visual distance between these two panels is the entire story of algorithm selection.*

The progression is clear:
- **Linear Regression** — scattered color, no structure, guessing in all directions
- **Naive Bayes** — diagonal emerging but significant off-diagonal noise
- **Decision Tree** — clearer diagonal, multiple systematic confusions visible
- **Logistic Regression** — bright diagonal, few off-diagonal spots
- **Random Forest** — near-clean diagonal, barely visible mistakes
- **KNN** — almost clean, a handful of scattered mistakes
- **SVM** — bright diagonal, off-diagonal nearly invisible

---

## Full Evaluation Summary

![Evaluation summary](evaluation%20images/evaluation_summary.png)
*Grouped bar chart showing all 4 metrics per algorithm, and color-coded comparison table. Green = above 95%. Yellow = 80–95%. Red = below 80%.*

The three zones tell the story. Red (Linear Regression, Naive Bayes): not trustworthy. Yellow (Decision Tree): useful but limited. Green (Logistic Regression, Random Forest, KNN, SVM): all viable. The choice between green-zone algorithms depends on context — interpretability, inference speed, and computational budget — not just accuracy.

---

## Key Findings

**1. SVM and KNN nearly tie.**
SVM wins by 0.37% — 2 predictions out of 540. Maximum margin boundary vs nearest-neighbor voting, converging to nearly the same answer. Both approaches are capturing something real about digit structure. This was the most unexpected result.

**2. Algorithm-problem fit matters more than implementation quality.**
Linear Regression (23.15%) vs Logistic Regression (97.04%). Same data, same pipeline, same preprocessing. The 73.89% gap is entirely explained by tool choice. This was the core hypothesis. It held completely.

**3. Hyperparameter tuning has measurable effects.**
100 → 150 estimators in Random Forest: +0.18% accuracy. Small, but real. Not all defaults are optimal.

**4. SVM's mistakes are a data problem, not a model problem.**
10 mistakes, mostly in the 3–8–9 cluster. Shared curved strokes become ambiguous at 8×8 resolution. At 28×28, these would be clearly distinct. Sometimes the data limits you before the model does.

**5. Naive Bayes' independence assumption costs 4.55% on recall.**
Precision 83.25%, Recall 78.70%. The gap is the cost of a violated assumption. On image data where pixels are always spatially correlated, this underperformance is predictable and explainable.

**6. The confusion matrix grid is more convincing than any number.**
The seven-panel progression from scattered chaos (Linear Regression) to clean diagonal (SVM) communicates what no accuracy table can — what it actually looks like when an algorithm understands the problem vs. when it doesn't.

---

## What I Learned

Building this in a single session after four days of foundation work made the difference. Every design decision had a reason because I understood what each algorithm actually does, not just how to call it.

I learned that `random_state=42` is not boilerplate — it's the difference between a fair comparison and meaningless numbers. Early in the build, running without a fixed seed gave me different test sets each run. The accuracy differences looked real. They weren't.

I learned that the gap between SVM and KNN (0.37%) is more interesting than SVM winning. The headline is "SVM is best." The real insight is "two fundamentally different approaches arrive at the same answer" — which means both are doing something right, not just one.

I learned that visual evidence carries more weight than numbers in isolation. The seven confusion matrices side by side communicate in one image what seven rows of a metrics table can't.

And I learned that theory-first works. Four days understanding algorithms before touching MNIST meant every line of code had a purpose. Jumping straight to code would have produced more code with less understanding.

---

## How to Run

1. Click the **Open In Colab** badge at the top
2. `Runtime` → `Run all`
3. Execution takes ~30–40 seconds on Colab's free CPU
4. 6 PNG files save automatically to the Colab file browser

**Dependencies:** numpy, matplotlib, scikit-learn — all pre-installed in Colab. No setup required.

---

## Repository Structure

```
mnist-digit-recognizer/
├── DigiReco.ipynb                      # Complete pipeline: load → train → evaluate
├── evaluation images/
│   ├── 01_sample_digits.png            # 25 sample digit images
│   ├── 02_class_distribution.png       # Class balance bar chart
│   ├── 03_confusion_matrix_svm.png     # SVM confusion matrix detail
│   ├── 04_confusion_matrices_all_7.png # All 7 confusion matrices side by side
│   ├── 05_accuracy_comparison.png      # Accuracy bar chart across all algorithms
│   └── evaluation_summary.png          # Grouped metrics + color-coded table
└── README.md
```

---

## Internship Context

**Company:** Watsoo Express Pvt. Ltd.
**Mentor:** Ankit Gupta
**Project assigned:** June 25, 2026
**Project completed:** June 29, 2026
**Phase 1 (learning foundation):** [ml-internship](https://github.com/Yuvrajtakk/ml-internship)

---

*Built with deliberation, not defaults.*