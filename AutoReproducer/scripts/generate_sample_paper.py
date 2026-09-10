# -*- coding: utf-8 -*-
"""生成一个极简、可控的论文 PDF 样本，确保 AutoReproducer 端到端复现成功。

设计原则：
- 单一算法（最小二乘线性回归），无歧义
- 仅 numpy 一个外部依赖
- 指标单一明确：验证集 MSE
- 代码完整、可直接执行
- PDF 格式朴素（纯文本为主），保证 PyPDF2/pdfplumber 能高质量解析
"""
import sys
from pathlib import Path

try:
    from fpdf import FPDF
except ImportError as e:
    print("请先安装 fpdf2: pip install fpdf2", file=sys.stderr)
    sys.exit(1)


class PDF(FPDF):
    def header(self):
        # 极简页眉，不干扰解析
        pass

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")


def generate(out_path: Path) -> None:
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    # 使用系统自带的 Arial（Windows 常见），确保中文不乱码
    pdf.set_font("Arial", "", 12)

    # ---- 1. 标题 ----
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 12, "Linear Regression on Synthetic Data: A Minimal Reproducible Example", ln=True, align="C")
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, "AutoReproducer Sample Paper", ln=True, align="C")
    pdf.cell(0, 8, "2024", ln=True, align="C")
    pdf.ln(4)

    # ---- 2. 摘要 ----
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, "Abstract", ln=True)
    pdf.set_font("Arial", "", 11)
    abstract = (
        "This paper presents a minimal reproducible example of ordinary least squares (OLS) "
        "linear regression on a synthetic dataset. The goal is to provide a clean baseline "
        "with a single algorithm, a single metric, and minimal dependencies, suitable for "
        "automated reproduction pipelines."
    )
    pdf.multi_cell(0, 6, abstract)
    pdf.ln(2)

    # ---- 3. Method ----
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, "1. Method", ln=True)
    pdf.set_font("Arial", "", 11)
    method_text = (
        "We use ordinary least squares (OLS) to fit a linear model y = Xw + b. "
        "Given a design matrix X in R^(n x d) and target vector y in R^n, the closed-form "
        "solution is obtained by appending a column of ones to X and solving the normal equations.\n\n"
        "Formally, let X_b = [X, 1] in R^(n x (d+1)). The optimal parameters are:\n\n"
        "    w_b = (X_b^T X_b)^(-1) X_b^T y\n\n"
        "where w_b[-1] is the bias term b."
    )
    pdf.multi_cell(0, 6, method_text)
    pdf.ln(2)

    # ---- 4. Experimental Setup ----
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, "2. Experimental Setup", ln=True)
    pdf.set_font("Arial", "", 11)
    setup_text = (
        "Dataset: synthetic, generated with numpy.random.randn.\n"
        "  - Samples: n = 1000\n"
        "  - Features: d = 2\n"
        "  - Ground-truth weights: w = [2.5, -1.8]\n"
        "  - Ground-truth bias: b = 3.0\n"
        "  - Noise: Gaussian with std = 0.3\n"
        "  - Random seed: 42 (for reproducibility)\n\n"
        "Evaluation metric:\n"
        "  - Mean Squared Error (MSE) on the same synthetic dataset.\n"
        "  - Expected value: MSE approximately 0.09 (depends on random noise).\n\n"
        "Dependencies:\n"
        "  - Python 3.11+\n"
        "  - numpy\n"
    )
    pdf.multi_cell(0, 6, setup_text)
    pdf.ln(2)

    # ---- 5. Code Implementation ----
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, "3. Implementation (Python)", ln=True)
    pdf.set_font("Courier", "", 9)
    code = """import numpy as np

# reproducible synthetic data
np.random.seed(42)
n_samples = 1000
n_features = 2
X = np.random.randn(n_samples, n_features)
true_w = np.array([2.5, -1.8])
true_b = 3.0
noise = np.random.randn(n_samples) * 0.3
y = X @ true_w + true_b + noise

# OLS closed-form solution
X_b = np.c_[X, np.ones(n_samples)]
w_b = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
w_pred = w_b[:-1]
b_pred = w_b[-1]

# validation
y_pred = X @ w_pred + b_pred
mse = float(np.mean((y - y_pred) ** 2))

print(f"MSE={mse:.4f}")
print(f"w_pred={w_pred.round(4).tolist()}")
print(f"b_pred={b_pred:.4f}")
"""
    for line in code.strip().splitlines():
        pdf.cell(0, 5, line, ln=True)
    pdf.ln(2)

    # ---- 6. Results ----
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, "4. Results", ln=True)
    pdf.set_font("Arial", "", 11)
    results_text = (
        "Running the code above on a standard machine (Python 3.11, numpy 1.26+) "
        "produces the following expected output:\n\n"
        "    MSE=0.0892\n"
        "    w_pred=[2.5074, -1.7963]\n"
        "    b_pred=2.9876\n\n"
        "Note: the exact MSE may vary slightly (+-0.01) due to floating-point "
        "differences across numpy versions and platforms, but should remain below 0.15."
    )
    pdf.multi_cell(0, 6, results_text)
    pdf.ln(2)

    # ---- 7. Conclusion ----
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, "5. Conclusion", ln=True)
    pdf.set_font("Arial", "", 11)
    conclusion = (
        "This minimal example demonstrates a fully reproducible linear regression pipeline. "
        "With only numpy as a dependency and a single evaluation metric (MSE), it serves "
        "as an ideal sanity-check for automated reproduction systems."
    )
    pdf.multi_cell(0, 6, conclusion)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(out_path))
    print(f"[OK] Generated sample paper: {out_path}")


if __name__ == "__main__":
    out = Path("samples/paper/minimal_linear_regression.pdf")
    generate(out)
