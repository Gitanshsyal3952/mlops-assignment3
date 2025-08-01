# MLOps Assignment 3 – Main Branch

This repository demonstrates an MLOps pipeline:

- Training a machine learning model using **scikit-learn**
- Dockerizing the app
- Automating CI/CD using **GitHub Actions**
- Applying post-training **quantization**

## 📁 Folder Structure

```mlops-assignment3/
├── src/ # Source code files
│ ├── train.py
│ ├── predict.py
│ └── quantize.py
├── Dockerfile # Docker configuration
├── requirements.txt # Python dependencies
├── model.joblib # Trained model
├── .github/
│ └── workflows/
│ ├── ci.yaml
│ └── quantize.yaml
└── README.md # Project documentation```



---
## ⚙️ Run Locally

```bash
git clone https://github.com/gitanshsyal3952/mlops-assignment3.git
cd mlops-assignment3
python -m venv .venv
.venv\Scripts\activate  # For Windows
pip install -r requirements.txt
python src/train.py
python src/predict.py
python src/quantize.py
