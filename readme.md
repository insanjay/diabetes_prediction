# **DiabetaPredict** - Gender-Specific Diabetes Risk Assessment

> *An intelligent web application providing personalized diabetes risk prediction through advanced machine learning and gender-specific modeling approaches.*

---

## 🚀 **Live Demo**
🔗 **[Try DiabetaPredict Live](your-demo-url-here)** *(Coming Soon)*

---

## 📋 **Project Overview**

**DiabetaPredict** is a full-stack web application designed to provide users with personalized and accessible diabetes risk assessment. Unlike traditional one-size-fits-all approaches, this project features a complete user authentication system, personal history dashboard, and leverages two distinct, highly-tuned machine learning models—one optimized for male users and one for female users—to deliver more accurate, gender-specific predictions that account for different biological risk factors.

---

## ✨ **Key Features**

- **🔐 User Authentication**: Secure user registration and login system with password hashing
- **⚥ Gender-Specific Modeling**: Two separate, optimized prediction models for male and female users
- **📊 Personal History Dashboard**: Track prediction results and health metrics over time  
- **🧮 Integrated BMI Calculator**: Automatically calculates BMI from height and weight input
- **📈 Risk Score Visualization**: Percentage-based confidence scores with actionable insights
- **🛡️ Secure Data Storage**: SQLAlchemy ORM with proper user relationships and data protection
- **⚡ Real-time Predictions**: Cached models for fast, responsive predictions

---

## 🛠️ **Technology Stack**

| **Category** | **Technologies** |
|--------------|------------------|
| **Frontend** | Streamlit |
| **Backend & Database** | Python, SQLAlchemy (ORM), SQLite |
| **Authentication** | passlib (bcrypt password hashing) |
| **Machine Learning** | Scikit-learn, XGBoost, Pandas, NumPy |
| **Model Persistence** | Joblib |
| **Ensemble Methods** | Stacking Classifier |

---

## 📁 **Project Structure**

```
diabetes_prediction/
├── .gitignore
├── app.py
├── auth.py
├── readme.md
├── requirements.txt
├── database/
│   ├── __init__.py
│   ├── crud.py
│   ├── database_setup.py
│   ├── models.py
│   └── schemas.py
└── models/
    ├── female_model/
    │   ├── female_diabetes_model_info.json
    │   ├── female_feature_scaler.pkl
    │   └── female_final_ensemble_model.pkl
    └── male_model/
        ├── diabetes_ensemble_model_info.json
        ├── diabetes_label_encoder_final.pkl
        └── diabetes_stacking_ensemble_model.pkl
```
---

## 🚀 **Setup and Installation**

### Prerequisites
- Python 3.8+ installed
- Git installed

### Installation Steps

1. **Clone the repository:**
git clone https://github.com/insanjay/diabetes_prediction.git cd DiabetaPredict


2. **Create and activate virtual environment:**
python -m venv myenv

Windows
.\myenv\Scripts\activate

Linux/Mac
source myenv/bin/activate


3. **Install dependencies:**
pip install -r requirements.txt


> **⚠️ Version Compatibility Note:** If you encounter bcrypt/passlib compatibility issues, try:
> ```
> pip install bcrypt==3.2.2 passlib==1.7.4
> ```

4. **Run the application:**
streamlit run app.py


5. **Access the app:**
- Open your browser and navigate to `http://localhost:8501`
- The database (`diabetes_app.db`) will be created automatically on first run

---

## 🧠 **Model Details & Performance**

### **The Innovation: Gender-Specific Approach**
Instead of using a traditional one-size-fits-all model, **DiabetaPredict** employs two distinct machine learning pipelines to better capture the unique risk factors and biological differences between male and female populations.

### **Training Process**

1. **📊 Data Sourcing**: 
- **Male Dataset**: Custom-cleaned dataset (1,535 records)
- **Female Dataset**: Pima Indians Diabetes Dataset (768 records)

2. **⚖️ Handling Class Imbalance**: 
- Applied SMOTE (Synthetic Minority Over-sampling Technique) on training data
- Balanced diabetic vs. non-diabetic classes for improved minority class detection

3. **🤖 Model Architecture**:
- **Base Models**: XGBoost, Random Forest, SVM, Logistic Regression
- **Ensemble Methods**: Hard Voting, Soft Voting, Stacking
- **Best Approach**: **Stacking Ensemble** with Logistic Regression meta-learner

4. **🔬 Optimization Techniques**:
- 5-fold cross-validation for robustness
- Threshold tuning for clinical relevance
- Probability calibration using isotonic regression

### **Final Model Performance**

| **Model** | **Accuracy** | **Key Metrics** |
|-----------|--------------|-----------------|
| **Male Model** | **80.46%** | Optimized for high overall accuracy |
| **Female Model** | **70.13%** | Balanced: 57% Precision, 61% Recall for diabetic class |

> **Why Gender-Specific?** This approach, while more complex, results in more nuanced and reliable predictions tailored to individual biological differences and risk factors.

---

## 🎯 **Usage**

1. **Registration**: Create an account with your email and select your gender
2. **Input Health Data**: Enter age, height, weight, family history, and (for females) pregnancy history
3. **Get Predictions**: Receive instant risk assessment with confidence scores
4. **Track History**: View all your previous predictions in your personal dashboard
5. **Monitor Progress**: Track changes in your risk profile over time

---

## 🔮 **Future Enhancements**

- **🤖 AI-Powered Insights**: Integration with LLM for personalized health recommendations and Q&A
- **📱 Mobile App**: Native mobile application for iOS and Android
- **📧 Smart Notifications**: Risk alerts and health reminders
- **👥 Healthcare Professional Dashboard**: Tools for medical professionals to monitor patients
- **🌐 Multi-language Support**: Accessibility for diverse user populations

---

## 🤝 **Contributing**

We welcome contributions! Please feel free to:
- 🐛 Report bugs and issues
- 💡 Suggest new features  
- 🔧 Submit pull requests
- 📚 Improve documentation

---

## 📄 **License**

This project is licensed under the MIT License

---

## ⚠️ **Disclaimer**

This application is designed for educational and informational purposes only. **It is not intended to replace professional medical advice, diagnosis, or treatment.** Always consult with qualified healthcare professionals for medical concerns and before making health-related decisions.

---

## 👨‍💻 **Author**

**Your Name**
- GitHub: [@insanjay](https://github.com/insanjay)
- LinkedIn: [insanjay](https://linkedin.com/in/insanjay)
- Email: insanjay.work@gmail.com

---

## 🌟 **Acknowledgments**

- Pima Indians Diabetes Database contributors
- Open-source machine learning community
- Healthcare professionals who provided domain expertise

---

*Empowering early diabetes risk predictions for healthier lives through intelligent, personalized data science.*
