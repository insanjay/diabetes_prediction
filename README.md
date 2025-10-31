# DiabetaPredict: Cloud-Native Diabetic Risk Assessment Platform

> **Future Feature:** The integrated AI Chatbot functionality is **scheduled for an upcoming release** and is not yet available.

### Table of Contents

1. [Overview](#overview)
2. [Tech Stack](#tech-stack)
3. [Local Setup](#local-setup-and-installation)
4. [Project Roadmap](#project-roadmap-future-enhancements)
5. [Author Details](#author)



**Note**: This readme and repository is only about the application side story, if you want to see the ML-side story follow the link below:

[ML-Workflow](https://github.com/insanjay/diabetes_prediction_workflow/tree/v2)

## Overview

DiabetaPredict is a high-availability web application designed to provide users with a risk assessment for diabetes based on key health metrics.

This project was developed to demonstrate skills in Python Backend Development, Microservice Architecture, and Cloud-Native Deployment (AWS) by successfully migrating a monolithic application into a decoupled, containerized, serverless stack.

---
To read more about this project follow the links below:

- [LinkedIn Post](link)
- [Medium Blog]([link](https://medium.com/@insanjay.work/the-developers-log-scaling-a-python-ai-service-past-aws-lambda-s-250mb-limit-b841918fd7f9))
- [Project’s SRS](link)
---
To see the demo of the working application follow the link below:

[YouTube Video] (link)

---

To use the live application, follow the links below:

- [Monolithic - Fully functional](https://heartsafev3.streamlit.app/)
- [Microservices Base - Partially Functional](https://heartsafev3-1.streamlit.app/)

---

**Note**: The Monolithic application has all the working functionalities (check the project SRS table no. x), but the Microservices base has only Auth and the Prediction Function working.

## Tech Stack

|Category | Technologies Used|
|---------|------------------|
|Backend & Core|Python, FastAPI, SQLAlchemy (ORM), Pydantic|
|Cloud & Deployment|AWS Lambda, AWS API Gateway, AWS ECR, Docker, CI/CD Principles|
|Data & Persistence|PostgreSQL, AWS RDS, Pandas, NumPy|
|Frontend|Streamlit|
|AI/ML|Scikit-learn, XGBoost, Hugging Face Inference API|

## Local Setup and Installation

This project is configured to run entirely locally using Docker Compose for a one-command setup of the Backend API and PostgreSQL database.

**Note**: If you encounter any issue on local setup create a pull request, I’ll try my best to fix the issue and push to GitHub ASAP. Or you can fix the issue and can push the changes to this repository, after confirmation the changes would be approved.

**Prerequisites**
1. **Git** (for cloning the repository)
2. **Docker** and **Docker Compose** installed on your machine.

**Steps**
1. **Clone the Repository:**
```
git clone [https://github.com/insanjay/DiabetaPredict.git](https://github.com/insanjay/DiabetaPredict.git)
cd DiabetaPredict
```

2. **Configure Environment Variables:**

Create a .env file in the root directory and define necessary secrets (e.g., database connection string, JWT secret key, Hugging Face API key).
```
# Example .env file content
SECRET_KEY="YOUR_JWT_SECRET_KEY"
DATABASE_URL="postgresql://user:password@db:5432/dbname"
HF_API_KEY="YOUR_HUGGING_FACE_KEY"
```

3. **Build and Run Services (Docker Compose):**
This command will build the Docker images, spin up the database and the FastAPI service, and automatically migrate the database schema.
```
docker-compose up --build -d
```

4. **Access the Application:**

- Backend API: `http://localhost:8000/docs` (FastAPI Swagger UI)
- Streamlit Frontend: Run the Streamlit app locally (assuming Streamlit is run outside the container, connecting to the API via `http://localhost:8000`).

## Project Roadmap Future Enhancements

The current version provides a stable, deployed foundation. Future development goals include:
- Full CI/CD Pipeline: Implementing automated testing and deployment workflows using GitHub Actions.
- Frontend Integration: Finalizing the frontend-backend integration for the AI Chatbot and User History display features.
- OAuth: Adding alternative login options (e.g., Google Sign-In) for improved user convenience.
- Model Management: Implementing infrastructure for regular ML model retraining and versioning.



## Author

- **Sanjay Kumar**
- [E-mail](mailto:insanjay.work@gmail.com)
- [LinkedIn](https://www.linkedin.com/in/insanjay)

📄 License

This project is provided under the MIT License.
