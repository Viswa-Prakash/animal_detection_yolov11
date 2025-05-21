# 🐾 Animal Detection using YOLOv11

This repository implements an **Animal Detection System** using the **YOLOv11** object detection algorithm. The system is trained to detect and localize different animals in images using annotated datasets in the YOLO format.

## 📂 Dataset Description

It includes:

- **Train images**: Located in the `train/images` directory
- **Validation images**: Located in the `val/images` directory
- **Annotations**: Each image has a corresponding YOLO-style `.txt` file in `train/labels` and `val/labels`, containing bounding box coordinates and class labels.

### Classes:
The dataset includes the following animal classes:

```yaml
names:
- cat
- chicken
- cow
- dog
- fox
- goat
- horse
- person
- racoon
- skunk

---

## 📊 End-to-End Automated Pipeline

### 📥 1. Data Ingestion
- Annotated garbage images using [Roboflow](https://roboflow.com/) in YOLO format.
- Exported and downloaded as a `.zip` file.
- Extracted images and labels for training.

### ✅ 2. Data Validation
- Checks the presence of expected files and folders.
- Verifies consistency between images and label files.
- Training only proceeds if validation passes.

### 🔄 3. Data Transformation
- No transformation required.
- YOLOv11 handles resizing and augmentation during training internally.

### 🧠 4. Model Training
- Fine-tuned a pre-trained YOLOv11 model on the annotated dataset.
- Real-time object detection across the six garbage categories.
- YOLOv11 provides faster and more accurate detection due to fewer parameters and optimized architecture.

### 📤 5. Model Pusher
- Final trained model weights are saved and pushed to an AWS S3 bucket for safe storage and deployment.

### ☁️ 6. Deployment
- Deployed the entire pipeline on **AWS EC2** using:
  - **Docker** for containerization
  - **AWS ECR** for container registry
  - **GitHub Actions** for CI/CD pipeline

### 🌐 7. Web Application
- A simple **Flask** web app allows users to upload garbage images.
- Returns real-time predictions with bounding boxes and labels.

---

## 🛠️ Tech Stack

- **YOLOv11 (PyTorch)**
- **Roboflow** for image annotation
- **AWS EC2 / S3 / ECR**
- **Docker**
- **GitHub Actions** (CI/CD)
- **Flask** (Web Application)

---


## 🧾 Git Commands

```bash
git add .
git commit -m "Updated"
git push origin main

## ▶️ How to Run Locally

```bash
conda create -n animal python=3.10 -y
conda activate animal
pip install -r requirements.txt
python app.py

## ☁️ AWS CLI Configuration
```bash
aws configure
Add Access key ID,Secret access key

## 🚀 AWS CI/CD Deployment with GitHub Actions
🔐 1. IAM Setup
 - Log in to your AWS Console.
 - Create a new IAM user with programmatic access.
 - Assign the following policies:
  - AmazonEC2ContainerRegistryFullAccess
  - AmazonEC2FullAccess

📦 2. ECR (Elastic Container Registry)
 - Create an ECR repository to store Docker images.
 - Example ECR URI: 533267398036.dkr.ecr.us-east-1.amazonaws.com/garbageyolov5

 💻 3. EC2 (Elastic Compute Cloud)
 - Launch a new EC2 Ubuntu instance.
 - Connect to the instance and install Docker:
 
 ```bash
 # Optional
sudo apt-get update -y
sudo apt-get upgrade

# Required
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker


## 🧪 GitHub Actions Workflow
 ## 🏃‍♂️ 4. Configure EC2 as a Self-hosted Runner
 1. Go to your GitHub repo:
    Settings → Actions → Runners → New self-hosted runner

 2. Choose OS (Ubuntu) and follow the setup commands provided.
 - 🔑 5. Set GitHub Secrets
  - Go to Settings → Secrets and variables → Actions and add the following:
    - AWS_ACCESS_KEY_ID
    - AWS_SECRET_ACCESS_KEY
    - AWS_REGION = us-east-2
    - AWS_ECR_LOGIN_URI = 533267398036.dkr.ecr.us-east-1.amazonaws.com
    - ECR_REPOSITORY_NAME = garbageyolov5