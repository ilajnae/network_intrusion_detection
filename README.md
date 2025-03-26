# Network Intrusion Detection

This project develops a **Network Intrusion Detection System (NIDS)** using machine learning to classify network traffic as either "normal" or "anomaly" (attack). It utilizes the **NSL-KDD dataset** and compares three models—**Random Forest Classifier**, **K-Nearest Neighbors (KNN)**, and **Support Vector Classifier (SVC)**—with Random Forest identified as the best-performing model for final predictions. The project was developed using Python in a Google Colab environment.

## Project Overview

The system performs binary classification on network traffic data to detect intrusions. It includes data preprocessing, model training, cross-validation, and evaluation. The NSL-KDD dataset, an improved version of the KDD Cup '99 dataset, simulates a military network environment (a US Air Force LAN) with a variety of intrusions.

### Key Features
- **Dataset**: NSL-KDD (`Train_data.csv` and `Test_data.csv`), included in the repository.
- **Models**: Random Forest, KNN, and SVC, with Random Forest selected as the optimal model.
- **Preprocessing**: StandardScaler for numerical features, LabelEncoder for categorical features (e.g., `protocol_type`, `service`, `flag`).
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix, and Classification Report.
- **Visualization**: Supports plotting (e.g., via Matplotlib and Seaborn) for exploratory data analysis.

## Installation

To set up and run this project locally or in Google Colab, follow these steps:

### Prerequisites
- Python 3.8+
- Jupyter Notebook (optional for local use) or Google Colab
- Required Python libraries (listed below)

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ilajnae/network_intrusion_detection.git
   cd network_intrusion_detection

2. **Install Dependencies**:
   - **Local Environment**: Install the required libraries using pip:
     ```bash
     pip install numpy pandas scikit-learn matplotlib seaborn jupyter
   - **Google Colab**: Libraries are pre-installed; simply run the notebook’s import cell.

3. **Dataset**:
   - The `Train_data.csv` and `Test_data.csv` files are included in the repository under the `Dataset/` folder. No additional download is required.

4. **Run the Project**:
   - **Local Jupyter Notebook**:
     ```bash
     jupyter notebook

    Open `network intrusion detection.ipynb` in your browser.
   - **Google Colab**:
     - Upload the notebook and dataset files to Google Colab.
     - Update the file paths in the notebook to match Colab’s environment (e.g., `/content/Train_data.csv`).


## Usage

1. **Run the Notebook**:
   - Open `network intrusion detection.ipynb` in Jupyter Notebook or Google Colab.
   - Execute the cells sequentially to:
     - Load and preprocess the NSL-KDD dataset.
     - Train and evaluate Random Forest, KNN, and SVC models.
     - Compare model performance using cross-validation (ROC-AUC) and test set metrics.
     - Use the Random Forest model to predict anomalies on the test dataset.

2. **Expected Output**:
   - Model performance metrics:
     - Random Forest: ~99.7% accuracy, near-perfect ROC-AUC (~0.9999).
     - KNN: ~99.2% accuracy, ROC-AUC (~0.9973).
     - SVC: ~96.4% accuracy, ROC-AUC (~0.9915).
   - Confusion matrices and classification reports for each model.
   - Predictions for the test dataset using Random Forest.

3. **Customization**:
   - Adjust model hyperparameters (e.g., `n_estimators` in Random Forest) to optimize performance.
   - Modify preprocessing steps (e.g., feature selection with RFE) for experimentation.

## Dataset

The **NSL-KDD dataset** simulates a typical US Air Force LAN blasted with multiple attacks, capturing raw TCP/IP dump data. It improves upon the KDD Cup '99 dataset by removing redundant records. Key details:
- **Training Data**: `Train_data.csv` (25,192 samples, 42 features).
- **Test Data**: `Test_data.csv`.
- **Features**: 41 features (3 qualitative: `protocol_type`, `service`, `flag`; 38 quantitative: e.g., `duration`, `src_bytes`) plus a `class` label (`normal` or `anomaly`).
- **Background**: Each connection is a sequence of TCP packets with a well-defined protocol, labeled as normal or a specific attack type.
- **Source**: [Kaggle: Network Intrusion Detection](https://www.kaggle.com/datasets/sampadab17/network-intrusion-detection).


## Results

The notebook evaluates three models:
- **Random Forest Classifier**: Best performer with ~99.7% accuracy and a mean ROC-AUC of ~0.9999 across 10-fold cross-validation.
- **K-Nearest Neighbors**: Achieves ~99.2% accuracy, ROC-AUC of ~0.9973.
- **Support Vector Classifier**: Yields ~96.4% accuracy, ROC-AUC of ~0.9915.

Random Forest is used for final predictions on the test dataset due to its superior performance.

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -m "Add feature"`).
4. Push to your fork (`git push origin feature-branch`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, feel free to reach out via GitHub Issues or contact me at [Anjali E](anjalinagaraju2004@gmail.com).
