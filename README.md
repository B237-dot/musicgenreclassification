Dataset: Downloads the GTZAN Music Genre Classification Dataset (features_3_sec.csv) from Kaggle.

Libraries Used:

Data handling: numpy, pandas

Visualization: matplotlib, seaborn

ML preprocessing & evaluation: sklearn

Audio feature extraction: librosa

Workflow:

Dataset loading and exploration (df.head(), class distribution check).

Preprocessing of features for model training.

Model training using CNN (Convolutional Neural Network).

Activation function used: ReLU.

Evaluation using classification metrics.

Task: Music Genre Classification (predicting the genre of audio clips using extracted features).

Now, here’s your detailed README.md:

Music Genre Classification 🎵🎶

This project implements a machine learning and deep learning pipeline to classify music into different genres using the GTZAN Dataset. It leverages audio feature extraction and Convolutional Neural Networks (CNNs) for genre prediction.

📂 Dataset

The dataset used is the GTZAN Music Genre Classification Dataset, containing 10 genres with 100 audio tracks each. Features are extracted as 3-second segments and stored in features_3_sec.csv.

Source: Kaggle - GTZAN Music Genre Classification

Format: CSV (pre-extracted features)

Classes: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock

🚀 Features

Download dataset via KaggleHub

Data exploration and visualization of class distribution

Feature preprocessing and normalization

Training Convolutional Neural Network (CNN) for classification

Activation function: ReLU

Evaluation using metrics such as accuracy, confusion matrix

📂 Project Structure
├── mlactivity2.ipynb        # Main notebook
├── requirements.txt         # Python dependencies
└── README.md                # Documentation

⚙️ Installation

Clone the repository:

git clone https://github.com/sushmitha/mlactivity2.git
cd mlactivity2


Install dependencies:

pip install -r requirements.txt

▶️ Usage

Run the notebook:

jupyter notebook mlactivity2.ipynb


Steps inside the notebook:

Load dataset (features_3_sec.csv)

Preprocess data (scaling, splitting)

Train CNN model

Evaluate accuracy and visualize results

🛠️ Technologies Used

Python (NumPy, Pandas)

Scikit-learn (preprocessing, evaluation)

Librosa (audio feature handling)

Matplotlib / Seaborn (visualizations)

CNN (Keras/TensorFlow)

📊 Example Results

Achieved genre classification accuracy using CNN

Visualization of dataset distribution

Confusion matrix comparing predicted vs actual genres

🤝 Contribution

Contributions are welcome! Fork the repo and submit a pull request.
