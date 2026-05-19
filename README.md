# Personality Finder

A Machine Learning web application that predicts personality categories based on user input features.

Built using:

* Python
* Scikit-learn
* Streamlit
* Pandas
* NumPy

---

<img width="960" height="405" alt="image 2" src="https://github.com/user-attachments/assets/0ed043d4-c05c-433f-ad86-eae0ccc9a4a5" />
<img width="960" height="398" alt="image 1" src="https://github.com/user-attachments/assets/296ce7ab-f3ad-49bc-8c73-5f83bf15ab27" />
<img width="960" height="423" alt="image 3" src="https://github.com/user-attachments/assets/b3072eaf-099f-4e54-91ed-e9d4c3fd5416" />



## Features

* Predicts personality category using ML model
* Interactive Streamlit web interface
* Fast real-time predictions
* Clean and simple UI
* High model accuracy

---

## Model Performance

### Accuracy

**99.27%**

### Classification Report

| Class | Precision | Recall | F1-Score |
| ----- | --------- | ------ | -------- |
| 0     | 0.99      | 0.99   | 0.99     |
| 1     | 1.00      | 0.99   | 0.99     |
| 2     | 1.00      | 1.00   | 1.00     |

### Confusion Matrix

```text
[[1251    4    6]
 [  13 1413    0]
 [   6    0 1307]]
```

The model demonstrates excellent multiclass classification performance with minimal prediction errors.

---

## Technologies Used

* Python
* Streamlit
* Scikit-learn
* Pandas
* NumPy
* Pickle

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ikbalhussa1n/personality_finder.git
cd personality_finder
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

---

## Project Structure

```text
personality_finder/
│
├── app.py
├── model/
│   ├── model.pkl
│   ├── scaler.pkl
│   └── encoder.pkl
│
├── requirements.txt
├── README.md
└── dataset/
```

---

## Machine Learning Workflow

1. Data Collection
2. Data Preprocessing
3. Feature Engineering
4. Model Training
5. Model Evaluation
6. Streamlit Deployment

---

## Future Improvements

* Deploy on Streamlit Cloud
* Add visualization dashboard
* Improve UI/UX
* Add more personality categories
* Experiment with Deep Learning models

---

## Author

GitHub: https://github.com/ikbalhussa1n

---

## License

This project is open-source and available under the MIT License.
