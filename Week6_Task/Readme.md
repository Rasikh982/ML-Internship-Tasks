# Diabetes Prediction App

This is a simple Python project that predicts whether a person has diabetes
or not, using a Logistic Regression model.

## Files

- `app.py` - main code file
- `diabetes.csv` - dataset used to train the model
- `README.md` - this file

## Dataset

The dataset has 768 rows and these columns:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome (0 = No Diabetes, 1 = Diabetes) - this is what we predict

## How the code works

1. Load the dataset using pandas
2. Split the data into training and testing parts
3. Scale the data (make all numbers on the same range)
4. Train a Logistic Regression model
   - `class_weight="balanced"` is used to handle class imbalance
     (this dataset has more "No Diabetes" cases than "Diabetes" cases)
5. Test the model and print:
   - Accuracy
   - Confusion Matrix
   - Classification Report
6. Predict a new sample (you can change the numbers to test your own data)

## How to run

1. Install the required library:

```
pip install pandas scikit-learn
```

2. Make sure `app.py` and `diabetes.csv` are in the same folder.

3. Run the code:

```
python app.py
```

## Notes

- Logistic Regression is used here because it is simple and easy to explain.
- You can try other models too (Decision Tree, Random Forest, SVM, KNN) by
  just changing the model line in `app.py`.
- Class imbalance is handled using `class_weight="balanced"`. You can also
  try SMOTE if you want more advanced handling.
