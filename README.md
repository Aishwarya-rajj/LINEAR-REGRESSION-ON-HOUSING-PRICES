**NAME:** AISHWARYA RAJ
**COMPANY:** CODETECH IT SOLUTIONS
**ID:** CT08ESQ
**DOMAIN** MACHINE LEARNING
**DURATION:** 20TH DECEMBER 2024 - 20TH JANUARY 2025
**MENTOR:**
# LINEAR-REGRESSION-ON-HOUSING-PRICES

## Project Overview

This project implements a linear regression model to predict housing prices using the California Housing dataset from Scikit-learn. The model uses features such as average rooms, average occupancy, latitude, median income, and house age to estimate housing prices.

## Objective

The goal of this project is to:

- Apply linear regression to predict housing prices.
- Utilize feature selection and data scaling to improve model performance.
- Visualize dataset correlations and evaluate model predictions.

## Dataset

The dataset used is the **California Housing** dataset, which contains information on various housing attributes and median house prices in different California districts.

### Features Used:

- **AveRooms** - Average number of rooms per household
- **AveOccup** - Average number of occupants per household
- **Latitude** - Geographical location
- **MedInc** - Median income in the area
- **HouseAge** - Average age of houses

### Target Variable:

- **Price** - Median house price in the district

## Technologies and Libraries Used:

- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

## Implementation

### Key Steps:

1. **Data Loading and Exploration**: The California Housing dataset is loaded using `fetch_california_housing()`. Basic exploratory data analysis (EDA) is conducted, including displaying summary statistics and visualizing feature correlations.

2. **Feature Selection**: A subset of features is selected based on correlation analysis.

3. **Data Preprocessing**:

   - Standard scaling is applied to normalize the features.

4. **Model Training and Evaluation**:

   - The data is split into training and test sets (80-20 split).
   - A Linear Regression model is trained on the training data.
   - Predictions are made on the test data.
   - Model performance is evaluated using Mean Squared Error (MSE) and R-squared (R2) scores.

5. **Visualization**:

   - A heatmap is generated to show feature correlations.
   - A scatter plot compares actual vs predicted housing prices.

## Results

- **Mean Squared Error (MSE):** Displays the error between actual and predicted values.
- **R-Squared (R2):** Measures the model's performance in explaining the variance of the data.
- Visualization shows how well the model's predictions align with actual prices.

## Project Structure

```
|-- california_housing_prediction
    |-- data_exploration.ipynb
    |-- linear_regression.py
    |-- results
        |-- feature_correlation_heatmap.png
        |-- actual_vs_predicted_plot.png
    |-- README.md
```

## How to Run the Project

1. Clone the repository:

```
git clone https://github.com/username/california_housing_prediction.git
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run the Python script:

```
python linear_regression.py
```

## Future Enhancements

- Implement Ridge and Lasso regression to prevent overfitting.
- Test performance with more complex models like Decision Trees and Random Forest.
- Deploy the model using Flask or FastAPI.

## Conclusion

This project demonstrates the effectiveness of linear regression for predicting housing prices and highlights the importance of feature selection and data scaling. Future improvements will focus on enhancing model accuracy and expanding the scope to more advanced models.

