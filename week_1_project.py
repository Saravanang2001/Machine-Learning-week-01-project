import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# Load your dataset
data = pd.read_csv("Global_Tech_Gadget_Consumption.csv")
df = pd.DataFrame(data)
# Example: Predict E-Waste Generated based on Smartphone Sales
X = df[['Smartphone Sales (Millions)']]   # Independent variable
y = df['E-Waste Generated (Metric Tons)'] # Dependent variable
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
# Get slope (m) and intercept (c)
m = model.coef_[0]
c = model.intercept_
print(f"Linear Regression Equation: E-Waste = {m:.2f} × Smartphone Sales + {c:.2f}")
# User input for prediction
user_sales = float(input("Enter Smartphone Sales (Millions): "))
user_sales_df = pd.DataFrame({'Smartphone Sales (Millions)': [user_sales]})
predicted_ewaste = model.predict(user_sales_df)
print(f"Predicted E-Waste Generated: {predicted_ewaste[0]:.2f} Metric Tons")