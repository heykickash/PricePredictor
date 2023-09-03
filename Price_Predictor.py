import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Function to perform polynomial regression and predict the price per unit
def predict_price(units_sold, selling_price, intended_units):
    # Sample data
    data = {
        'Units Sold': units_sold,
        'Selling Price': selling_price
    }

    # Separate the independent variable (X) and dependent variable (Y)
    X = np.array(data['Units Sold']).reshape(-1, 1)
    Y = np.array(data['Selling Price'])

    # Polynomial regression with a quadratic polynomial (you can adjust the degree)
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    # Fit the polynomial regression model
    model = LinearRegression()
    model.fit(X_poly, Y)

    # Predict the selling price per unit for the intended number of units
    predicted_price = model.predict(poly.transform([[intended_units]]))[0]

    return predicted_price

if __name__ == '__main__':
    # Input 10 pairs of units sold and selling price
    units_sold = []
    selling_price = []
    for i in range(10):
        units = float(input(f'Enter Units Sold {i + 1}: '))
        price = float(input(f'Enter Selling Price {i + 1}: '))
        units_sold.append(units)
        selling_price.append(price)

    # Input the intended number of units to sell
    intended_units = float(input('Enter the Intended Number of Units to Sell: '))

    # Predict the price per unit
    predicted_price_per_unit = predict_price(units_sold, selling_price, intended_units)

    # Print the predicted price per unit
    print(f'Predicted Selling Price per Unit for {intended_units} units: Rs {predicted_price_per_unit:.2f}')
