import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def comparison_chart(experimental_data, predicted_data):
    plt.figure(figsize=(8, 6))
    plt.scatter(experimental_data, predicted_data, color='blue', label='Predictions')
    plt.plot([0, 100], [0, 100], color='red', linestyle='--', label='Ideal Line')
    plt.xlabel('Experimental Data')
    plt.ylabel('Data Predicted by ANN')
    plt.title('Comparison of Experimental Data and ANN Predictions')
    plt.legend()
    plt.grid(True)
    plt.show()

def mae_chart(experimental_data, predicted_data):
    mae = np.abs(predicted_data - experimental_data)
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(mae)), mae, color='green')
    plt.xlabel('Samples')
    plt.ylabel('Absolute Error')
    plt.title('Mean Absolute Error (MAE) Chart')
    plt.grid(True)
    plt.show()

def correlation_plot(experimental_data, predicted_data):
    correlation_coefficient = np.corrcoef(experimental_data, predicted_data)[0, 1]
    plt.figure(figsize=(8, 6))
    plt.scatter(experimental_data, predicted_data, color='blue', label=f'Correlation Coefficient: {correlation_coefficient:.2f}')
    plt.plot([0, 100], [0, 100], color='red', linestyle='--', label='Ideal Line')
    plt.xlabel('Experimental Data')
    plt.ylabel('Data Predicted by ANN')
    plt.title('Correlation Plot')
    plt.legend()
    plt.grid(True)
    plt.show()

def nanoparticle_concentration_effect(concentrations, impedance_modulus):
    plt.figure(figsize=(8, 6))
    plt.plot(concentrations, impedance_modulus, marker='o', color='purple')
    plt.xlabel('Nanoparticle Concentration')
    plt.ylabel('Impedance Modulus')
    plt.title('Effect of Nanoparticle Concentration on Impedance Modulus')
    plt.grid(True)
    plt.show()

def histogram_of_errors(experimental_data, predicted_data):
    errors = predicted_data - experimental_data
    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=20, color='orange')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.title('Histogram of Errors')
    plt.grid(True)
    plt.show()

def residual_plot(experimental_data, predicted_data):
    residuals = predicted_data - experimental_data
    plt.figure(figsize=(8, 6))
    plt.scatter(predicted_data, residuals, color='blue')
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True)
    plt.show()

def box_plot_of_errors(experimental_data, predicted_data):
    errors = predicted_data - experimental_data
    plt.figure(figsize=(8, 6))
    plt.boxplot(errors, vert=False)
    plt.xlabel('Error')
    plt.title('Box Plot of Errors')
    plt.grid(True)
    plt.show()

def calculate_statistics(experimental_data, predicted_data):
    mae = np.mean(np.abs(predicted_data - experimental_data))
    mse = np.mean((predicted_data - experimental_data) ** 2)
    rmse = np.sqrt(mse)
    r_squared = np.corrcoef(experimental_data, predicted_data)[0, 1] ** 2

    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared: {r_squared:.2f}")

def main():
    print("Please select an option:")
    print("1. Use generated dummy data for demonstration")
    print("2. Read data from results.csv")
    
    choice = input("Enter the number of your choice: ").strip()
    
    if choice == '1':
        experimental_data = np.random.rand(50) * 100
        predicted_data = experimental_data + np.random.randn(50) * 5
        concentrations = np.linspace(0, 10, 10)
        impedance_modulus = 100 + 10 * concentrations - (concentrations ** 2)
    elif choice == '2':
        try:
            data = pd.read_csv('results.csv')
            experimental_data = data['actual'].values
            predicted_data = data['Predictions'].values
            concentrations = np.linspace(0, 10, 10)
            impedance_modulus = 100 + 10 * concentrations - (concentrations ** 2)
        except FileNotFoundError:
            print("File 'results.csv' not found. Exiting the program.")
            return
        except KeyError:
            print("The CSV file does not contain the required columns 'Predictions' and 'actual'. Exiting the program.")
            return
    else:
        print("Invalid choice. Exiting the program.")
        return
    
    while True:
        print("\nPlease select a chart or analysis to display:")
        print("1. Comparison Chart of Experimental Data and ANN Predictions")
        print("2. Mean Absolute Error (MAE) Chart")
        print("3. Correlation Plot")
        print("4. Effect of Nanoparticle Concentration Chart")
        print("5. Histogram of Errors")
        print("6. Residual Plot")
        print("7. Box Plot of Errors")
        print("8. Calculate Statistics")
        print("q. Quit")
        
        chart_choice = input("Enter the number of the chart or analysis you want to see (or 'q' to quit): ").strip().lower()
        
        if chart_choice == '1':
            comparison_chart(experimental_data, predicted_data)
        elif chart_choice == '2':
            mae_chart(experimental_data, predicted_data)
        elif chart_choice == '3':
            correlation_plot(experimental_data, predicted_data)
        elif chart_choice == '4':
            nanoparticle_concentration_effect(concentrations, impedance_modulus)
        elif chart_choice == '5':
            histogram_of_errors(experimental_data, predicted_data)
        elif chart_choice == '6':
            residual_plot(experimental_data, predicted_data)
        elif chart_choice == '7':
            box_plot_of_errors(experimental_data, predicted_data)
        elif chart_choice == '8':
            calculate_statistics(experimental_data, predicted_data)
        elif chart_choice == 'q':
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main()

