import matplotlib.pyplot as plt
import numpy as np

def comparison_chart():
    # Sample experimental and predicted data
    experimental_data = np.random.rand(50) * 100
    predicted_data = experimental_data + np.random.randn(50) * 5

    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(experimental_data, predicted_data, color='blue', label='Predictions')
    plt.plot([0, 100], [0, 100], color='red', linestyle='--', label='Ideal Line')
    plt.xlabel('Experimental Data')
    plt.ylabel('Data Predicted by ANN')
    plt.title('Comparison of Experimental Data and ANN Predictions')
    plt.legend()
    plt.grid(True)
    plt.show()

def mae_chart():
    # Sample experimental and predicted data
    experimental_data = np.random.rand(50) * 100
    predicted_data = experimental_data + np.random.randn(50) * 5

    # Mean Absolute Error
    mae = np.abs(predicted_data - experimental_data)

    # Bar chart
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(mae)), mae, color='green')
    plt.xlabel('Samples')
    plt.ylabel('Absolute Error')
    plt.title('Mean Absolute Error (MAE) Chart')
    plt.grid(True)
    plt.show()

def correlation_plot():
    # Sample experimental and predicted data
    experimental_data = np.random.rand(50) * 100
    predicted_data = experimental_data + np.random.randn(50) * 5

    # Correlation Coefficient
    correlation_coefficient = np.corrcoef(experimental_data, predicted_data)[0, 1]

    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(experimental_data, predicted_data, color='blue', label=f'Correlation Coefficient: {correlation_coefficient:.2f}')
    plt.plot([0, 100], [0, 100], color='red', linestyle='--', label='Ideal Line')
    plt.xlabel('Experimental Data')
    plt.ylabel('Data Predicted by ANN')
    plt.title('Correlation Plot')
    plt.legend()
    plt.grid(True)
    plt.show()

def nanoparticle_concentration_effect():
    # Sample concentrations and impedance modulus data
    concentrations = np.linspace(0, 10, 10)
    impedance_modulus = 100 + 10 * concentrations - (concentrations ** 2)

    # Line plot
    plt.figure(figsize=(8, 6))
    plt.plot(concentrations, impedance_modulus, marker='o', color='purple')
    plt.xlabel('Nanoparticle Concentration')
    plt.ylabel('Impedance Modulus')
    plt.title('Effect of Nanoparticle Concentration on Impedance Modulus')
    plt.grid(True)
    plt.show()

def main():
    while True:
        print("Please select a chart to display:")
        print("1. Comparison Chart of Experimental Data and ANN Predictions")
        print("2. Mean Absolute Error (MAE) Chart")
        print("3. Correlation Plot")
        print("4. Effect of Nanoparticle Concentration Chart")
        print("q. Quit")
        
        choice = input("Enter the number of the chart you want to see (or 'q' to quit): ").strip().lower()
        
        if choice == '1':
            comparison_chart()
        elif choice == '2':
            mae_chart()
        elif choice == '3':
            correlation_plot()
        elif choice == '4':
            nanoparticle_concentration_effect()
        elif choice == 'q':
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main()

