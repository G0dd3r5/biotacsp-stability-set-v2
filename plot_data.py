import matplotlib.pyplot as plt
import pandas as pd

def plot_data(data: pd.DataFrame, columns: list):
    for column in columns:
        plt.plot(data[column], label=column)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Data Plot')
    plt.legend()
    plt.show()