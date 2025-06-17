import pandas as pd
import matplotlib.pyplot as plt

def check_class_distribution(filepath, label_column='label', plot=True):
    df = pd.read_csv(filepath)

    class_counts = df[label_column].value_counts()
    print("\nClass Distribution:\n")
    print(class_counts)

    if plot:
        class_counts.plot(kind='bar', title='Class Distribution', xlabel='Class', ylabel='Count')
        plt.grid(axis='y')
        plt.show()

if __name__ == "__main__":
    check_class_distribution('data/training/UNSW_NB15_training-set.csv')
    check_class_distribution('data/training/training_data_adasyn.csv')