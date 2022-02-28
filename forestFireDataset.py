from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

'''
Description:
Class that represents a DAO object for the dataset and contains all the necessary functions around it.

Inputs: 
csv_file:String = Name of .csv file to read the data from.
encode:Boolean (optional) = Flag to either keep the target data as numerical or encode them and prepare them for classification.
'''


class ForestFireDataset():

    def __init__(self, csv_file, encode=False):
        self.data = pd.read_csv(csv_file)
        self.to_encode = encode

    def __len__(self):
        return len(self.data)

    # Using functional programming to process and prepare the data.
    def process_data(self):
        self.data.dropna()
        self.data.drop("damage_category", axis=1, inplace=True)  #
        self.data.reset_index()

        # Change categorical data to numerical
        self.data["month"].replace({'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                                    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12},
                                   inplace=True)
        self.data["day"].replace({'sun': 1, 'mon': 2, 'tue': 3, 'wed': 4, 'thu': 5, 'fri': 6, 'sat': 7}, inplace=True)

        # Split training data and targets
        X, Y = self.data.iloc[:, 8:12], self.data.iloc[:, 12:]

        # Generate extra data and append
        X = pd.concat([X, self.generate_data(X)], ignore_index=True)
        X.reset_index()

        gen_data = pd.DataFrame(self.generate_data(Y), columns=["area"])
        gen_data["area"] = gen_data["area"].apply(lambda x: 0.0 if x < 1.0 else x)
        Y = pd.concat([Y, gen_data], ignore_index=True)
        Y.reset_index()

        if (self.to_encode):
            Y = ForestFireDataset.one_hot_encoding_targets(Y)

            # Standarize data to have a mean of 0 and a standard deviation of 1.
        X = StandardScaler().fit_transform(X, Y)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=0.25, shuffle=True)

        if (self.to_encode):
            self.y_train = self.y_train.to_numpy().reshape(-1).astype(int)
            self.y_test = self.y_test.to_numpy().reshape(-1).astype(int)
        else:
            self.y_train = self.y_train.to_numpy().reshape(-1).astype(float)
            self.y_test = self.y_test.to_numpy().reshape(-1).astype(float)

    # Function to generate data based on data's mean and standard deviation. Requires either dataframe or series as
    # input
    def generate_data(self, data):
        gen_data = pd.DataFrame()
        if len(data.columns) > 1:
            for column in data.columns:
                gen_data[column] = pd.Series(
                    np.random.normal(data[column].mean(), data[column].std(), int(len(data[column]) / 2)))
        else:
            gen_data = pd.Series(np.random.normal(data.mean(), data.std(), int(len(self) / 2)))
        return gen_data

    def eda(self):
        # Kernel Density Esimate plot for visualizing distribution
        ForestFireDataset.kde_plot(self.data["area"])

        # Create new column to categorize the damage of the fire
        self.data["damage_category"] = self.data["area"].apply(ForestFireDataset.area_categorisation)

        # Plot fire damage per month
        categ_columns = self.data[["month", "day"]]
        for col in categ_columns:
            plt.figure(figsize=[12, 8])
            cross = pd.crosstab(index=self.data["damage_category"], columns=self.data[col], normalize="index")
            cross.plot.barh(stacked=True, rot=40, cmap="hot")
            plt.xlabel("% distribution per category")
            plt.xticks(np.arange(0, 1.1, 0.1))
            plt.title("Forestfire damage each {}".format(col))
            plt.tight_layout()
            plt.show()
            plt.savefig(f'fire_damage_{col}.pgf')

        # Correlation Heatmap of the features in the dataset
        plt.rcParams['figure.figsize'] = [12, 10]
        plt.figure().suptitle("Correlation Heatmap of features")
        sns.set(font_scale=1)
        sns.heatmap(self.data.corr(), annot=True)
        plt.tight_layout()
        plt.show()
        plt.savefig('Heatmap_correlation_feat.pgf')

        return None

    def cast_targets(self):
        self.y_train = self.y_train.astype(int)
        self.y_test = self.y_test.astype(int)

    # One-Hot Encoding for numerical data (for classification purposes)
    def one_hot_encoding_targets(data):
        data.apply(lambda x: 0 if x <= 0.0 else 1 for x in data)
        return data

    # One-Hot Encoding for categorical data
    def label_encode(target):
        return LabelEncoder().fit_transform(target)

    # Kernel Density Estimator, Skew and kutrosis of distribution.
    def kde_plot(target):
        print("Feature: {}".format(target.name))
        print("Skew: {}".format(target.skew()))
        print("Kurtosis: {}".format(target.kurtosis()))
        plt.figure(figsize=(8, 5)).suptitle("Distribution of {} values.\n".format(target.name))
        sns.kdeplot(target, shade=True, color="r", vertical=True)
        plt.xticks([i for i in range(0, 400, 50)])
        plt.tight_layout()
        plt.show()
        plt.savefig('kde_plot_area.png')

    def area_categorisation(area):
        if area == 0.0:
            return "No damage"
        elif area <= 1:
            return "Low"
        elif area <= 25:
            return "Moderate"
        elif area <= 100:
            return "High"
        else:
            return "Very high"
