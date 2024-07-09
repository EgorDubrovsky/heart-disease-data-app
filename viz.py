import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

class visualize():
    def categorical_distr(self, df, feature):
        fig, ax = plt.subplots()
        ax.set_axisbelow(True)
        ax.grid()
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of feature ' + feature)
        ax.bar(np.vectorize(lambda x: str(int(x)))(np.unique(df[feature])), 
                np.unique(df[feature], return_counts=True)[1], edgecolor='black',
                color='skyblue')
        return fig
    
    def continuous_distr(self, df, feature):
        fig, ax = plt.subplots()
        ax.set_axisbelow(True)
        ax.grid()
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of feature ' + feature)
        ax.hist(df[feature], bins=30, color='skyblue', edgecolor='black')
        return fig
    
    def scatterplot(self, df, feature1, feature2):
        fig, ax = plt.subplots()
        ax.set_axisbelow(True)
        ax.grid()
        ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)
        ax.set_title('Scatterplot of features ' + feature1 + ' and ' + feature2)
        ax.scatter(df[feature1], df[feature2], color='skyblue', edgecolor='black')
        return fig
    
    def boxplot(self, df, feature_cont, feature_cat):
        fig, ax = plt.subplots()
        ax.set_axisbelow(True)
        ax.grid()
        ax.set_xlabel(feature_cat)
        ax.set_ylabel(feature_cont)
        ax.set_title('Boxplot of feature ' + feature_cont + ' for different values of feature ' + feature_cat)
        to_plot = [df[df[feature_cat] == val][feature_cont] for val in sorted(df[feature_cat].unique())]
        ax.boxplot(to_plot, labels=sorted(df[feature_cat].unique()))
        return fig
    
    def heatmap(self, df, feature1, feature2):
        fig, ax = plt.subplots()
        ax.set_axisbelow(True)
        ax.set_title('Instances of features ' + feature1 + ' and ' + feature2)
        to_plot = [[df[df[feature1] == val1][df[feature2] == val2].shape[0] for 
                    val2 in sorted(df[feature2].unique())] for val1 in sorted(df[feature1].unique())]
        sns.heatmap(to_plot, annot=True, cmap=cm.YlOrBr, fmt='d')
        ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)
        #ax.imshow(to_plot)
        #ax.scatter(df[feature1], df[feature2], color='skyblue', edgecolor='black')
        return fig