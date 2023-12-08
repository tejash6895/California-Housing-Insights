# California-Housing-Insights

The "California Housing Price Prediction" project aims to develop a predictive model that estimates the median house prices in various districts of California. This project utilizes the California Housing dataset, which is derived from the 1990 U.S. census, and contains information about different attributes of housing districts, including median income, house age, population, and geographical coordinates.


## Implementation Details

- Dataset: California Housing Dataset (view below for more details)
- Model: [Linear Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- Input: 8 features - Median Houshold income, House Area, ...
- Output: House Price

## Dataset Details

This dataset was obtained from the StatLib repository ([Link](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html))

This dataset was derived from the 1990 U.S. census, using one row per census block group. A block group is the smallest geographical unit for which the U.S. Census Bureau publishes sample data (a block group typically has a population of 600 to 3,000 people).

A household is a group of people residing within a home. Since the average number of rooms and bedrooms in this dataset are provided per household, these columns may take surprisingly large values for block groups with few households and many empty houses, such as vacation resorts.

It can be downloaded/loaded using the sklearn.datasets.fetch_california_housing function.

- [California Housing Dataset in Sklearn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
- 20640 samples
- 8 Input Features: 
    - MedInc median income in block group
    - HouseAge median house age in block group
    - AveRooms average number of rooms per household
    - AveBedrms average number of bedrooms per household
    - Population block group population
    - AveOccup average number of household members
    - Latitude block group latitude
    - Longitude block group longitude
- Target: Median house value for California districts, expressed in hundreds of thousands of dollars ($100,000)

## Evaluation and Results

![Alt Text](https://github.com/tejash6895/California-Housing-Insights/raw/main/predict.png)

The significant error in your model, based on the R-squared and MSE values, indicates that the model only moderately predicts median income based on house age. 

| Metric        | Value         |
| ------------- | ------------- |
| R2 Score      | 0.48          |
| MSE           | 0.69          |


The above quant results show that <>
## Key Takeaways

What did you learn while building this project? What challenges did you face and how did you overcome them?

learned to assess data relationships and select important features for linear regression. I also mastered the evaluation of models using R-squared and MSE. Challenges included meeting linear regression assumptions, dealing with its limited complexity, and handling outliers. To overcome these challenges, I used data visualization, regularization techniques, and simplified communication methods for non-experts.


## How to Run

The code is built on Google Colab on an iPython Notebook. 

```bash
Simply download the repository, upload the notebook and dataset on colab, and hit play!
```


## Roadmap

What are the future modification you plan on making to this project?

- Try more models

- Wrapped Based Feature Selection


## Libraries 

**Language:** Python

**Packages:** Sklearn, Matplotlib, Pandas, Seaborn


## FAQ

#### How does the linear regression model work?

Linear regression is a statistical method used to model the relationship between a dependent variable and one or more independent variables. The model assumes a linear relationship between the variables, meaning that the change in the dependent variable is proportional to the change in the independent variable(s).

Here's how it works in simple terms:

Model Structure: The linear regression model can be represented by the equation of a straight line:

![Alt Text](https://github.com/tejash6895/California-Housing-Insights/raw/main/linear_regression_fom.JPG)

  are the coefficients that represent the weights for each independent variable.
�
ϵ is the error term, the part of 
�
Y that the model can't explain, which is assumed to be random and normally distributed.
Fitting the Model: To find the line of best fit, linear regression calculates the coefficients (
�
β) that result in the smallest possible difference between the predicted values and the actual values. This difference is the "error" or "residual."

Least Squares: The most common method to estimate the coefficients is the Ordinary Least Squares (OLS) method. It works by minimizing the sum of the squares of the residuals (the differences between the observed values and the values predicted by the model).

Model Evaluation: After fitting the model, you evaluate how well the model performs by looking at metrics like R-squared, which tells you the proportion of variance in the dependent variable that's predictable from the independent variables, and the Mean Squared Error (MSE), which measures the average of the squares of the errors.

Prediction: Once the model is fitted and you're satisfied with its performance, you can use it to make predictions. You simply plug in the values of the independent variables into the model's equation to get the predicted value of the dependent variable.

#### How do you train the model on a new dataset?

To build a predictive model, start by collecting a dataset with your target variable and relevant features. Clean the data by handling missing values and duplicates. Select the features you'll use and split the data into training and testing sets. Adjust the feature distribution if needed. Choose a suitable model (like linear regression), train it on the training data, and evaluate its performance using metrics like R-squared and MSE. Validate the model on the testing set to ensure it generalizes well. If necessary, fine-tune the model by adjusting features or applying regularization. Finally, use the trained model to make predictions on new data.

#### What is the California Housing Dataset?

The California Housing Dataset is a widely used dataset in machine learning and statistics. It contains information related to housing prices in various districts in California, USA. The dataset includes features such as median housing price, population, median income, and others. Researchers and data scientists often use this dataset for regression tasks, aiming to predict housing prices based on the available features. It serves as a valuable resource for testing and developing predictive models and exploring the relationships between different factors influencing housing prices in California.
## Acknowledgements

All the links, blogs, videos, papers you referred to/took inspiration from for building this project. 

 - [Awesome Readme Templates](https://awesomeopensource.com/project/elangosundar/awesome-README-templates)
 - [Awesome README](https://github.com/matiassingers/awesome-readme)
 - [How to write a Good readme](https://bulldogjob.com/news/449-how-to-write-a-good-readme-for-your-github-project)


## Contact

If you have any feedback/are interested in collaborating, please reach out to me at fake@fake.com


## License

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

