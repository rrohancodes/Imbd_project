# IMDb Movie Rating Prediction

This project uses machine learning to predict IMDb movie ratings based on different movie attributes such as genre, certificate, director, number of votes, gross collection, runtime, release year, IMDb metadata score, and the movie overview.

I worked on this project as a machine learning project to understand how different types of movie data can be processed and used for both classification and regression.

## What the Project Does

The project looks at IMDb's Top 1000 movies and performs two different tasks:

1. **Classification**
   Predicts whether a movie is likely to have a **High** or **Low** IMDb rating.

2. **Regression**
   Predicts the actual IMDb rating of a movie.

The classification threshold is based on the median IMDb rating in the dataset.

## Dataset

The dataset contains information about the top 1000 IMDb movies and TV shows.

Some of the columns used in the project are:

* `Meta_score`
* `No_of_Votes`
* `Gross`
* `Runtime`
* `Released_Year`
* `Genre`
* `Certificate`
* `Director`
* `Overview`
* `IMDB_Rating`

The dataset was obtained using KaggleHub.

## Data Preprocessing

Before training the models, the data was cleaned and transformed.

Some of the preprocessing steps include:

* Converting `Gross` into a numeric value by removing commas.
* Converting `Runtime` from values such as `169 min` into numbers.
* Converting `Released_Year`, `Meta_score`, and `No_of_Votes` into numeric columns.
* Applying `log1p` transformation to `No_of_Votes` and `Gross`.
* Removing rows with missing values after preprocessing.
* Using median imputation for numerical features.
* Using most-frequent imputation for categorical features.

### Handling Text Data

The `Overview` column contains the movie description, so I used **TF-IDF** to convert the text into numerical features.

The TF-IDF vectorizer uses a maximum of 300 features and removes English stop words.

### Handling Directors

Instead of directly one-hot encoding directors, I created a custom `DirectorFrequencyEncoder`.

It calculates how frequently each director appears in the dataset. Directors that were not seen during training are assigned a frequency of 0.

## Machine Learning Models

### Classification

For classification, I used an `ExtraTreesClassifier`.

The model is trained to classify movies into:

* High rating
* Low rating

The classes are created using the median IMDb rating as the threshold.

### Regression

For regression, I used a `RandomForestRegressor` to estimate the actual IMDb rating.

## Model Pipeline

The preprocessing and model training are combined using Scikit-learn pipelines.

The preprocessing pipeline handles:

* Numerical features
* Categorical features
* Director frequency encoding
* Movie overview TF-IDF features

This makes it easier to apply the same transformations to both the training and test data.

## Results

### Classification Results

The classification model achieved an accuracy of approximately:

**71.33%**

Classification report:

| Class | Precision | Recall | F1-Score |
| ----- | --------- | ------ | -------- |
| High  | 0.74      | 0.51   | 0.60     |
| Low   | 0.70      | 0.87   | 0.78     |

The model performs better at identifying Low-rated movies than High-rated movies.

### Regression Results

The regression model achieved:

| Metric | Score |
| ------ | ----- |
| MAE    | 0.159 |
| RMSE   | 0.212 |
| R²     | 0.471 |

The R² score shows that the model is able to explain a reasonable amount of the variation in IMDb ratings, although there is still plenty of room for improvement.

## Feature Importance

Some of the features that had a noticeable influence on the models were:

* Number of votes
* Meta score
* Runtime
* Release year
* Director frequency
* Gross collection
* Genre
* Certificate
* Words from the movie overview

`No_of_Votes` was the most important feature in the feature importance results.

## Example Prediction

I also tested the model with a custom movie example using the following information:

* Meta Score: 74
* Number of Votes: 1,512,360
* Gross: 188,020,017
* Runtime: 169 minutes
* Release Year: 2014
* Genre: Sci-fi
* Certificate: UA
* Director: Christopher Nolan

The model predicted:

**Rating Class: High**

Prediction confidence:

* High: 93.5%
* Low: 6.5%

Estimated IMDb rating:

**8.52**

## Visualizations

The project includes several visualizations to understand the dataset and model results, including:

* Feature importance
* Class distribution
* Confusion matrix
* Prediction confidence
* Classification metrics
* Important words from movie overviews
* Average votes by genre
* Meta Score distribution
* Votes distribution
* Runtime distribution

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* KaggleHub
* Jupyter Notebook / Google Colab

## Project Structure

```text
Imbd_project/
│
├── imbd_classification.ipynb
└── README.md
```

## How to Run

1. Clone the repository.
2. Open `imbd_classification.ipynb`.
3. Install the required Python libraries if they are not already installed.
4. Run the notebook from top to bottom.
5. The notebook downloads the dataset using KaggleHub and performs preprocessing, training, evaluation, and prediction.

## What I Learned

This project helped me understand how to work with a dataset containing different types of features at the same time.

In particular, I worked with:

* Data cleaning
* Missing value handling
* Numerical transformations
* Categorical encoding
* TF-IDF for text data
* Custom Scikit-learn transformers
* Classification
* Regression
* Random Forest
* Extra Trees
* Model evaluation
* Feature importance

## Limitations

The model is not meant to perfectly predict IMDb ratings.

IMDb ratings depend on many factors that are difficult to capture from the available dataset. The dataset is also limited to the movies included in the IMDb Top 1000 dataset.

The classification model also performs noticeably better for the Low class than the High class, so there is room to improve the classification performance.

## Future Improvements

Some things I would like to improve in the future are:

* Try other classification and regression algorithms.
* Perform more extensive hyperparameter tuning.
* Experiment with better text representations.
* Add more movie-related features.
* Improve handling of directors and genres.
* Compare different feature engineering approaches.
* Build a simple web interface where a user can enter movie details and get a prediction.

## Author

**Rohan**

This project was built as part of my machine learning practice and to get more hands-on experience with real-world data and Scikit-learn.
