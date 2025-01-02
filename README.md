# Formula One Race Predictor

![Screenshot: Source Database](images/f1_racing.png)

## L1 - Executive Summary

This project analyses F1 data to predict race finishing positions. Exploratory data analysis (EDA) has shown the that the sport has changed in many areas, making prediction challenging, especially when using supervised models. Therefore, focus was placed on driver-related features, which have remained more consistent. Regression models were employed to identify which driver-related features most significantly influence the target variable, 'race position order'. Findings show that age and high career performance are significant factors. A gradient boosted decision tree improved prediction accuracy to within four race positions.

The 2024 F1 race season has ended. Max Verstappen, 27, won his 4th championship, while 39-year-old Lewis Hamilton, with 7 titles in his career, finished 9th. Further analysis is suggested on the impact of driver dynamics and risk attitudes.

## L1 - Introduction / Background
Formula 1 (F1) began competitively in 1950 and has been shaped by changing technology and race regulations. This project leverages data science to analyse F1 data to identify the biggest factors influencing race outcomes. Unlike similar Kaggle projects that sometimes dive straight into complex models like neural networks, we spend more focus on Exploratory Data Analysis (EDA) to guide simpler models suitable for a non-technical audience.

## L1 - Data Source Selection
F1 data from the Ergast-MRD API (ERG-API) was chosen as it contains data going back to the official start of the championships in 1950 to the present day and has a rich set of potential features / variables for analysis (see figure 1). It is publicly available and licensed for non-commercial purposes and is therefore permitted for use by this project (Ergast, 2024).
![Screenshot: Source Database](images/ergast_database_erd.png)
<sub>Figure 1 - Entity relationship diagram for Ergast API database (Source: Ergast, 2024)</sub>

## L1 - Methods

### L2 - Data Infrastructure and Tools
Python was selected as the programming language as it is the standard tool for data science within the author’s organisation. Specialist libraries e.g. Numpy, Pandas, Matplotlib, and Scikit-Learn enabled quicker robust analysis (Plas, 2016). VS Code and Jupyter Notebooks were used for writing Python scripts to enable a step-by-step approach to data analysis. All tools were free to use and backed by vendor support.

Organisational firewall restrictions prevented use of Python to invoke the ERG-API. Instead, static CSV files from the were manually downloaded from the ERG-API (as of 07/10/2024). This required additional processing because the ERG-API uses MySQL as a relational database using 3rd normal form, so each dimension / fact table were extracted as separate CSV files, which sometimes needed merging.

### L2 - End-to-end Data Science Process
Figure 2 shows the end-to-end data science process. The data pipeline loaded source data, fixed data quality issues, performed transformations and applied feature engineering (FE) to enrich the base data for modelling (Murel, 2024).
![Screenshot: Source Database](images/e2e_data_science_process.png) 

<sub>Figure 2 - End-to-end data science process flow</sub>

#### L3 - Loading Source Data

CSV files were loaded into a Pandas dataframe using the `pd.read_csv()` function. Pandas was chosen as it offers functions for analysing, cleaning, exploring, and manipulating data in a simple table-like structure for inspection, which makes it ideal for EDA (Altexsoft, 2024). 

#### L3 - Data Quality
The data pipeline treated data quality issues where necessary e.g. replacing critical missing values and fixing incorrect data types. Figure 3 shows an example Python script for this.

```python
# EDA for Drivers dataset: structure of the data & quality of the data - summary statistics & check uniqueness / missing values / datatype / format
stats = df_drv.describe()
print(df_drv.shape), print(df_drv), print(stats)

# Check 'number' column for missing values
df_drv['number'].unique

# Check 'driverRef' column for missing values
df_drv['driverRef'].unique

# Replace missing values ('\N') in 'number' column with 'NaN' and convert to numeric datatype
df_drv['number'] = df_drv['number'].replace('\\N', pd.NA)
df_drv['number'] = pd.to_numeric(df_drv['number'], errors='coerce')

# Fix datatypes for numerical or datetime columns
df_drv['dob'] = pd.to_datetime(df_drv['dob'])
```
<sup>Figure 3 - Example Python script to replace missing values for 'driverRef' and change datatype for 'dob'/sup>

#### L3 - Data Transformations
The data pipeline also performed transformations to reveal quick insights e.g. 'df_results' and 'df_drv' dataframes were merged and grouped to show the top 10 drivers with highest total career points (figure 4).

```python
# EDA for Drivers dataset: key business insight - top-10 drivers with highest career points

# Merge results with drivers to get 'driverRef'
df_pts = pd.merge(df_results, df_drv, on='driverId', how='right')

# Use 'driverRef' to calc total points
df_pts_grp = df_pts.groupby('driverRef')['points'].sum().reset_index()

# Remove drivers with no points
df_pts_grp_filtered = df_pts_grp[df_pts_grp['points'] > 0]

# Sorted from largest to smallest by points
df_pts_grp_sorted = df_pts_grp_filtered.sort_values(by='points', ascending=False)

# Filter for the top 10 drivers
df_top_drv = df_pts_grp_sorted.head(10)

# Plot bar chart
plt.figure(figsize=(45, 6))
sns.barplot(x='driverRef', y='points', data=df_top_drv)
plt.title('Top-10 drivers with highest career points')
plt.xlabel('Constructor')
plt.ylabel('Points')
plt.xticks(rotation=45)
plt.show()
```
![Screenshot: Source Database](images/eda_top10_career_pts_drivers.png)
<sup>Figure 4 - Example Python script to merge and group 'df_results' and 'df_drv' dataframes</sup>

### L2 - Exploratory Data Analysis
Given the author's limited F1 domain knowledge, thorough EDA was essential. Univariate Analysis (UA) identified each feature's structure, such as uniqueness, distribution, and spot outliers. Multivariate Analysis (MA) explored relationships between features and the target variable, to guide model and feature selection (Statology, 2022).

### L2 - Model Selection & Preparatiion
Regression models were selected to identify which features had the greatest impact on the target variable and in whether the relationship was positive or negative. A gradient boosted decision tree, XGBoost, was used for the last model as it can handles features that are not normally distributed, as shown by the pair grid (figure 15). Pre-processing for these supervised learning models involved splitting data into 'train' and 'test' sets to validate performance on new data (Raheja, 2024). Race position order rather than race points was selected as the target variable as EDA showed that points rules changed in 2003 and 2010 (Wikipedia, 2024).

### Feature Engineering (FE)
UA showed that driver-related features were much more consistent over time, making supervised learning more reliable (See Results section). FE was used to produce more useful model inputs related to short-term and long-term driver performance (Jacob, 2024). Two examples are shown in figures 5 and 6.

```python
# Feature Engineering: avg_career_wins

# Calc total wins for each driver
df_results['wins'] = df_results['position'] == 1
df_total_wins = df_results.groupby('driverId')['wins'].sum().reset_index()

# Calc total races for each driver
df_total_races = df_results.groupby('driverId')['raceId'].count().reset_index()
df_total_races.columns = ['driverId', 'total_races']

# Merge total wins and total races
df_drv_wins = pd.merge(df_total_wins, df_total_races, on='driverId')

# Calc average number of career wins
df_drv_wins['avg_career_wins'] = df_drv_wins['wins'] / df_drv_wins['total_races']

# Add avg_career_wins back to df_drv dataframe
df_drv = pd.merge(df_drv, df_drv_wins[['driverId', 'avg_career_wins']], on='driverId', how='left')```
```
<sup>Figure 5 - Exxample Python script to create 'avg_career_wins' using feature engineering</sup>

```python
# Feature Engineering: Add drv_won_last_race column

# Merge results with races to retrieve race season
df_res = pd.merge(df_results, df_races, on='raceId', how='left')

# Sort all race results by driver + year + race id
df_res = df_res.sort_values(['driverId', 'year_x', 'raceId'])

# Use window funtion to calc whether driver won last race by seeing if previous race result position was first
df_drv['won_last_race'] = df_res.groupby('driverId')['position'].shift(1) == 1
df_drv['won_last_race'] = df_drv['won_last_race'].fillna(False).astype(int)
```
<sup>Figure 6 - Example Python script to create 'drv_won_last_race' using feature engineering</sup>

### L1 - Results

### L2 - Univariate Analysis
UA shows that many elements of F1 have changed since 1950. For example, in the USA different circuits have been raced at in different locations, probably to make the sport more appealing compared to the most popular ones e.g. American football or basketball (figure 7).

```python
# EDA for the Circuits table: meaning of the data - which countries have changed their circuits?

# Create count of number of circuits by country sorted from largest to smallest
df_circ_count = df_circuits['country'].value_counts().sort_values(ascending=False)

# Plot bar chart
plt.figure(figsize=(20, 6))
sns.barplot(x=df_circ_count.index, y=df_circ_count.values)
plt.title('Which countries have changed their circuits?')
plt.xlabel('Country')
plt.ylabel('Number of Races')
plt.xticks(rotation=45)
plt.show()
```
![Screenshot: Source Database](images/eda_countries_that_have_changed_race_circuits.png)
<sup>Figure 7 - Ordered bar chart showing which number of circuits by country</sup>

Similarly, maximum points awarded changed in 2003 and 2010 to make the sport more competitive and introduce changes to the number of starting drivers as can be seen in figure 8 (Autosport, 2024).
```python
# EDA for Results datase: key business insight - what is the total points awarded by season?

# Merge results with races to retrieve race 'year''
df_results = pd.merge(df_results, df_races, on='raceId', how='left', suffixes=('_res', '_race'))

# Use race 'year' to calc the total race points awarded for each season
df_total_points_by_season = df_results.groupby('year')['points'].sum().reset_index()

# Plot bar chart
plt.figure(figsize=(12, 8))
plt.bar(df_total_points_by_season['year'], df_total_points_by_season['points'], edgecolor='black')
plt.xlabel('Season')
plt.ylabel('Total Points Awarded')
plt.title('What is the total points awarded by season?')
plt.grid(True)
plt.show()
```
![Screenshot: Source Database](images/eda_races_total_points_by_season.png)
<sup>Figure 8 - Ordered bar chart showing maximum points awarded by season</sup>

UA for other areas of significant changes in the data e.g. circuits, races schedule, pit stops, and lap times, etc. This has partly been due to the development of technology as well as changes to sport rules and regulations – "Evolution is the lifeblood of Formula 1 - faster cars, safer cars, new circuits and fresh-faced world champions" (BBC, 2013).
The above lack of consistency of these features led to a key decision to focus remaining analysis on driver-related features only as these have been much more consistent.

### L2 - Univariate Analysis for Driver-Related Features
Figure 9 uses an ordered-bar chart to show the drivers with the highest career points and figure 10 uses a line plot to show their relative rankings by season. Together, these indicate that consistency of driver performance could be an influencing factor for race outcomes. FE was also used to create additional features for short-term and long-term driver performance.

```python
# EDA for Drivers dataset: key business insight - top-10 drivers with highest career points

# Merge results with drivers to get 'driverRef'
df_pts = pd.merge(df_results, df_drv, on='driverId', how='right')

# Use 'driverRef' to calc total points
df_pts_grp = df_pts.groupby('driverRef')['points'].sum().reset_index()

# Remove drivers with no points
df_pts_grp_filtered = df_pts_grp[df_pts_grp['points'] > 0]

# Sorted from largest to smallest by points
df_pts_grp_sorted = df_pts_grp_filtered.sort_values(by='points', ascending=False)

# Filter for the top 10 drivers
df_top_drv = df_pts_grp_sorted.head(10)

# Plot bar chart
plt.figure(figsize=(45, 6))
sns.barplot(x='driverRef', y='points', data=df_top_drv)
plt.title('Top-10 drivers with highest career points')
plt.xlabel('Constructor')
plt.ylabel('Points')
plt.xticks(rotation=45)
plt.show()
```
![Screenshot: Source Database](images/eda_top10_career_pts_drivers.png)
<sup>Figure 9 - Ordered bar chart showing top 10 drivers with highest career points</sup>

```python
# EDA for Drivers dataset: key business insight - comparative rankings of top-10 drivers with highest career points

# Merge constructor standings with driver references
df_drv_rank = pd.merge(df_drv_standings, df_drv, on='driverId', how='right')
df_drv_rank = pd.merge(df_drv_rank, df_races, on='raceId', how='right')

# Use driver ref to filter for the top drivers based on total points earnt across all seassons
# TODO: Change from hard-coded 'driverRef' to ensure it is dynamically calculated
df_top_drv_rank = df_drv_rank[df_drv_rank['driverRef'].isin(['hamilton', 'vettel', 'max_verstappen', 'alonso', 'raikkonen', 'bottas', 'rosberg', 'perez', 'michael_schumacheer', 'ricciardo'])]

# Plot line plot
plt.figure(figsize=(40, 6))
sns.lineplot(x='year', y='position', hue='driverRef', data=df_top_drv_rank, marker='o', errorbar=None)
plt.title('comparative rankings of top-10 drivers with highest career points')
plt.xlabel('Season')
plt.ylabel('Standing')
plt.gca().invert_yaxis()  # Invert y-axis to have rank 1 at the top
plt.legend(title='Driver')
plt.show()
```
![Screenshot: Source Database](images/eda_top10_career_pts_drivers_ranked.png)
<sup>Figure 10 - Line plot showing changes in driver rankings for top 1- drivers with highest career points</sup>

Figures 11 uses a histogram to show the distribution of driver age at first race and last race. Figure 12 also uses a histogram but to show the distribution of driver age for winning drivers only. Together, these indicate that driver age might be an influencing factor for race outcomes. 
```python
# EDA for Drivers dataset: key business insight - What is the distribution of driver age when they first raced versus age when they last raced (in years)?

# Merge driver standings with drivers and races to retrieve dob
df_age1 = pd.merge(df_drv_standings, df_drv, on='driverId', how='right')
df_age2 = pd.merge(df_age1, df_races, on='raceId', how='right')

# Use dob to calculate driver date of first race & driver date of last race
df_age_grp = df_age2.groupby('driverId').agg(fr_date=('date', 'min'), lr_date=('date', 'max'), dob=('dob', 'first')).reset_index()

# Fix datatypes for numerical or datetime columns
df_age_grp['fr_date'] = pd.to_datetime(df_age_grp['fr_date'])
df_age_grp['lr_date'] = pd.to_datetime(df_age_grp['lr_date'])

# Feature engineering - use driver dob to calculate driver age at first race & age at last race
df_age_grp['age_first_race'] = (df_age_grp['fr_date'] - df_age_grp['dob']).dt.days // 365
df_age_grp['age_last_race'] = (df_age_grp['lr_date'] - df_age_grp['dob']).dt.days // 365

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(df_age_grp['age_first_race'], alpha=0.5, label='Age at First Race')
plt.hist(df_age_grp['age_last_race'], alpha=0.5, label='Age at Last Race')
plt.title('What is the distribution of driver age when they first raced race vs. when they last raced (years)?')
plt.xlabel('Age (years)')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.show()

# Feature engineering - add age at first race & age at last race back to original dataset. 
df_drv = pd.merge(df_drv, df_age_grp, on='driverId', how='left')
```
![Screenshot: Source Database](images/eda_drivers_age_at_first_and_last_race.png)

<sup>Figure 11 - Histogram showing distribution of driver age at first race and age at last race</sup>

```python
# EDA for Drivers dataset: key business insight - what is the distribution of winner driver age (in years)?

# Merge driver standings with drivers and races to retrieve dob for winning drivers
df_win_age1 = pd.merge(df_drv_standings[df_drv_standings['position'] == 1], df_drv, on='driverId', how='left')
df_win_age2 = pd.merge(df_win_age1, df_races, on='raceId', how='left')

# Ensure the date columns are in datetime format
df_win_age2['dob_x'] = pd.to_datetime(df_win_age2['dob_x'])
df_win_age2['date'] = pd.to_datetime(df_win_age2['date'])

# Use dob for winning drivers to calculate driver age at the time of the race
df_win_age2['race_age'] = (df_win_age2['date'] - df_win_age2['dob_x']).dt.days // 365
df_win_age2['race_age'] = df_win_age2['race_age'].astype(int)

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(df_win_age2['race_age'].dropna(), bins=10, alpha=0.5, label='Race Win Age')
plt.title('What is the distribution of Winning Driver Age (years)?')
plt.xlabel('Age (years)')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.show()
```
![Screenshot: Source Database](images/eda_winning_drivers_age_distribution.png)

<sup>Figure 12 - Histogram showing distribution of driver age for winning drivers only</sup>

Figure 13 uses a boxplot to show driver age outliers for winning drivers only and shows that most winners are between 24 and 37 yrs of age. Figure 14 uses a boxplot to show the average driver age by season and shows that driver age has consistently declined over time - it is much lower now than in 1950. This insight was used for model 2, whereby driver age outliers were removed.
```python
# EDA for Drivers dataset: key business insight - what is the typical age of winning drivers (years)?

# Plot box plot of driver age for race winners
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_win_age2, y='race_age')
plt.title('What is the typical age of winning drivers (years)?')
plt.xlabel('Driver')
plt.ylabel('Age (years)')
plt.xticks(rotation=45)
plt.show()
```
![Screenshot: Source Database](images/eda_winning_drivers_age_outliers.png)

<sup>Figure 13 - Boxplot showing outliers for driver age for winning drivers</sup>

```python
# EDA for Drivers dataset: key business insight - what is the average driver age (years) by season?

# Merge driver standings with drivers and races to retrieve 'dob'
df_avg_age1 = pd.merge(df_drv_standings, df_drv, on='driverId', how='left')
df_avg_age2 = pd.merge(df_avg_age1, df_races[['raceId', 'year', 'date']], on='raceId', how='left')

# Convert 'dob' and race 'date' to datetime
df_avg_age2['dob_x'] = pd.to_datetime(df_avg_age2['dob_x'])
df_avg_age2['date'] = pd.to_datetime(df_avg_age2['date'])

# Calculate driver age in each race
df_avg_age2['age'] = df_avg_age2['date'].dt.year - df_avg_age2['dob_x'].dt.year

# Feature engineering - group by year and calculate the average age of drivers
df_avg_age_per_season = df_avg_age2.groupby('year')['age'].mean()

# Line Plot
plt.figure(figsize=(10, 6))
df_avg_age_per_season.plot(kind='line', marker='o')
plt.xlabel('Season')
plt.ylabel('Average Driver Age')
plt.title('What is the average driver age (years) by season?')
plt.grid(True)
plt.show()

# Feature engineering - add age at back to original dataset. 
df_drv = pd.merge(df_drv, df_age_grp, on='driverId', how='left')
```
![Screenshot: Source Database](images/eda_drivers_age_by_season.png)

<sup>Figure 14 - Line Plot showing average driver age by season</sup>

### L2 - Multivariate Analysis
MA was conducted on the final dataframe containing driver performance features. Correlation coefficients were calculated and visually plotted using a colour-coded heatmap where the strongest correlations are highlighted in ‘red’ (see figure 15). The reason for doing this was to check model assumptions forlinear regression models i.e. features / variables have a normal distribution, linearity of variables and variable independence (Robert J Casson & Lachlan, 2014). Those feature with high correlation could indicate multi-collinearity and were thus removed from the final dataframe.

![Screenshot: Source Database](images/eda_feature_correlation_all_features.png)

<sup>Figure 15 - Heatmap to show correlation coefficients for final dataframe.</sup>

MA was repeated without the highly correlated feaures as shown in figure 16. This was used as the final dataset as input into the predictive models (Figure 16).

![Screenshot: Source Database](images/eda_feature_correlation_minus_highly_correlated_features.png)

<sup>Figure 16 - Heatmap to show correlation coefficients for final dataframe minus highly correlated features.</sup>

### Predictive Modellling

#### PRED_MDL01 - Multi Linear Regression Model
MDL01 used Multi Linear Regression (MLR) to analyse the predictor variables affecting the target variable and determine if relationships were negative or positive (figure 17). The model produced an R-squared value of 0.134, showing that 13.4% of the variation in race finishing positions was explained. The 'age_first_race_x' variable was not statistically significant (p-value > 0.05) and should be removed in future iterations. The mean absolute error (MAE) was 5.94, indicating race predictions were accurate within plus or minus 5.9 positions.

```python
# Split data into train and test data
X1 = df_dp_vars_nan2[['current_age', 'exp_years', 'age_first_race_x', 'avg_career_wins']]
y1 = df_dp_vars_nan2[['positionOrder']]

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.25, random_state=101)

# Convert train and test datasets into DataFrames
X1_train = pd.DataFrame(X1_train)
X1_test = pd.DataFrame(X1_test)

# Add constant (intercept) to predictor variables for the training set
X1_train = sm.add_constant(X1_train)
X1_test = sm.add_constant(X1_test)

# Train the model
model1 = sm.OLS(y1_train, X1_train).fit()
print(model1.summary())

# Evalute model performance
y1_pred = model1.predict(X1_test)
r1_sqr_test = r2_score(y1_test, y1_pred)
print(f'R-squared on testing dataset: {r1_sqr_test}')
```
![Screenshot: Source Database](images/mdl_model1_results.png)

<sup>Figure 17 - Model 1 results.</sup>

#### PRED_MDL02 - Multi Linear Regression Model (Minus driver age outliers)
Removing driver age outliers from MDL01 to create MDL02 (figure 18) improved the R-squared to 0.219, explaining 22% of the variation in race finishes. All predictors, including 'age_first_race_x', were significant (p < 0.05). The Mean Absolute Error (MAE) improved to 4.29, indicating predictions were accurate within ±4.3 positions.

```python
# Decision: Remove outliers for age_at_first_race_x & age_at_last_race
df_filtered = df_dp_vars_nan1[df_dp_vars_nan1['current_age'] <= 45]
df_dp_no_age_outliers = df_filtered[['current_age', 'exp_years', 'age_first_race_x', 'avg_career_wins', 'positionOrder']]

# Split data into train and test data sets
X2 = df_dp_no_age_outliers[['current_age', 'exp_years', 'age_first_race_x', 'avg_career_wins']]
y2 = df_dp_no_age_outliers[['positionOrder']]

# Split the data
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.25, random_state=101)

# Convert train and test datasets into DataFrames
X2_train = pd.DataFrame(X2_train)
X2_test = pd.DataFrame(X2_test)

# Add constant (intercept) to predictor variables for the training set
X2_train = sm.add_constant(X2_train)
X2_test = sm.add_constant(X2_test)

# Train the model
model2 = sm.OLS(y2_train, X2_train).fit()
print(model2.summary())

# Evaluate model peformance
y2_pred = model2.predict(X2_test)
r2_sqr_test = r2_score(y2_test, y2_pred)
print(f'R-squared on testing dataset: {r2_sqr_test}')
```
![Screenshot: Source Database](images/mdl_model2_results.png)

<sup>Figure 18 - Model 2 results.</sup>

#### PRED_MDL03 - XGBoost Model (Minus driver age outliers)
An XGBoost model was used for MDL03 (figure 19) to handle non-linear relationships and feature interactions. The R2 coefficient improved to 0.311, explaining 31% of the variation in race finishing positions. The model’s mean absolute error (MAE) was 4.09, indicating race predictions were accurate within ±4.1 positions.

```python
# Split data into train and test data sets
X3 = df_dp_no_age_outliers[['current_age', 'exp_years', 'age_first_race_x', 'avg_career_wins']]
y3 = df_dp_no_age_outliers[['positionOrder']]

# Split the data
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.25, random_state=101)

# Convert train and test data into DMatrix
dtrain = xgb.DMatrix(X3_train, label=y3_train)
dtest = xgb.DMatrix(X3_test, label=y3_test)

# Set parameters for XGBoost
params = {
    'objective': 'reg:squarederror',  # for regression task
    'max_depth': 6,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 101
}

# Train the model
model3 = xgb.train(params, dtrain, num_boost_round=100)

# Evalute model performance
y3_pred = model3.predict(dtest)
# r3_sqr_test = r2_score(y3_test, y3_pred)
# print(f'R-squared on testing dataset: {r3_sqr_test}')

# Predictions on training data
y3_train_pred = model3.predict(dtrain)

# Calculate R-squared for training data
r3_sqr_train = r2_score(y3_train, y3_train_pred)
print(f'R-squared on training dataset: {r3_sqr_train}')

# Predictions on test data
y3_test_pred = model3.predict(dtest)

# Calculate R-squared for test data
r3_sqr_test = r2_score(y3_test, y3_test_pred)
print(f'R-squared on testing dataset: {r3_sqr_test}')

# Evalute model results
# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y3_test, y3_pred)
print(f'Mean Squared Error (MSE): {mse}')

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y3_test, y3_pred)
print(f'Mean Absolute Error (MAE): {mae}')
```
![Screenshot: Source Database](images/mdl_model3_results.png)

<sup>Figure 19 - Model 3 results.</sup>

## Conclusion
Analysing historical race data from 1950 to October 2024 revealed key factors affecting race outcomes: average career wins, experience, age at first race, and current age. The best model predicted race positions within plus or minus 4, explaining 31% of predictions.

Further research should explore how 'youth' and 'driver dynamics', including risk attitude, affect outcomes. Data could include reasons for not finishing, time/point penalties, and fastest lap times. Weather data could also reveal driver performance under different conditions.


