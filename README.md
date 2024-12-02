# Flight Delay Prediction Project

## Introduction
This project aims to help passengers predict whether their flight will be delayed based on the details available on their flight tickets. By analyzing historical flight data, machine learning models are trained to predict flight delays and their potential causes, enabling passengers to plan their trips more effectively and avoid unnecessary inconveniences.

---

## Problem Statement
Flight delays cause inconvenience to passengers and significant operational challenges for airlines. This project provides a solution to predict whether a flight will be delayed using basic information such as the flight date, airline, origin and destination airports, and scheduled times from the flight ticket. By offering predictions on flight delays, this tool empowers passengers to make informed decisions about their travel plans.

---

## Dataset
The dataset consists of historical flight information, including:

- **Flight Details:** Airline, flight number, origin and destination airports.
- **Scheduled Times:** Scheduled departure and arrival times.
- **Geographical Data:** Distance of the flight.
- **Delay Details:** Categories of delay reasons (if any).

### Key Features
The final features used for model training are:
- **FL_DATE:** Flight date.
- **AIRLINE:** Airline operating the flight.
- **FL_NUMBER:** Flight number.
- **ORIGIN:** Origin airport.
- **DEST:** Destination airport.
- **CRS_DEP_TIME:** Scheduled departure time.
- **CRS_ARR_TIME:** Scheduled arrival time.
- **DISTANCE:** Distance of the flight.
- **REASON_OF_DELAY:** Reason for delay (target variable, which includes categories like No Delay, NAS Delay, Carrier Delay, Weather Delay, Security Delay, Late Aircraft Delay).
- **DAY_OF_WEEK:** Day of the week for the flight.
- **DAY_OF_MONTH:** Day of the month for the flight.

---

## Setup

### Environment
- **Big Data Framework:** Apache Spark for distributed data processing.
- **Programming Language:** Python.
- **Required Libraries:**
  - `pyspark` for machine learning and data processing.
  - `matplotlib` and `seaborn` for data visualization.
  - `pandas` for data manipulation.

### Installation
To set up the environment, ensure Apache Spark and Python are installed. Install the required Python libraries using:

```bash
pip install pyspark matplotlib seaborn pandas
```

### Loading the Data
The dataset is loaded as a Spark DataFrame:

```bash
data = spark.read.csv("./datasets/flights_data.csv", inferSchema=True, header=True)
```
---

### Exploratory Data Analysis (EDA)
EDA helps understand the data distribution and relationships. Key insights include:

1. Flight Distance Distribution: The distribution of flight distances was visualized to identify typical flight lengths.
2. Flights by Airline: The number of flights per airline was analyzed.
Flights by Day and Month: Flight frequency was analyzed across days of the week and month.
3. Scheduled Times: Compared scheduled departure and arrival time distributions.
4. Delays by Distance: Analyzed how delay reasons vary with flight distances.
5. Correlation Analysis: Examined correlations between numerical features using a heatmap.

---

### Preprocessing
#### Key Steps
1. **Filtering Data:**
    - Removed canceled or diverted flights.
    - Handled missing values.
2. Feature Engineering:
    - Extracted `DAY_OF_WEEK` and `DAY_OF_MONTH` from `FL_DATE`.
    - Combined delay columns into a single categorical column, `REASON_OF_DELAY`.
3. Feature Selection:
    - Dropped irrelevant columns such as detailed operational metrics.
4. Encoding:
    - Used StringIndexer to encode categorical columns (`AIRLINE`, `ORIGIN`, `DEST`, `REASON_OF_DELAY`) as numerical indices.
5. Feature Assembly:
    - Combined all features into a single vector using `VectorAssembler`.

---

### Model Building
#### Models Used
Three machine learning models were trained and evaluated to classify flight delays:

1. Logistic Regression: A baseline model using regularization.
2. Random Forest Classifier: A tree-based ensemble model for better interpretability and accuracy.

#### Data Splitting
The dataset was split into training and testing sets:

```bash
train_data, test_data = processed_data.randomSplit([0.7, 0.3], seed=42)
```
---

### Evaluation Metrics
The models were evaluated using:

- `Accuracy`: Overall correctness of predictions.
- `Precision`: Ability to correctly identify true positive results.
- `Recall`: Ability to capture all relevant results.
- `F1-Score`: Harmonic mean of precision and recall.

---

### Results
The performance of each model was as follows:

| **Model**             | **Accuracy** | **Precision** | **Recall** | **F1-Score** |
|-----------------------|--------------|---------------|------------|--------------|
| Logistic Regression   | 0.829562     | 0.688173      | 0.829562   | 0.752282     |
| Random Forest         | 0.829562     | 0.688173      | 0.829562   | 0.752282     |



---

### Conclusion
The project successfully predicts flight delays based on ticket information using machine learning models. Among the models, Random Forest generally performed best, but further improvements could include:

1. **Hyperparameter Tuning**: To enhance model performance.
2. **Additional Features**: Incorporating weather data or real-time airport conditions.
3. **Deployment**: Making the model accessible via a web or mobile application.