# Machine Learning Project: Effects of Carbohydrate Ingestion on Workout, Weight Training and Physical performance. 

## Table of Contents
- [Introduction](#introduction)
- [Purpose](#purpose)
- [Data Source](#data-source)
- [Approach](#approach)
- [Setup](#setup)
- [How to Use](#how-to-use)
- [Findings](#findings)
- [Next Steps](#next-steps)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Introduction
This project focuses on creating a machine learning model to investigate how carbohydrate consumption impacts Body Mass Index (BMI) during physical training. By analyzing various data points, the goal is to understand the correlation between carbohydrate intake and changes in BMI.

## Purpose
The main goals of this endeavor include:
1. Exploring the relationship between carbohydrate intake and Workout, Weight Training and Physical performance with varying BMI.
2. Building a predictive model to estimate Performance fluctuations based on the amount of carbohydrates consumed before training.

## Data Source
The dataset utilized in this project comprises details about participants, which includes:
- **BMI**: The Body Mass Index for each participant.
- **Carbohydrate Intake**: The quantity of carbohydrates ingested prior to training sessions.
- **Training Duration**: How long the training lasts.
- **Additional Variables**:
## Data Parameters
-  [VO2 Max](https://github.com/Shr3yash/CarbMI/tree/main/data-collected/PerformanceMetrics/VO2_max)
- Resting Heart Rate (RHR)
- Heart Rate Variability (HRV)
- Blood Lactate Threshold
- Maximum Heart Rate (MHR)
- Ventilatory Threshold (VT)
- Metabolic Equivalent (MET)
- Body Mass Index (BMI)
- Body Fat Percentage
- Basal Metabolic Rate (BMR)
- Fat-Free Mass (FFM)
- Respiratory Exchange Ratio (RER)
- Peak Power Output (PPO)
- Workload (Watts)
- Energy Expenditure (Calories Burned)
- Stride Length & Cadence
- Blood Pressure (Systolic/Diastolic)
- Oxygen Saturation (SpO2)
- Gait Analysis Parameters
- Rate of Perceived Exertion (RPE)
- Core Temperature
- Muscle Activation (EMG)
- Hydration Levels
- Recovery Time
- Total Work (Joules)


**Note**: Ensure you have the right to use the dataset and provide proper attribution if required.

## Approach
1. **Data Cleaning**: Preparing and organizing the dataset for analysis.
2. **Feature Selection**: Identifying key features that significantly influence BMI.
3. **Model Selection**: Testing various machine learning algorithms (like Linear Regression, Decision Trees, and Random Forest) to determine the best model for our data.
4. **Training the Model**: Training the chosen model using the training dataset.
5. **Evaluating the Model**: Assessing the model’s performance using metrics such as RMSE, MAE, and R-squared.

## Setup
To execute this project, you will need several libraries:
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`

You can install the necessary libraries using pip:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## How to Use
1. Clone this repository to your local machine:

```bash
git clone https://github.com/Shr3yash/CarbMI
cd CarbMI
```

2. Open the provided Jupyter Notebook or Python script and run the code.

3. Feel free to modify the parameters, including carbohydrate intake and training duration.

4. Review the predictions and visualizations generated by the model.

## Findings
The model produced an R-squared value of XX% on the testing dataset, suggesting a notable link between carbohydrate intake and BMI adjustments. Visual aids will assist in interpreting the effects of varying carbohydrate levels.

## Next Steps
- Explore the influence of different carbohydrate types on BMI.
- Include more variables such as exercise type and intensity for a deeper analysis.
- Enhance the model using advanced methods like neural networks.

## Acknowledgments
- [Shreyash](https://github.com/shr3yash)

## License
This project is licensed under the MIT License. Refer to the [LICENSE](LICENSE) file for more information.
