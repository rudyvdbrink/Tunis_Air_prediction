# Tunis Air flight delay prediction

Machine learning challenge to predict flight delays for Tunis Air. Information on the challenge can be found [here](https://zindi.africa/competitions/flight-delay-prediction-challenge).

Before running the code, you need to download the [data](https://zindi.africa/competitions/flight-delay-prediction-challenge/data).

For this coding challenge I used some of the code from [this](https://github.com/rudyvdbrink/Tunis_Air_prediction) repository.

And overview of some of the the models I've tried:

 


<p align="center">
    <img src="https://github.com/rudyvdbrink/Tunis_Air_prediction/blob/main/plots/models.png" width="600"\>
</p>

The (current) best solution to the coding challenge involved breaking the problem down into two distinct components: one a classification task, and one a regression task. The classification task involved predicting if a flight is delayed (or not), and the regression task involved predicting the amount of delay of a flight, assuming that it is delayed. 

### List of files:
- `0_EDA.ipynb`: Initial data exploration and making some plots.
- `1_preprocess_data.ipynb`: Prepare data for ML models (including train-test split).
- `2a_fit_classifcation_model.ipynb`: Fit XGBoost classifier to make binary delay / on time predictions.
- `2b_fit_regression_model.ipynb`: Fit XGBoost regression model to make graded delay predictions.
- `3_combine_models.ipynb`: Make a predictions based on the combination of the classifier and regressor, and run model evaluation.


### **Installation, for `macOS`** do the following: 


- Install the virtual environment and the required packages:

    ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
### **Installation, for `WindowsOS`** do the following:

- Install the virtual environment and the required packages:

   For `PowerShell` CLI :

    ```PowerShell
    pyenv local 3.11.3
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

    For `Git-Bash` CLI :

  ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/Scripts/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```