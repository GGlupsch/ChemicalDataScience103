# Assignment_Session06: Analysing Chemical Reaction Data

This exercise is supposed to be done collaboratively to split the workload and also to explore collaborative coding. While you can complete the exercise on your own, we highly recommend to try in collaboration with a partner.

In order to complete the assignment, **each student has to provide working solutions for both tasks**, i.e. two python scripts that can be run using the provied `reaction_analysis.py` script. That means if you collaborated with a partner, both of you have to submit both scripts. **Please indicate in your email with whom you collaborated and who worked on which task.**


## Scenario
Your research group has collected kinetic data from two different experimental setups studying the same catalytic reaction. You will be working in a team of two to analyse and combine this data to determine the optimal reaction conditions. 

## Goals
- Apply pandas for data cleaning and analysis
- Use numpy for numerical calculations
- Collaborate on a shared codebase using Git

## Setup
1. Use _only_ one Github repository on which you are going to collaborate
2. Add your partner as a collaborator
3. Both team members clone the repository

## Tasks to solve

### Person A: Data Cleaning & Standardisation (Branch: `data-cleaning`)

Your tasks:
1. Create branch `data-cleaning`. You will be making your changes on this branch.
2. Create a script `data_cleaner.py` with functions to:
   - Load both CSV files using pandas
   - Standardize column names across both datasets
   - Handle any missing or inconsistent data
   - Convert units if necessary (ensure temperature is in Celsius)
   - Create a combined dataset with a new column indicating the setup ('A' or 'B')

3. Required functions:
   ```python
   def load_and_clean_setup_data(filepath, setup_name):
       """Load and standardize data from one setup"""
       pass
   
   def combine_datasets(setup_a_data, setup_b_data):
       """Combine datasets from both setups"""
       pass
   ```

You can find the required data in the data folder.

### Person B: Analysis & Calculations (Branch: `kinetics-analysis`)

Your tasks:
1. Create branch `kinetics-analysis`. You will be working on this branch.
1. Create a module `kinetics_analyser.py` with functions to:
   - Calculate reaction rates using numpy (change in concentration over time)
   - Determine which setup achieved higher average yield
   - Find the optimal temperature range (where yield increase is highest)

2. Required functions:
   ```python
   def calculate_reaction_rates(dataframe):
       """Calculate reaction rates for each time interval"""
       # hint: use df.groupby("setup")["columname"].diff()
       pass
   
   def find_optimal_conditions(dataframe):
       """Analyze data to find optimal reaction conditions"""
       pass
   
   def compare_setups(dataframe):
       """Compare performance between setups A and B"""
       pass
   ```

## Suggested Timeline

### Step 1: Solve the tasks
- Each person works on their branch
- Make regular commits with descriptive messages
- Test your functions with the sample data and where necessary adds unit tests

### Step 2: Merging the code
Once you are satisfied with your implementation, open a pull request on github to merge your code into main. Let your team mate review the code that you wrote. There will likely be questions or opportunities to improve the code, so use Githubs review function to address these.

### Step 3: Final Integration 
We have provided a script that integrates both tasks and uses them to analyse the reactions. Run `reaction_analysis.py` to evaluate your code and if needed make changes to ensure that the functions that you built work (**Make sure that the scripts are all in the same folder and you specify the correct path to the data!**).



