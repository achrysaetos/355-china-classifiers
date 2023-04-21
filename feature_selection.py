import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# Read data
df = pd.read_csv("data_formatted.csv")
features = np.array(['GDP','CPI','Inflation','Tariff rate','Unemployment','Trade balance','Military spending'])

num_runs = 100
d = {'GDP':0, 'CPI':0, 'Inflation':0, 'Tariff rate':0, 'Unemployment':0, 'Trade balance':0, 'Military spending':0}

for i in range(num_runs):
    # Perform classification
    classifier = RandomForestClassifier() # or GradientBoostingClassifier
    classifier.fit(df[features], df['FDI']) # or FPI, CPI, etc.
    importances = classifier.feature_importances_

    # Track scores over all runs
    d['GDP'] += importances[0]
    d['CPI'] += importances[1]
    d['Inflation'] += importances[2]
    d['Tariff rate'] += importances[3]
    d['Unemployment'] += importances[4]
    d['Trade balance'] += importances[5]
    d['Military spending'] += importances[6]

# Sort indicators based on importance, and print averages
cumulative_scores = sorted(d.items(), key=lambda x: x[1])[::-1]
for feature, score in cumulative_scores:
    print(feature, score/num_runs)
