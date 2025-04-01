import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier



file_path = r"G:\code\OmkarVSCode\github_projects\cricket_ball_tracking\cricket_data.csv"  
df = pd.read_csv(file_path)
X = df.drop(columns = ["Out"])
y = df["Out"]




model = DecisionTreeClassifier()
model.fit(X, y)
prediction = model.predict([[629, 872, 420,	0, 491, 17, 0,	1,]])


if prediction == 0:
    print("Not Out")

else:
    print("Out")

