import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
#Load Data
data = pd.read_csv(r"C:\Users\DELL\Downloads\archive (23)\Health_insurance.csv")
data
#Handling missing values
data.dropna(inplace=True)
#Encoding the Categorical Variable Using LabelEncoder
label_encoder=LabelEncoder()
data["sex"]=label_encoder.fit_transform(data["sex"])
data["smoker"]=label_encoder.fit_transform(data["smoker"])
data["region"]=label_encoder.fit_transform(data["region"])
#Visualizing the distribution of charges using Seaborn
plt.figure(figsize=(10,6))
sns.histplot(data['charges'],bins=30,kde=True)
plt.title("Distribution of Charges")
plt.xlabel("Charges")
plt.ylabel("Frequency")
plt.show()
#Visualizing Correlation using Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(data.corr(),annot=True,cmap="coolwarm",fmt=".2f")
plt.title("Correlation Matrix")
plt.show()
# Plotting pie chart for regions
fig_pie = px.pie(data, names="region", title="Distribution of Regions")
fig_pie.show()
#Splitting Features and Target
X=data.iloc[:,:-1]
y=data["charges"]
#Splitting the dataset into Train and Test set
X_train,X_test,y_train,y_test=tts(X,y,random_state=42,test_size=0.2)
#Training Random forest model
forest=RandomForestRegressor()
forest.fit(X_train,y_train)
#Predicting on Test set
y_pred=forest.predict(X_test)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Calculate R-squared score
r2 = r2_score(y_test, y_pred)
print("R-squared Score:", r2)
# Visualizing predicted vs actual charges
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted Charges")
plt.show()
