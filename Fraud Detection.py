import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree



df = pd.read_csv("onlinefraud.csv")
print(df.head())

df = df.dropna(axis = 0, how = 'any')
print(df.isnull().sum())

#Data is now clean and free from null/invalid points.

fraud = df[df['isFraud'] == 1]
notFraud = df[df['isFraud'] == 0]
print("Fradulent Transactions: " + str(fraud['isFraud'].count()))
print("Non-Fradulent Transactions: " + str(notFraud['isFraud'].count()))
print("Percetange of Transactions that were Fraudulent: " + str(round(100*(fraud['type'].count()/df['type'].count()),2)) + "%")

#A lot less than one percent of the total transactions were fraud, so it might be beneficial to look at trends in these fraudulent transactions.

fraud_counts = df.groupby('type')['isFraud'].sum()
fig, ax = plt.subplots()
fraud_counts.plot(kind='bar', ax=ax)
ax.set_xlabel('Transaction Type')
ax.set_ylabel('Number of Fraudulent Transactions')
ax.set_title('Fraudulent Transactions by Type')
plt.show()

#Transaction type

cashout_df = df.loc[df['type'].isin(['CASH_OUT'])]
cashout_fraud_count = df.loc[(df['type'] == 'CASH_OUT') & (df['isFraud'] == 1), 'type'].count()
transfer_df = df.loc[df['type'].isin(['TRANSFER'])]
transfer_fraud_count = df.loc[(df['type'] == 'TRANSFER') & (df['isFraud'] == 1), 'type'].count()
print("Total Cashout transactions: " + str(cashout_df['type'].count()))
print("Total Transfer transactions: " + str(transfer_df['type'].count()))
print("Fraudulent Transfers: " + str(transfer_fraud_count))
print("Fraudulent Cash-outs: " + str(cashout_fraud_count))
print("Percetange of Transactions that were Fraudulent: " + str(round(100*(8213/(cashout_df['type'].count()+transfer_df['type'].count())),2)) + "%")


plt.pie([cashout_fraud_count,transfer_fraud_count], labels=['Fraudulent Cash-outs','Fraudulent Transfers'], autopct='%1.1f%%')
plt.title('Fraudulent Transactions for Cashout and Transfer')
plt.show()

#Stats

df = df.loc[df['type'].isin(['CASH_OUT', 'TRANSFER'])]

print("Amounts median: " + str(df['amount'].median()))
print("Amounts average: " + str(df['amount'].mean()))
print("Amounts max: " + str(df['amount'].max()))
bins = range(0, 1000000, 500)

#Visualization 1

amount_counts = df.groupby(pd.cut(df['amount'], bins=bins))['isFraud'].count()

intervals = [pd.Interval(left=bins[i], right=bins[i+1]) for i in range(len(bins)-1)]
amount_counts.index = pd.IntervalIndex(intervals)

fig, ax = plt.subplots()
ax.bar(amount_counts.index.mid, amount_counts, width=450, align='center')
ax.set_xlabel('Amount')
ax.set_ylabel('Number of Transactions')
ax.set_title('Transactions by Amount')
plt.show()

fraud_counts = df.groupby(pd.cut(df['amount'], bins=bins))['isFraud'].sum()

intervals = [pd.Interval(left=bins[i], right=bins[i+1]) for i in range(len(bins)-1)]
fraud_counts.index = pd.IntervalIndex(intervals)

fig, ax = plt.subplots()
ax.bar(fraud_counts.index.mid, fraud_counts, width=450, align='center')
ax.set_xlabel('Amount')
ax.set_ylabel('Number of Fraudulent Transactions')
ax.set_title('Fraudulent Transactions by Amount')
plt.show()

#Visualization 2

df["changeOrig"] = abs(df['newbalanceOrig'] - df['oldbalanceOrg'])
print(df["changeOrig"].max())
bins = np.arange(-255000, 255000, 5000)

amount_counts = df.groupby(pd.cut(df['changeOrig'], bins=bins))['isFraud'].count()
change_counts = df.groupby(pd.cut(df['changeOrig'], bins=bins))['isFraud'].sum()

intervals = [pd.Interval(left=bins[i], right=bins[i+1]) for i in range(len(bins)-1)]
amount_counts.index = pd.IntervalIndex(intervals)

fig, ax = plt.subplots()
ax.bar(amount_counts.index.mid, amount_counts, width=450, align='center')
ax.set_xlabel('Change in Account Balance')
ax.set_ylabel('Number of Transactions')
ax.set_title('Transactions by Change in Account Balance')
plt.show()

change_counts.index = pd.IntervalIndex(intervals)

fig, ax = plt.subplots()
ax.bar(change_counts.index.mid, change_counts, width=450, align='center')
ax.set_xlabel('Change in Account Balance')
ax.set_ylabel('Number of Fraudulent Transactions')
ax.set_title('Fraudulent Transactions by Change in Account Balance')
plt.show()

#Removing unnecessary columns

df = df.drop(columns = ["nameOrig", "nameDest"])
print(df.head())

# Train/Test Split

df = pd.concat([df, pd.get_dummies(df['type'])], axis = 1).drop(columns = ['type'])
print(df.head())

X = df.drop(columns = ['isFraud'])
y = df['isFraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = True)
print('Training: ', X_train.shape)
print('Testing: ', X_test.shape)



scaler = StandardScaler()  
scaler.fit(X_train)  
X_train = scaler.transform(X_train) 
X_test = scaler.transform(X_test)

pca = PCA(n_components = 2)
pca.fit(X_train)
X_train2D = pca.transform(X_train)
X_test2D = pca.transform(X_test)

f, axarr = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(10, 5))
for i in range(2):
      axarr[0].scatter(X_train2D[y_train == i, 0], X_train2D[y_train == i, 1], label = str(i))
                                    
      axarr[0].legend()
      axarr[0].set_title('Training data')

      axarr[1].scatter(X_test2D[y_test == i, 0], X_test2D[y_test == i, 1], label = str(i))
                                    
      axarr[1].legend()
      axarr[1].set_title('Testing data')

#Logistic Regression

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Logistic Regression Accuracy:", accuracy)

#Decision Tree

clf_tree = tree.DecisionTreeClassifier(max_leaf_nodes = 6)
clf_tree.fit(X, y) 
clf_tree.fit(X_train, y_train) 

plt.figure(figsize = (20, 12))
tree.plot_tree(clf_tree,filled = True,fontsize=10,feature_names = ['amount','nameOrig', 'oldbalanceOrg','newbalanceDest','nameDest', 'isFraud','isFlaggedFraud', 'ChangeOrig', 'CASH_OUT', 'TRANSFER'])
plt.show()

