#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# #### 1. Loading and Checking Data

# In[2]:


#Load Customers dataset
customers_df=pd.read_csv(r"C:\Users\USER\Documents\Data Portfolio Projects\Retail\Customer Retention\Datasets\Customers.csv")
customers_df.head()


# In[3]:


#Load Products dataset
products_df=pd.read_csv(r"C:\Users\USER\Documents\Data Portfolio Projects\Retail\Customer Retention\Datasets\Products.csv")
products_df.head()


# In[4]:


#Load Engagements dataset
engagements_df=pd.read_csv(r"C:\Users\USER\Documents\Data Portfolio Projects\Retail\Customer Retention\Datasets\Engagements.csv")
engagements_df.head()


# In[5]:


#Load Loyalty dataset
loyalty_df=pd.read_csv(r"C:\Users\USER\Documents\Data Portfolio Projects\Retail\Customer Retention\Datasets\LoyaltyProgram.csv")
loyalty_df.head()


# In[8]:


#Check type of data we have
print(customers_df.info())


# In[10]:


#Check stats of the data
customers_df.describe()


# In[11]:


# Check for missing values
missing_values = customers_df.isnull().sum()
print("Missing values in each column:\n", missing_values)


# In[ ]:





# #### 2. Customer Segmentation

# ##### Questions:
# ###### 1. What are the key characteristics used to segment customers currently?
# ###### 2. Are there any existing customer personas or profiles?
# ###### 3. How frequently should customer segments be updated?

# In[13]:


# Segmenting customers based on PurchaseHistory, TotalSpend, and LoyaltyProgram
customer_segments = customers_df.groupby(['PurchaseHistory', 'LoyaltyProgram']).agg({
    'TotalSpend': 'mean',
    'CustomerID': 'count'
}).rename(columns={'CustomerID': 'CustomerCount'}).reset_index()


# In[14]:


customer_segments


# In[15]:


# Visualization of the segmentation
plt.figure(figsize=(10, 6))
sns.barplot(data=customer_segments, x='PurchaseHistory', y='TotalSpend', hue='LoyaltyProgram')
plt.title('Average Total Spend by Purchase History and Loyalty Program')
plt.xlabel('Purchase History')
plt.ylabel('Average Total Spend')
plt.legend(title='Loyalty Program')
plt.show()


# In[ ]:





# #### 3. Churn Prediction Model
# 

# ##### Questions:
# ###### 1. What historical data is available for developing the churn prediction model?
# ###### 2. Are there specific behaviors or events that have been associated with customer churn in the past?
# ###### 3. What machine learning tools or platforms are preferred or currently in use?

# A churn prediction model will be built using logistic regression to identify at-risk customers.

# In[17]:


#Import relevant Libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


# In[18]:


# Prepare data for churn prediction model
features = ['TotalSpend', 'FeedbackScore', 'EmailOpenRate', 'ClickThroughRate', 'WebsiteVisits', 'CustomerServiceInteractions']
X = customers_df[features]
y = customers_df['Churn']


# In[19]:


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[20]:


# Building the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# In[21]:


# Making predictions
y_pred = model.predict(X_test)


# In[22]:


# Evaluating the model
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# Result:
# Classification report and confusion matrix for the churn prediction model, showing precision, recall, and accuracy.

# In[ ]:





# #### 4. Personalized Marketing

# ##### Questions:
# ###### 1. What channels (email, SMS, in-app notifications) are used for marketing communications?
# ###### 2. How personalized are the current marketing efforts?
# ###### 3. What type of product recommendations have been successful in the past?

# In[ ]:





# Engagement data is analysed to see which channels are most effective. We also examine the success of product recommendations.

# In[23]:


# Analyzing engagement data
engagement_summary = engagements_df.groupby(['EngagementType', 'EngagementOutcome']).size().unstack(fill_value=0)


# In[24]:


# Display the engagement summary
engagement_summary.head()


# In[25]:


# Visualizing engagement outcomes
engagement_summary.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Engagement Outcomes by Type')
plt.xlabel('Engagement Type')
plt.ylabel('Count')
plt.legend(title='Engagement Outcome')
plt.show()


# Result:
# A stacked bar plot showing engagement outcomes by type, and a snippet of the engagement summary table.

# In[ ]:





# #### 5. Customer Lifetime Value (CLV) Analysis

# ##### Questions:
# ###### 1. How is CLV currently calculated?
# ###### 2. Are there specific customer segments or behaviors associated with higher CLV?
# ###### 3. What marketing strategies have been linked to increases in CLV?

# In[27]:


# Calculating CLV
customers_df['CLV'] = customers_df['TotalSpend']


# In[28]:


# Analyzing CLV by segments
clv_segments = customers_df.groupby(['PurchaseHistory', 'LoyaltyProgram']).agg({
    'CLV': 'mean',
    'CustomerID': 'count'
}).rename(columns={'CustomerID': 'CustomerCount'}).reset_index()


# In[29]:


# Display the CLV segments
clv_segments.head()


# In[30]:


# Visualizing CLV
plt.figure(figsize=(10, 6))
sns.barplot(data=clv_segments, x='PurchaseHistory', y='CLV', hue='LoyaltyProgram')
plt.title('Average CLV by Purchase History and Loyalty Program')
plt.xlabel('Purchase History')
plt.ylabel('Average CLV')
plt.legend(title='Loyalty Program')
plt.show()


# In[ ]:





# #### 6. Loyalty Program Evaluation

# ##### Questions:
# ###### 1. What are the current loyalty program's key features and benefits?
# ###### 2. How is participation in the loyalty program tracked and measured?
# ###### 3. What feedback have customers given about the loyalty program?

# Below the loyalty program is evaluated to understand participation and its impact on customer retention.

# In[32]:


# Analyzing loyalty program data
loyalty_summary = loyalty_df.describe()
loyalty_summary


# In[34]:


# Visualizing points distribution
plt.figure(figsize=(10, 6))
sns.histplot(loyalty_df['PointsEarned'], bins=20, kde=True)
plt.title('Distribution of Points Earned in Loyalty Program')
plt.xlabel('Points Earned')
plt.ylabel('Count')
plt.show()


# Result:
# A histogram showing the distribution of points earned in the loyalty program, and a summary of the loyalty program data.

# In[ ]:





# #### 7.  Customer Feedback Analysis

# ##### Questions:
# ###### 1. What methods are used to collect and store customer feedback?
# ###### 2. Are there any common themes or issues that have already been identified?
# ###### 3. How frequently is customer feedback reviewed and analyzed?

# Feedback scores will be analysed. Commonalities in feedback will also be checked.

# In[35]:


# Analyzing customer feedback scores
feedback_summary = customers_df['FeedbackScore'].describe()
feedback_summary


# In[36]:


# Visualizing feedback scores distribution
plt.figure(figsize=(10, 6))
sns.histplot(customers_df['FeedbackScore'], bins=5, kde=True)
plt.title('Distribution of Customer Feedback Scores')
plt.xlabel('Feedback Score')
plt.ylabel('Count')
plt.show()


# Result:
# Above is a histogram showing the distribution of customer feedback scores, and a summary of the feedback scores.

# In[ ]:





# In[39]:


import numpy as np


# #### 8. A/B Testing and Optimization

# ##### Questions:
# ###### 1. What types of A/B tests have been conducted previously?
# ###### 2. What metrics are used to determine the success of A/B tests?
# ###### 3. How are test results currently documented and implemented?

# The analysis will involve a simulation of an A/B test by splitting the engagement data and comparing outcomes.

# In[40]:


# Simulating A/B test with engagement data
engagements_df['Group'] = np.random.choice(['A', 'B'], size=len(engagements_df))


# In[41]:


# Analyzing A/B test results
ab_test_results = engagements_df.groupby(['Group', 'EngagementOutcome']).size().unstack(fill_value=0)
ab_test_results.head()


# In[42]:


# Visualizing A/B test outcomes
ab_test_results.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('A/B Test Outcomes by Group')
plt.xlabel('Group')
plt.ylabel('Count')
plt.legend(title='Engagement Outcome')
plt.show()


# Result:
# A stacked bar plot showing A/B test outcomes by group, and a snippet of the A/B test results table.

# In[ ]:





# #### 9. Regular Monitoring and Reporting

# ##### Questions:
# ###### 1. What key performance indicators (KPIs) are most critical for monitoring customer retention?
# ###### 2. How are these KPIs currently tracked and reported?
# ###### 3. What tools and platforms are used for creating dashboards and reports?

# We will identify key KPIs and create a sample dashboard using matplotlib.

# In[45]:


# Define key KPIs
kpis = {
    'Total Customers': len(customers_df),
    'Average CLV': customers_df['CLV'].mean(),
    'Churn Rate': customers_df['Churn'].mean(),
    'Average Feedback Score': customers_df['FeedbackScore'].mean()
}


# In[46]:


# Display the KPIs
kpis


# In[47]:


# Creating a simple KPI dashboard
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax = ax.flatten()

for i, (kpi, value) in enumerate(kpis.items()):
    ax[i].text(0.5, 0.5, f"{kpi}\n{value:.2f}", fontsize=18, ha='center')
    ax[i].axis('off')

plt.suptitle('Customer Retention KPIs', fontsize=20)
plt.show()


# Result:
# A simple KPI dashboard visualizing key metrics for customer retention, and a dictionary showing the KPI values.

# In[ ]:




