# Capstone_Project:-

**------ Overview of Project ---------**
- The goal of this project is to build a model that predicts the best insurance
cost for an individual based on their health and lifestyle habits. This
involves analyzing factors like age, BMI, smoking habits, physical activity,
and medical history to estimate personalized insurance premiums. The aim
is to make insurance pricing fair and accurate, ensuring that people pay
based on their actual health risks. For insurance companies, this helps
create better pricing strategies and improves customer trust by offering
policies tailored to each person. It also encourages people to adopt
healthier habits by showing how their lifestyle affects their insurance costs.
Overall, this approach makes insurance more accessible, helps people
manage their health better, and creates a win-win for both customers and
insurers.
- The data provided here it’s related to users as with their medical related
history where we have many features like ‘years_of_insurance_with_us’,
‘age’, ‘gender’, ‘smoking_status’ on the basis of that we have to predict
insurance cost by using some Machine Learning Algorithm.

**------- Understanding business/social opportunity -----------**
- Predicting healthcare life insurance costs can benefit both companies and
society. For businesses, it helps insurance companies set fair and accurate
premiums based on individual health risks. This ensures they don’t lose
money by underestimating risks or lose customers by overcharging. It also
allows companies to create personalized policies, improve their financial
planning, and build trust with customers.
- For society, it makes life insurance more affordable and accessible,
especially for people who might otherwise struggle to get coverage. It also
encourages healthier lifestyles by identifying common health risks, helping
people take preventive actions. Overall, this improves the financial security
and well-being of individuals and their families while supporting a stronger
relationship between insurers and healthcare providers. It is going to help
people to get better insurance as per their usage and coverage also it’ll
BUSINESS REPORT 6
help companies to provide better service to user as per their need and also
it’ll help companies to decide insurance cost on the basis of usage.

**---------- Model Building and Conclusion -------------**

In this project, multiple regression models were developed and evaluated to predict insurance costs based on health and lifestyle data. The key steps included data scaling, training-test splitting (75%-25%), model building, evaluation, and hyperparameter tuning.

- Tuned Random Forest Regressor
* Best accuracy (highest R²)
* Lowest error metrics (MAE & RMSE)
* Stable and generalizable performance due to ensemble averaging
* Ideal for complex, non-linear relationships in the dataset

- Tuned Random Forest delivered the best results:

R² = 95.61%
MAE = 2435.85
RMSE = 3051.97
