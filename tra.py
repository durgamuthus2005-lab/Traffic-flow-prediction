
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")


st.set_page_config(page_title="Traffic Analysis & Prediction", layout="wide")


st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://media.istockphoto.com/id/599718046/photo/traffic-lights-different-focus.jpg?s=612x612&w=0&k=20&c=nTiGQwMezC5vOMRBOQDQeYmcrrq2_JEuVY9wqDXiJ64=");
        background-size: cover;
        }
        
    
    
    
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown('<h1 style="color:white;">🚦 Real Time Traffic Flow Prediction 🚦</h1>', unsafe_allow_html=True)



section = st.sidebar.radio("", ["Load Dataset",  "EDA",  "Data Visualization",  "Model Training",  "Check Overfitting / Underfitting"  , "Accuracy & Reports",  "New Prediction"])


if section == "Load Dataset":
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of Dataset:")
        st.dataframe(df.head())

elif section == "EDA":
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        if st.checkbox("Show Dataset Information"):
            st.write("### Dataset Information")
            buffer = df.info()
            st.text(buffer)
        
        if st.checkbox("Show Statistical Summary"):
            st.write("### Statistical Summary")
            st.write(df.describe())
        
        if st.checkbox("Show Missing Values"):
            st.write("### Missing Values:")
            st.write(df.isnull().sum())

        
        if df.isnull().sum().sum() > 0:
            st.warning("Dataset contains missing values!")
            impute_option = st.selectbox("Select Imputation Method", ["Mean", "Median", "Mode"])
            
            if st.button("Apply Imputation"):
                from sklearn.impute import SimpleImputer
                
                if impute_option == "Mean":
                    imputer = SimpleImputer(strategy="mean")
                elif impute_option == "Median":
                    imputer = SimpleImputer(strategy="median")
                else:
                    imputer = SimpleImputer(strategy="most_frequent")
                
                df[df.columns] = imputer.fit_transform(df)
                st.success(f"Missing values filled with {impute_option} strategy.")
                st.write("### Updated Dataset Preview:")
                st.dataframe(df.head())
                
    else:
        st.warning("Please upload a dataset first.")



elif section == "Data Visualization":
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        
        df = pd.read_csv(uploaded_file)
        if st.checkbox("Show Traffic Distribution Over Time"):
            fig, ax = plt.subplots()
            sns.histplot(df, x="Time", hue="Traffic Situation", element="step", ax=ax)
            st.pyplot(fig)

        if st.checkbox("Show Traffic Situation Count"):
            fig, ax = plt.subplots()
            sns.countplot(x=df["Traffic Situation"], ax=ax)
            st.pyplot(fig)
    else:
        st.warning("Please upload a dataset first.")


elif section == "Model Training":
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        
        if df.isnull().sum().sum() > 0:
            st.warning("Dataset contains missing values! Applying default imputation (mean for numeric, mode for categorical).")
            from sklearn.impute import SimpleImputer
            
            
            numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
            categorical_cols = df.select_dtypes(include=["object"]).columns
            
            numeric_imputer = SimpleImputer(strategy="mean")
            categorical_imputer = SimpleImputer(strategy="most_frequent")
            
            df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
            df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])
            
            st.success("Imputation completed.")
        # Encoding target variable
        label_encoder = LabelEncoder()
        df["Traffic Situation"] = label_encoder.fit_transform(df["Traffic Situation"])
        
        
        categorical_features = ["Time", "Day of the week"]
        numerical_features = ["CarCount", "BikeCount", "TruckCount", "BusCount", "Total"]
        
        preprocessor = ColumnTransformer([
            ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('scaler', StandardScaler(), numerical_features)
        ])
        
        
        X = df.drop(columns=["Traffic Situation", "Date"])
        y = df["Traffic Situation"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
      
        model_choice = st.selectbox("Select Model", ["Logistic Regression", "SVM", "Random Forest"])
        
        if model_choice == "Logistic Regression":
            model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500))])
        elif model_choice == "SVM":
            model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', SVC(kernel='rbf', probability=True))])
        else:
            model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])
        
        if st.button("Train Model"):
            model.fit(X_train, y_train)
            st.success(f"{model_choice} Model Trained Successfully!")
            
            joblib.dump(model, "traffic_model.pkl")
            joblib.dump(label_encoder, "label_encoder.pkl")
    else:
        st.warning("Please upload a dataset first.")

elif section == "Check Overfitting / Underfitting":
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        try:
            model = joblib.load("traffic_model.pkl")
            label_encoder = joblib.load("label_encoder.pkl")
            
            df = pd.read_csv(uploaded_file)
            df["Traffic Situation"] = label_encoder.fit_transform(df["Traffic Situation"])
            
            X = df.drop(columns=["Traffic Situation", "Date"])
            y = df["Traffic Situation"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Training & Testing Accuracy
            train_accuracy = accuracy_score(y_train, model.predict(X_train))
            test_accuracy = accuracy_score(y_test, model.predict(X_test))

            st.write("### Model Overfitting/Underfitting Check")
            st.write(f"🔵 *Training Accuracy:* {train_accuracy:.2f}")
            st.write(f"🟢 *Testing Accuracy:* {test_accuracy:.2f}")

           
            if train_accuracy > test_accuracy + 0.15:
                st.warning("⚠ Overfitting Detected! The model performs well on training data but poorly on testing data.")
                st.write("✅ *Solution:* Try reducing complexity (e.g., decrease tree depth in Random Forest, add regularization in SVM/Logistic Regression).")
            elif train_accuracy < 0.60 and test_accuracy < 0.60:
                st.warning("⚠ Underfitting Detected! The model is too simple and not learning patterns well.")
                st.write("✅ *Solution:* Try using more features, tuning hyperparameters, or using a more complex model.")
            else:
                st.success("✅ Model has a good balance between training and testing performance.")

        except Exception as e:
            st.warning("Please train the model first in the 'Model Training' section.")
            st.error(f"Error: {e}")
    else:
        st.warning("Please upload a dataset first.")


elif section == "Accuracy & Reports":
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        
        try:
            model = joblib.load("traffic_model.pkl")
            label_encoder = joblib.load("label_encoder.pkl")
            
            df = pd.read_csv(uploaded_file)
            df["Traffic Situation"] = label_encoder.fit_transform(df["Traffic Situation"])
            
            X = df.drop(columns=["Traffic Situation", "Date"])
            y = df["Traffic Situation"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            y_pred = model.predict(X_test)
            
            st.write("### Model Performance")
            st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
            st.text(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
        except:
            st.warning("Please train the model first in the 'Model Training' section.")
    else:
        st.warning("Please upload a dataset first.")


elif section == "New Prediction":
    try:
        
        model = joblib.load("traffic_model.pkl")
        label_encoder = joblib.load("label_encoder.pkl")

        
        days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        st.write("### Predict New Data")
        
        # User Inputs
        time = st.text_input("Time (e.g., 14:00 or 2:00 PM)")
        day = st.selectbox("Day of the Week", days_of_week)
        car_count = st.number_input("Car Count", min_value=0, step=1)
        bike_count = st.number_input("Bike Count", min_value=0, step=1)
        truck_count = st.number_input("Truck Count", min_value=0, step=1)
        bus_count = st.number_input("Bus Count", min_value=0, step=1)
        total = car_count + bike_count + truck_count + bus_count  
        st.write(f"calculate total vehicles : {total}")
        
        if st.button("Predict Traffic Situation"):
            
            new_data = pd.DataFrame([{
                "Time": time,
                "Day of the week": day,
                "CarCount": car_count,
                "BikeCount": bike_count,
                "TruckCount": truck_count,
                "BusCount": bus_count,
                "Total": total
            }])

            
            preprocessor = model.named_steps["preprocessor"]
            transformed_data = preprocessor.transform(new_data)

           
            prediction = model.named_steps["classifier"].predict(transformed_data)
            predicted_traffic = label_encoder.inverse_transform(prediction)

            st.success(f"Predicted Traffic Situation: *{predicted_traffic[0]}*")
    
    except Exception as e:
        st.warning("Please train the model first in the 'Model Training' section.")
        st.error(f"Error: {e}")