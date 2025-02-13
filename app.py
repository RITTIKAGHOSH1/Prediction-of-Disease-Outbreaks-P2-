import streamlit as st
from streamlit_option_menu import option_menu
import joblib

st.set_page_config(page_title="Prediction of Disease Outbreak",layout='wide',page_icon="👩🏼‍⚕️")

# Load models
def load_model(path):
    return joblib.load(path)

Diabetes_model = load_model("models/Diabetes_model.pkl")
heart_model = load_model("models/heart_model.pkl")
parkinsons_model = load_model("models/parkinsons_model.pkl")


st.title("Health Prediction App")


# Sidebar menu with icons
with st.sidebar:
    model_choice = option_menu(
        "Choose a Model",
        ["Diabetes Prediction", "Heart Prediction", "Parkinson's Prediction"],
        icons=['activity', 'heart', 'person'],
        menu_icon="hospital",
        default_index=0
    )


if model_choice == "Diabetes Prediction":
    st.write("Diabetes Prediction Model selected!")
    
    col1,col2=st.columns(2)

    with col1:
        Preg=st.text_input("Number of Pregnancies")
        BP=st.text_input("BloodPressure level")
        Insulin = st.text_input('Insulin Level')
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

    with col2:
        Glucose=st.text_input("Glucose level")
        SkinThickness = st.text_input('Skin Thickness value')
        BMI = st.text_input('BMI value')
        Age = st.text_input('Age of the Person')

 

    #prediction
    diab_ans=''
    if st.button("Test Result"):
        userinput=[Preg, Glucose, BP, SkinThickness,Insulin, BMI, DiabetesPedigreeFunction, Age]
        userinput=[float(x) for x in userinput]
        diabPred= Diabetes_model.predict([userinput])
        if(diabPred[0]==1):
            diab_ans='This person is Diabetic'
        else:
            diab_ans='Congratulation, This person is Healthy'
    st.success(diab_ans)


elif model_choice == "Heart Prediction":
    st.write("Heart Disease Prediction Model selected!")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')
        trestbps = st.text_input('Resting Blood Pressure')
        restecg = st.text_input('Resting Electrocardiographic results')
        oldpeak = st.text_input('ST depression induced by exercise')
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    with col2:
        sex = st.text_input('Sex')
        chol = st.text_input('Serum Cholestoral in mg/dl')
        thalach = st.text_input('Maximum Heart Rate achieved')
        slope = st.text_input('Slope of the peak exercise ST segment')

    with col3:
        cp = st.text_input('Chest Pain types')
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
        exang = st.text_input('Exercise Induced Angina')
        ca = st.text_input('Major vessels colored by flourosopy')
 
    heart_diagnosis = ''

    if st.button('Test Result'):

        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

        user_input = [float(x) for x in user_input]

        heart_prediction = heart_model.predict([user_input])

        if heart_prediction[0] == 1:
            heart_diagnosis = 'Person has heart disease'
        else:
            heart_diagnosis = 'Congratulation, This person does not have any heart disease'

    st.success(heart_diagnosis)


elif model_choice == "Parkinson's Prediction":
    st.write("Parkinson's Disease Prediction Model selected!")
   
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        RAP = st.text_input('MDVP:RAP')
        APQ3 = st.text_input('Shimmer:APQ3')
        HNR = st.text_input('HNR')
        D2 = st.text_input('D2')

    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
        PPQ = st.text_input('MDVP:PPQ')
        APQ5 = st.text_input('Shimmer:APQ5')
        RPDE = st.text_input('RPDE')
        PPE = st.text_input('PPE')

    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
        DDP = st.text_input('Jitter:DDP')
        APQ = st.text_input('MDVP:APQ')
        DFA = st.text_input('DFA')

    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
        Shimmer = st.text_input('MDVP:Shimmer')
        DDA = st.text_input('Shimmer:DDA')
        spread1 = st.text_input('spread1')

    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        NHR = st.text_input('NHR')
        spread2 = st.text_input('spread2')


    parkinsons_diagnosis = ''
   
    if st.button("Test Result"):

        user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP,Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

        user_input = [float(x) for x in user_input]

        parkinsons_prediction = parkinsons_model.predict([user_input])

        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "This person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "Congratulation, This person does not have Parkinson's disease"

    st.success(parkinsons_diagnosis)
