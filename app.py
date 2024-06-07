import streamlit as st
import pandas as pd
import joblib
import pyreadstat
from PIL import Image
from streamlit_option_menu import option_menu
from sklearn.ensemble import RandomForestClassifier

# Define the path to your files
data_path = ("Data.sav")
model_path = ('Random_forest_model.pkl')

# Load dataset
try:
    data, meta = pyreadstat.read_sav(data_path)
except FileNotFoundError:
    st.error(f"Error: Data file '{data_path}' not found!")
    st.stop()

# Load the trained model
try:
    model = joblib.load(model_path)
    expected_columns = model.feature_names_in_
except FileNotFoundError:
    st.error(f"Error: Model file '{model_path}' not found!")
    st.stop()

# Set page configuration
st.set_page_config(page_title="HeartBeats", page_icon=":heart:", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for background colors, fonts, and text justification
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Urbanist:wght@400;700&display=swap');
    .reportview-container .main {
        color: black;
        font-family: 'Urbanist', sans-serif;
        text-align: justify;
    }
    .css-1d391kg {
        font-family: 'Urbanist', sans-serif;
        text-align: justify;
    }
    h1, h2, h3, h4, h5, h6, p, div, label, input, textarea {
        font-family: 'Urbanist', sans-serif;
        text-align: justify;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar with logo
st.sidebar.image('Logo.png', use_column_width=True)

# Initialize navigation state
if 'navigation' not in st.session_state:
    st.session_state.navigation = 'Home'

# Sidebar menu with icons
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Scan Prediction Test", "Contact"],
        icons=["house", "search", "envelope"],
        menu_icon="cast",
        default_index=["Home", "Scan Prediction Test", "Contact"].index(st.session_state.navigation),
    )
    st.session_state.navigation = selected

# Custom function to navigate between pages
def navigate_to(page):
    st.session_state.navigation = page
    st.experimental_rerun()

# Function to collect user input features into a dataframe
def Rekap_Data_Pasien():
    age = st.number_input('Usia', min_value=0, max_value=119, step=1, format="%d", key='age', value=None)
    sex = st.selectbox('Jenis kelamin', ['Pilih', 'Laki-laki', 'Perempuan'], key='sex')
    cp = st.selectbox('Jenis nyeri dada', ['Pilih', 'Typical angina', 'Atypical angina', 'Non-angina', 'Tanpa gejala'], key='cp')
    tres = st.number_input('Tekanan darah istirahat (mmHg)', min_value=0, max_value=370, step=1, format="%d", key='trestbps', value=None)
    chol = st.number_input('Serum kolestrol (mg/dL)', min_value=100, max_value=1000, step=1, format="%d", key='chol', value=None)
    fbs = st.selectbox('Gula darah puasa >120 mg/dL?', ['Pilih', 'Tidak', 'Ya'], key='fbs')
    res = st.selectbox('Hasil elektrokardiografi istirahat', ['Pilih', 'Normal', 'Ada kelainan', 'Hypertrophy'], key='restecg')
    tha = st.number_input('HRmax - Denyut jantung maksimum yang tercapai', min_value=27, max_value=600, step=1, format="%d", key='thalach', value=None)
    exa = st.selectbox('Nyeri dada yang dipicu oleh olahraga', ['Pilih', 'Tidak', 'Ya'], key='exang')
    old = st.number_input('Oldpeak', min_value=0.0, max_value=6.2, step=0.1, format="%.1f", key='oldpeak', value=None)
    slope = st.selectbox('Kemiringan puncak segmen ST yang dipicu oleh olahraga', ['Pilih', 'Meningkat', 'Mendatar', 'Menurun'], key='slope')
    ca = st.selectbox('Jumlah pembuluh darah besar yang diwarnai fluoroskopi', ['Pilih', 0, 1, 2, 3], key='ca')
    thal = st.selectbox('Hasil tes Stres Thalium', ['Pilih', 'Normal', 'Cacat tetap', 'Cacat reversibel'], key='thal')

    # Mapping the categorical features to numerical values
    sex = 1 if sex == 'Laki-laki' else 0 if sex == 'Perempuan' else None
    cp = {'Typical angina': 0, 'Atypical angina': 1, 'Non-angina': 2, 'Tanpa gejala': 3}.get(cp, None)
    fbs = 1 if fbs == 'Ya' else 0 if fbs == 'Tidak' else None
    res = {'Normal': 0, 'Ada kelainan': 1, 'Hypertrophy': 2}.get(res, None)
    exa = 1 if exa == 'Ya' else 0 if exa == 'Tidak' else None
    slope = {'Meningkat': 0, 'Mendatar': 1, 'Menurun': 2}.get(slope, None)
    thal = {'Normal': 0, 'Normal': 1, 'Cacat tetap': 2, 'Cacat reversibel': 3}.get(thal, None)

    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': tres,
        'chol': chol,
        'fbs': fbs,
        'restecg': res,
        'thalach': tha,
        'exang': exa,
        'oldpeak': old,
        'slope': slope,
        'ca': ca,
        'thal': thal,
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Preprocess the input to match the model's training data
def preprocess_input(input_df):
    # One-hot encoding for categorical features
    categorical_columns = ['cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    input_df = pd.get_dummies(input_df, columns=categorical_columns)

    # Ensure all expected columns are present
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match the training data
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)
    
    return input_df

if st.session_state.navigation == "Home":
    st.write("## HeartBeats")
    st.image('Beranda.jpg')
    
    st.write("""
    HeartBeats adalah platform yang dapat membantu para dokter untuk memberikan diagnosa awal tentang kondisi dan kesehatan jantung. Dengan fitur scan kesehatan jantung yang dimiliki oleh HeartBeats, 
    setiap dokter dapat terbantu untuk mendapatkan hasil diagnosa terbaik dari platform ini.
    """)

    # Tombol "Periksa sekarang" yang mengarahkan ke halaman "Scan Prediction Test"
    if st.button('Periksa sekarang'):
        navigate_to("Scan Prediction Test")

elif st.session_state.navigation == "Scan Prediction Test":
    st.write("## SCAN PREDICTION TEST")
    
    input_df = Rekap_Data_Pasien()
    
    if st.button('Scan'):
        if any(value is None for value in input_df.iloc[0]):
            st.error('Harap mengisi semua data terlebih dahulu!')
        else:
            input_df = preprocess_input(input_df)
            try:
                prediction = model.predict(input_df)
                proba = model.predict_proba(input_df)
                if prediction[0] == 1:
                    st.error(f'Terdeteksi penyakit jantung dengan probabilitas: {proba[0][1]:.2f}')
                else:
                    st.success(f'Tidak terdeteksi penyakit jantung dengan probabilitas: {proba[0][0]:.2f}')
            except ValueError as e:
                st.error(f"Error during prediction: {e}")
    
    if st.button('Kembali'):
        navigate_to("Home") 

elif st.session_state.navigation == "Contact":
    st.write("## CONTACT PAGE")
    st.write("Get in touch with us:")
    st.write("- Email: petikmanggafm@gmail.com")
    st.write("- Phone: 0852-1234-1117")
    st.write("- Address: Universitas Negeri Jakarta, Rawamangun, Jakarta Timur")

    st.write("## Contact Form:")
    name = st.text_input("Name")
    email = st.text_input("Email")
    message = st.text_area("Message", height=150)
    submitted = st.button("## Submit")

    if submitted:
        if all([name, email, message]):
            st.write(f"Thank you, {name}! Your message has been submitted.")
            # Here you can add code to handle the submission, such as sending an email or saving to a database
        else:
            st.write("Please fill in all fields.")
