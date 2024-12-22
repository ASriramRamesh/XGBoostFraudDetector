# Create virtual environment with python. I have python 3.12.4 installed
python -m venv venv_fraud_detection

# After creating virtual environment run the requirements.txt
pip install -r requirements. txt

# Activate virtual environment first in the terminal
venv_fraud_detection\Scripts\activate

# After installing the dependencies create the requirements.txt file
pip freeze > requirements. txt 

# Run streamlit command in the terminal
streamlit run credit_card_app.py

