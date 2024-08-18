## Clone The Project

##change directory to venv

cd MSc-project-Backend/venv

##INSTALL THE PACKAGES USING BELOW COMMANDS

pip install torch torchvision torchaudio

pip install transformers

pip install flask-cors

##CHANGE THE PATH TO THE SAVED HYBRID MODEL DIRECTORY FOLDER WHICH IS PRESENT IN THIS REPO `https://github.com/sc23mm/MSc-Project`

in the `app.py` file line no 21 `model_dir = "give your own directory path to the model saved_model_hybrid_one" `

##RUN THE APP

python app.py
