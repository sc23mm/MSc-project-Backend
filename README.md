##change directory to venv
cd /Users/mukundhanmohan/Downloads/mscBackend/MSc-project-Backend/venv

#install the below require packages
pip install torch torchvision torchaudio

pip install transformers

pip install flask-cors

##Change the path to the saved model directory
in the app.py file line no 21 model_dir = "give your own directory path to the model saved_model_hybrid_one"

##RUN THE APP
python app.py
