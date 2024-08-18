## FLASK APP WHICH LOADS THE TRAINED DATA MODEL AND CONNECTS TO THE FRONT END

##change directory to venv

`cd MSc-project-Backend/venv`

##INSTALL THE PACKAGES USING BELOW COMMANDS

`pip install torch torchvision torchaudio`

`pip install transformers`

`pip install flask-cors`

##CHANGE THE PATH TO THE SAVED HYBRID MODEL DIRECTORY FOLDER WHICH IS PRESENT IN THIS REPO `https://github.com/sc23mm/MSc-Project` or `https://leeds365-my.sharepoint.com/:f:/g/personal/sc23mm_leeds_ac_uk/EuIZg_zdnFNLjeNp9vOYBakBrYcgUKxjqEQIn3VroOfv7Q?e=cOSxAg` 

in the `app.py` file line no 21 `model_dir = "give your own directory path to the model saved_model_hybrid_one" `

##RUN THE APP

`python app.py`

##AFTER RUNNING NOTE THE LOCAL IP ADDRESS TO USE IN THE FRONTEND. eg 'http://127.0.0.1:5000' 
