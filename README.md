# type-predictor

Installation on debian-based system

 1. Instal pip from apt
 > apt install python3-pip
 
 2. Install requred python packages
 > pip3 install joblib scikit-learn python-speech-features prettytable matplotlib pandas seaborn
 
 
 ## Train model
 
 ./main -wav /dir/
 
 
 ## Predict 
 
 ./main --wav /path/to/test/audio/file.wav --attack /path/to/pretrained/model/model.file
 
