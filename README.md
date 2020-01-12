# Overview
This project takes a set of positive and negative movie reviews to build a machine learning model for sentiment analyses.

Currently it uses a bag of words model. The bag of words consist of adjectives (ADJ), ngeated adjectives (not, ADJ) and verbs (VERB),  (not, VERB)  that have followed the word "I" . It selects optimal amount of features based un a chi2 test and selects the best classifier based on the best f-score. 
# Requirements
* Python 3+ 64bit
* Linux

# How to run

1. Open terminal and cd into the repository you wish to clone the project
2. Clone the git repo: `git clone https://github.com/c1531958/CMT307_Part2.git`
3. cd into the repository `cd CMT307_Part2`
4. Use pipenv to create a virtual environment and install requirements
```sh
    pipenv run pip install -r requirements.txt
```
5. Optional. If you would like to use your own sentiment training, dev and testing files, place them in the main directory and make sure they are organised as follows. Alternatively, you may use the provided IMDb files.
```
Your_file_folder_name/train/imdb_train_neg.txt
Your_file_folder_name/train/imdb_train_pos.txt
Your_file_folder_name/dev/imdb_dev_neg.txt
Your_file_folder_name/dev/imdb_dev_pos.txt
Your_file_folder_name/test/imdb_test_neg.txt
Your_file_folder_name/test/imdb_test_pos.txt
```
6. Run the following command to run the code. Replace IMDb with your own folder name if you wish to use your own files
```
    pipenv run python main.py IMDb
```
