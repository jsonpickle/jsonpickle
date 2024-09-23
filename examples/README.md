## Purpose
This directory exists to provide examples of how to use jsonpickle to achieve simple goals, such as:
- Saving instances of a class to a file
- And more to come!

## Running the code
### Linux (Ubuntu)
If you don't have python and/or git installed, install those first:
```sh
apt install git python3 python3-dev python3-pip
```
Once you have those installed, start by cloning the repository and entering into the directory (or copying the file locally):
```sh
git clone https://github.com/jsonpickle/jsonpickle.git && cd jsonpickle/examples
```
Then, you should install jsonpickle. The preferred method, using a virtual environment (which isolates packages), goes as so:
```sh
# DO NOT RUN LINES BEGINNING WITH #
# Create a virtualenv in the "env" directory. You don't need to run this again.
python3 -m venv env

# You must activate the virtualenv anytime you open a new shell in order to
# make the to-be-installed jsonpickle package available. You should run this
# in the same directory that you ran the previous command in.
source env/bin/activate

pip install jsonpickle
python3 save_class_to_file.py
```
If you'd rather install jsonpickle globally though, you can just run this:
```sh
pip install jsonpickle
```
Lastly, you can run any example file as long as you have jsonpickle and Python installed!
```sh
python3 save_class_to_file.py
```
You can also open the .py file in your favorite text editor, such as VSCode, gedit, or others!
### Windows
If you don't have python, pip, and/or git installed, install those first. Instructions are not included here due to their verbosity, but there exist online guides for this!
Once you have those installed, start by cloning the repository in a terminal (such as cmd prompt) and entering into the directory (or copying the file locally):
```sh
git clone https://github.com/jsonpickle/jsonpickle.git
cd jsonpickle/examples
```
Then, you should install jsonpickle in the same terminal. The preferred method, using a virtual environment (which isolates packages), goes as so:
```sh
pip install virtualenv

# DO NOT RUN LINES BEGINNING WITH #
# Create a virtualenv in the "env" directory. You don't need to run this again.
virtualenv venv

# You must activate the virtualenv anytime you open a new shell in order to
# make the to-be-installed jsonpickle package available. You should run this
# in the same directory that you ran the previous command in.
.\venv\Scripts\activate

pip install jsonpickle
python save_class_to_file.py
```
If you'd rather install jsonpickle globally though, you can just run this:
```sh
pip install jsonpickle
```
Lastly, you can run any example file as long as you have jsonpickle and Python installed!
```sh
python save_class_to_file.py
```
You can also open the .py file in your favorite text editor, such as VSCode, Notepad++, or others!
