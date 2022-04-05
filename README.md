# Twitter Sentiment Web App
A simple web application to serve a machine learning model to predict sentiments of tweets. The [`master`](https://github.com/xofbd/sentiment-app/tree/master) branch is the starting point of this project with the [`flask-app`](https://github.com/xofbd/sentiment-app/tree/flask-app) branch having a minimal working [Flask](https://flask.palletsprojects.com/) application.

## Data
The training data is a collection of tweets labeled as positive (1) or negative (0) sentiment. More about the data can be read [here](http://thinknook.com/twitter-sentiment-analysis-training-corpus-dataset-2012-09-22/) and it can be downloaded [here](http://thinknook.com/wp-content/uploads/2012/09/Sentiment-Analysis-Dataset.zip).

## Usage
To launch the application locally:

* Create the `.env` file. You can use the `.env.template` as an example.
* With GNU Make, run `make all`. That will take care of creating the virtual environment and trained machine learning model.
* Without GNU Make, you'll need to use `requirements.txt` to create the virtual environment and run the `app/model/model.py` script to generate the serialized machine learning model. Finally, run `flask run`.

## License
This project is distributed under the GNU General Public License. Please see `COPYING` for more information.
