from flask import Flask, render_template, request, redirect, send_file, url_for
from bs4 import BeautifulSoup as bs4
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import requests
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import time
import nltk
import regex as re
import tensorflow_text as text
import tensorflow_hub as hub
import json
import newspaper
import csv
import article_analysis
import frameworkfunctions
import xlsxwriter
import excel_output
from flask_sqlalchemy import SQLAlchemy
import time

csv_time = ""
app = Flask(__name__)
app.config["CACHE_TYPE"] = "null"
app.config["SQLALCHEMY_DATABSE_URI"] = 'sqlite:///websitelist.db'
double = r"!([\w ]+)!"

db = SQLAlchemy(app)


class selected_articles(db.Model):
    #SQL class for articles identified as relevant
    id = db.Column(db.Integer, primary_key=True)
    article_headline = db.Column(db.String(), nullable=False)
    article_link = db.Column(db.String, nullable=False)
    article_subject = db.Column(db.String)
    article_sentiment = db.Column(db.Integer)
    article_summary = db.Column(db.String)
    article_internal_code = db.Column(db.String, unique=True, nullable=False)

    def __repr__(self):
        return "{}||{}||{}||{}||{}".format(article_headline, article_subject, article_link, article_sentiment, article_summary)


class not_selected_article(db.Model):
    #SQL class for articles identified as not-relevant
    id = db.Column(db.Integer, primary_key=True)
    article_headline = db.Column(db.String(), unique=True, nullable=False)
    article_link = db.Column(db.String, unique=True, nullable=False)
    article_subject = db.Column(db.String)

    def __repr__(self):
        return "{}||{}||{}".format(article_headline, article_link, article_subject)


db.create_all()
db.session.commit()

#ML modles are loaded
classifier_model = tf.keras.models.load_model(
    "models/Classifier_Model_Collab", compile=False)
sentimental_classifier = tf.keras.models.load_model(
    "models/Sentiment_Classifier_Collab", compile=False)


@app.route("/", methods=["POST", "GET"])
def index():
    """Function handles the initial portions of handling of the users query,
    sorting them into a list, creating the initial names of the export and
    calling the relevant html webpages after entire ML porting of the
    web-app has run"""
    if request.method == "POST":
        start = time.time()
        entry = request.form["keywords"]
        if entry:
            big_inter = re.findall(double, entry)
            small_inter = re.sub(double, "", entry)
            small_inter = small_inter.split(" ")
            small_inter = frameworkfunctions.clean_list(small_inter)
            key_words = small_inter+big_inter
            key_words = [word.lower() for word in key_words]
        else:
            key_words = article_analysis.big_query_list
        name_of_export = "/root/WebSentimentWebsite/static/csv_outputs/output"
        _, total_articles, articles_of_interest, inter_csv_time, plot_time = article_analysis.get_headlines(
            export=True, NN_model=classifier_model, name_of_export=name_of_export, AI_filter=True, big_query_list=key_words, BERT=True, sentiment=True, sentiment_model=sentimental_classifier, bs4_call=True)
        end = time.time()
        print(end-start)
        if not articles_of_interest:
            return render_template("nothing_found.html")
        if plot_time:
            image_source = "images/output" + plot_time + ".jpg"
        else:
            image_source = "images/notfound.jpg"
        image_source = url_for('static', filename=image_source)
        global csv_time
        csv_time = inter_csv_time
        return render_template("output.html", source=image_source, total_articles=total_articles, articles_of_interest=articles_of_interest)

    else:
        return render_template("index.html")


@app.route("/getPlotCSV")
def plot_csv():
    global csv_time
    return send_file("/root/WebSentimentWebsite/static/csv_outputs/output" + csv_time + ".xlsx",
                     attachment_filename="Results.xlsx",
                     as_attachment=True, cache_timeout=0)


if __name__ == "__main__":
   app.run(debug=True)
