#!/usr/bin/env python
# coding: utf-8

# In[1]:


# First things first we need to go ahead and pull news articles related to vessel freights/costs/transit etc in the last 2 weeks
# For this we will use BeautifulSoup


# In[2]:


#get_ipython().run_line_magic('matplotlib', 'notebook')
#get_ipython().run_line_magic('config', 'Completer.use_jedi = False')

from nltk.tokenize import RegexpTokenizer
import re
from nltk.corpus import wordnet
from bs4 import BeautifulSoup as bs4
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import requests
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import pandas as pd
import time
import nltk
import regex as re
import tensorflow_text as text
#from official.nlp import optimization
import tensorflow_hub as hub
import json
import newspaper
import csv
import time
import excel_output
import database_handling

import warnings
warnings.filterwarnings('ignore')

def fix_query(query):
    """Fixes the query to allow for it to be searched via Google"""
    return "%20".join(query.split(" "))


#Additional key words to add to usery queries
key_word_list = ["Stock", "Price", "Growth", "Production",
                 "Demand", "Supply", "Outlook", "Market", "Freight"]


def get_data(big_query_list, last_week=True, bs4_call=False, word_list=key_word_list):
    """Main function that searches the queries via google, creates a list of articles found,
    articles already explored not skipped"""

    articles_so_far = 0
    article_link_pre = "https://news.google.com/"
    article_headlines = []
    article_link = []
    subject_list = []
    for key_word in key_word_list:
        for root_query in big_query_list:
            query = root_query + " " + key_word
            query_fixed = fix_query(query)
            if last_week:
                search_term = "https://news.google.com/search?q=" + \
                    query_fixed + "%20when%3A7d&hl=en-CA&gl=CA&ceid=CA%3Aen"
            else:
                search_term = "https://news.google.com/search?q=" + query_fixed + "&hl=en-CA&gl=CA&ceid=CA%3Aen"
            if bs4_call:
                r = requests.get(search_term)
                soup = bs4(r.text, "html.parser")
                haedline_list = soup.findAll("h3", "ipQwMb ekueJc RD0gLb")
            else:
                driver = webdriver.PhantomJS()
                driver.get(search_term)
                haedline_list = driver.find_elements_by_xpath("//h3[@class='ipQwMb ekueJc RD0gLb']")
            for item in haedline_list:
                article_headline = item.text
                if article_headline in article_headlines:
                    continue
                if root_query.lower() not in article_headline.lower():
                    continue
                article_headlines.append(article_headline)
                if bs4_call:
                    article_link.append(article_link_pre + item.find("a")["href"][2:])
                else:
                    article_link.append(item.find_element_by_tag_name("a").get_attribute("href"))
                subject_list.append(root_query)
                articles_so_far += 1
            time.sleep(1)
    print("Total news headlines analyzed: {}".format(articles_so_far))
    df = pd.DataFrame(list(zip(article_headlines, subject_list, article_link)),
                      columns=["article_headline", "subject", "article_link"])
    return df, articles_so_far


#Query list when user preforms a blank search
big_query_list = ["Vessel",
                  "Ship",
                  "Vessel Charges",
                  "Aluminum",
                  "Rail Cars",
                  "Nickel",
                  "Copper",
                  "Commodity Industry",
                  "Steel",
                  "Lithium",
                  "Lead Acid Batteries",
                  "Fertilizer",
                  "Acid",
                  "Pulp and Paper"
                  "Industrial Chemical",
                  "Sulfur",
                  "Vanadium",
                  "Industrial Chemicals",
                  "Lead",
                  "Zinc",
                  "High Value Metals",
                  "Alum and Water Treatment"
                  "Glencore"]

def get_wordnet_pos(treebank_tag):
    """pos tags words"""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def tokenize_data(features, train=True, tokenizer_layer_pre=None, number_expression=r"([\d]+)",
                  percentage_expression=r"([\d ]+%)"):
    """Function tokenizes the headlines and returns the tokenizer layer"""
    lemmatizer = nltk.stem.WordNetLemmatizer()

    features = np.array([re.sub(number_expression, " number ", re.sub(
        percentage_expression, " percentage ", feature)) for feature in features])

    features = np.array([" ".join([lemmatizer.lemmatize(text[0].lower(), get_wordnet_pos(text[1])) for
                                   text in nltk.pos_tag(nltk.word_tokenize(feature))]) for feature in features])
    if train:
        tokenizer_layer = tf.keras.preprocessing.text.Tokenizer(oov_token=100)
        tokenizer_layer.fit_on_texts(features)
    else:
        tokenizer_layer = tokenizer_layer_pre
    tokenized_features = tokenizer_layer.texts_to_sequences(features)

    return tokenized_features, tokenizer_layer




def pad_sequeunces(tokenized, max_len=50):
    """Pad sequences so they are all the same lenght"""
    padded_data = []
    for line in tokenized:
        padded_data.append(np.pad(line, (0, (max_len)-len(line)), constant_values=(0)))
    return np.array(padded_data)


def to_texts(text_data, tokenizer):
    """Converts tokenized data back into strings"""
    reverse_words = dict(map(reversed, tokenizer.word_index.items()))
    texts = []
    for text in text_data:
        texts.append((" ").join([str(reverse_words[char]) for char in text if char != 0]))
    return texts


def clean_text(dirty_text):
    """A basic function that cleans the data for common problems/acronyms"""
    clean_text = dirty_text.replace("\n\n\n", "\n\n")
    clean_text = clean_text.replace("\n\n", "\n")
    clean_text = clean_text.replace("\n", " ")
    clean_text = clean_text.replace("U.S.", "United States")
    clean_text = clean_text.replace("u.s.", "United States")
    return clean_text


def summarizer(text, keywords, summary_sentences=5):
    """Summarizer function, the 'read' text is summarized using IDF, user
    defined lines is outputed"""
    if type(text) == list:
        text = ". ".join(text)
    text = clean_text(text)
    tokens = nltk.tokenize.word_tokenize(text)
    freq = nltk.FreqDist(tokens)
    max_freq = max(freq.values())
    sorted_dict = {}
    for i, j in freq.items():
        sorted_dict[i] = j/max_freq
    spits = text.split(". ")
    sentences_library = {}
    for sentence in spits:
        sentence_score = 0
        word_counter = 0
        for word in nltk.tokenize.word_tokenize(sentence):
            if word in [":,.; "]:
                continue
            word_counter += 1
            try:
                sentence_score += sorted_dict[word]
            except:
                sentence_score += 0
        key_word_not_found = True
        if keywords:
            for key in keywords:
                if key in sentence.lower():
                    key_word_not_found = False
        else:
            key_word_not_found = False
        if key_word_not_found:
            continue
        try:
            normalized_score = sentence_score/word_counter
            # normalized_score=sentence_score
        except:
            continue
        sentence = sentence+"\n"
        sentences_library[sentence] = normalized_score
    sorted_lib = sorted(sentences_library, key=sentences_library.get, reverse=True)
    return sorted_lib[:summary_sentences]


def custom_summarize_article(article_link, sentences_to_return=5):
    """Function reads the article and sends over the read text to 'summarizer'"""
    try:
        article = newspaper.Article(article_link)
        article.download()
        article.parse()
        article.nlp()
        return summarizer(article.text, article.keywords)
    except:
        return "Summary Not Possible"


def paid_summarize_article(article_link, sentences_to_return=5):
    """Function uses smmry to summarize articles, not used in deployment"""
    data = {"SM_Length": str(sentences_to_return)}
    data = json.dumps(data)
    r = requests.post("https://api.smmry.com/&SM_API_KEY=C59BD46316&SM_URL=" +
                      article_link, json=data)
    try:
        return r.json()["sm_api_content"]
    except:
        return "Summarizing Error"


def plot_sentiment(scores, queries, e_name, graphical=True,):
    """Function plots the overall sentiment for the user's queries, if applicable"""
    if graphical:
        zipper = list(zip(scores, queries))
        pos_ = [[s, q] for s, q in zipper if s > 0]
        neg_ = [[s, q] for s, q in zipper if s < 0]
        nue_ = [[s, q] for s, q in zipper if s == 0]

        fig = plt.figure()
        breadth = len(scores)*0.125
        ax = fig.add_axes([0, 0, breadth, 1])

        ordered_scores = []

        for sample in pos_:
            ax.bar(sample[1], sample[0], color="g", width=0.2)
            ordered_scores.append(sample[0])

        for sample in nue_:
            ax.bar(sample[1], sample[0], color="b", width=0.2)
            ordered_scores.append(sample[0])

        for sample in neg_:
            ax.bar(sample[1], sample[0], color="r", width=0.2)
            ordered_scores.append(sample[0])

        plt.ylim([-150, 150])
        plt.yticks([-100, -75, -50, -25, 0, 25, 50, 75, 100])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        for i, v in enumerate(ordered_scores):
            if v > 0:
                ax.text(i, v+10, str(v) + "% Bull", color="green", horizontalalignment="center")
            elif v == 0:
                ax.text(i, v+10, "neutral", color="black", horizontalalignment="center")
            else:
                ax.text(i, v-10, str(v) + "% Bear", color="red", horizontalalignment="center")

        plt.axhline(y=0, color='black', linestyle='-')
        plt.tight_layout()

        plot_time = str(time.time())
        plt.savefig(
            "/root/WebSentimentWebsite/static/images/output" + plot_time + ".jpg", bbox_inches="tight")

        return plot_time


def score_sentiment(headlines_sentiment, query, show_text=True):
    """Function calculates total sentimental score for the user's queries,
    if applicable"""
    score = 0
    for sentiment in headlines_sentiment:
        if sentiment == 0:
            score += 0
        elif sentiment == 1:
            score -= 1
        else:
            score += 1

    max_score = len(headlines_sentiment)
    min_score = len(headlines_sentiment)*-1

    if score > 0:
        sentiment_word = "bullish"
        return_score = int(np.round((score/max_score)*100, 0))
        if show_text:
            print("Sentiment for '{}' is {:.0f}% {}".format(query, (return_score), sentiment_word))
    elif score == 0:
        sentiment_word = "Neutral"
        return_score = 0
        if show_text:
            print("Sentiment for '{}' is {}".format(query, sentiment_word))
    else:
        sentiment_word = "Bearish"
        return_score = -int(np.round((score/min_score)*100, 0))
        if show_text:
            print("Sentiment for '{}' is {:.0f}% {}".format(query, (return_score), sentiment_word))

    return return_score, query


def split_sentences(headline, root_query):
    """Function splits headline into different sentences, shortlists the sentences
    where the actual query can be found"""
    tokenizer = RegexpTokenizer(r'\w+')
    headline = headline.replace(";", ",")
    splits = headline.split(",")

    if len(splits) == 1:
        return [headline]

    shortlisted_phrases = []
    for split in splits:
        if root_query.lower() in split.lower():
            if len(tokenizer.tokenize(split)) < 1:
                return [headline]
            else:
                shortlisted_phrases.append(split)

    return shortlisted_phrases


def get_market_sentiment(query_list, dataset, scoring_model, e_name, previous_found):
    """Function scores eaach of the relevant shortlisted sentences from 'split sentences',
    then proceeds to calculate overall headline score"""
    sentiments = np.array([])
    graphing_score = []
    graphing_query = []
    first_run = True
    plot_time = False
    for query in query_list:
        headline_sentiment = []
        query_headlines = (dataset[dataset["subject"] == query]["article_headline"])
        previous_headlines = (
            previous_found[previous_found["subject"] == query]["article_headline"])
        previous_scores = (previous_found[previous_found["subject"] == query]["sentiment"])
        indexes = (dataset[dataset["subject"] == query].index)
        for headline, index in zip(query_headlines, indexes):
            shortlisted_phrases = split_sentences(headline, query)
            total_sentiment = 0
            for phrase in shortlisted_phrases:
                inter_sentiment = np.argmax(scoring_model.predict([phrase]), axis=-1)
                inter_sentiment = inter_sentiment[0]
                if inter_sentiment == 2:
                    total_sentiment += 1
                elif inter_sentiment == 1:
                    total_sentiment -= 1
            if total_sentiment > 0:
                inter_headline_sentiment = 2
            elif total_sentiment < 0:
                inter_headline_sentiment = 1
            else:
                inter_headline_sentiment = 0
            headline_sentiment.append([inter_headline_sentiment, index])
        headline_sentiment = np.array(headline_sentiment)
        # headline_sentiment=np.argmax(scoring_model.predict(query_headlines),axis=1)
        if first_run and len(headline_sentiment) > 0:
            sentiments = headline_sentiment
            first_run = False
        elif len(headline_sentiment) > 0:
            sentiments = np.r_[sentiments, headline_sentiment]
        if (len(query_headlines)+len(previous_headlines)) < 5:
            print("Too little data for '{}' to predict sentiment".format(query))
            continue
        # print(previous_scores)
        if len(headline_sentiment) > 0:
            total_query_sentiment = np.r_[
                np.array(headline_sentiment[:, 0]), previous_scores.values]
        else:
            total_query_sentiment = previous_scores.values
        intermediate = score_sentiment(total_query_sentiment, query)
        graphing_score.append(intermediate[0])
        graphing_query.append(intermediate[1])
    if graphing_score:
        plot_time = plot_sentiment(graphing_score, graphing_query, e_name)
    return sentiments, plot_time


def get_headlines(AI_filter=True, NN_model=None, export=False, tokenizer=None, headlines_col="article_headline",
                  articles_link="article_link", subject_header="subject", cut_off=0.5, name_of_export=None,
                  last_week=True, big_query_list=big_query_list, AI_in_deployment=False, sentiment=False, sentiment_model=None, bs4_call=False,
                  summarize_articles=True, paid_summarizer=False, custom_summarizer=True):
    """Function handles the queries received from the user, calls relevant functions to Google the queries,
    tokenizes the returned headlines, identifies ones that are relevant. Calls relevant sub-functions to score the headlines,
    score  the sentiment and plot the sentiment. Function performs calls to relevant sub functions that handle article
    summarization. Function perfroms calls to relevant subfunctions that handle the SQL database side of the application.
    Function creates and saves the relevant export files"""
    dataset, total_articles_found = get_data(big_query_list, last_week, bs4_call)
    no_new = False
    if AI_filter:

        previous_selected, previous_unselected, dataset = database_handling.deal_new(dataset)

        if AI_in_deployment:
            prepared = dataset[headlines_col]
        else:
            headlines = dataset[headlines_col]
            prepared, _ = tokenize_data(np.array(headlines).reshape(-1,),
                                        train=False, tokenizer_layer_pre=tokenizer)
            prepared = pad_sequeunces(prepared)

        if (len(prepared)+len(previous_selected)) == 0:
            print("No articles found")
            return False, False, False, False, False
        if len(prepared) > 0:
            predictions = NN_model.predict(prepared)
            dataset["AI_preds"] = predictions

            selected_dataset = dataset.loc[dataset["AI_preds"] >
                                           cut_off][[headlines_col, subject_header, articles_link]]
            not_selected_dataset = dataset.loc[dataset["AI_preds"] <
                                               cut_off][[headlines_col, subject_header, articles_link]]
        else:
            null_data = []
            selected_dataset = pd.DataFrame(
                null_data, columns=["article_headline", "subject", "article_link"])
            not_selected_dataset = []

        articles_of_interest = len(selected_dataset)+len(previous_selected)
        print("Articles of potential interest: {}".format(articles_of_interest))

        if articles_of_interest == 0:
            return False, False, False, False, False

        if sentiment:
            query_list = np.unique(
                np.r_[selected_dataset["subject"].values, previous_selected["subject"].values])
            inter, plot_time = get_market_sentiment(query_list, selected_dataset,
                                                    sentiment_model, e_name=name_of_export, previous_found=previous_selected)
            selected_dataset["sentiment"] = ""
            for row in inter:
                selected_dataset["sentiment"].loc[row[1]] = row[0]

        if summarize_articles:
            article_links = selected_dataset["article_link"].values
            articles_summarized = []
            for link in article_links:
                if custom_summarizer:
                    articles_summarized.append("".join(custom_summarize_article(link)))
                elif paid_summarizer:
                    articles_summarized.append(paid_summarize_article(link))
                else:
                    break
            if len(selected_dataset) > 0:
                selected_dataset["summary"] = np.array(articles_summarized)
                database_handling.add_selected(selected_dataset)
                if len(not_selected_dataset) > 0:
                    database_handling.add_not_selected(not_selected_dataset)
                frames = [selected_dataset, previous_selected]
                selected_dataset = pd.concat(frames)
            else:
                selected_dataset = previous_selected

        if export:
            selected_dataset["sentiment"] = selected_dataset["sentiment"].apply(
                lambda x: "Positive" if x == 2 else ("Negative" if x == 1 else "Neutral"))
            selected_dataset = selected_dataset.sort_values(by=["subject"])
            current_time_csv = str(time.time())
            excel_output.save_xlsx(selected_dataset, current_time_csv)
            return selected_dataset, total_articles_found, articles_of_interest, current_time_csv, plot_time
        else:
            current_time_csv=""
            return selected_dataset,total_articles_found, articles_of_interest, current_time_csv, plot_time
    else:
        return dataset
    return dataset, total_articles_found, articles_of_interest
