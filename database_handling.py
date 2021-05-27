from sentimentapp import selected_articles, not_selected_article, db
import pandas as pd
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import exc


def add_selected(df):
    """Adds new selected headlines into relevant table in SQL"""
    for index, row in df.iterrows():
        code = row["article_headline"]+row["subject"]
        new_selected = selected_articles(article_headline=row["article_headline"], article_subject=row["subject"], article_link=row["article_link"], article_sentiment=int(
            row["sentiment"]), article_summary=row["summary"], article_internal_code=code)
        try:
            db.session.add(new_selected)
            db.session.commit()
        except exc.IntegrityError:
            db.session.rollback()
            continue


def add_not_selected(df):
    """Adds new unselected headlines into relevant table in SQL"""
    for index, row in df.iterrows():
        new_not_selected = not_selected_article(
            article_headline=row["article_headline"], article_subject=row["subject"], article_link=row["article_link"])
        try:
            db.session.add(new_not_selected)
            db.session.commit()
        except exc.IntegrityError:
            db.session.rollback()
            continue


def deal_new(df):
    """Performs search of SQL database, identifying & returning headlines already previously.Significantly reducing code run time"""
    previous_selected_list = []
    previous_unselected_list = []
    new = []
    for index, row in df.iterrows():
        code = row["article_headline"]+row["subject"]
        previous_selected = selected_articles.query.filter_by(article_internal_code=code).first()
        previous_unselected = not_selected_article.query.filter_by(
            article_headline=row["article_headline"]).first()
        if previous_selected:
            previous_selected_list.append([previous_selected.article_headline, previous_selected.article_subject,
                                           previous_selected.article_link, previous_selected.article_sentiment, previous_selected.article_summary])
            continue
        if previous_unselected:
            previous_unselected_list.append(
                [previous_unselected.article_headline, previous_unselected.article_subject, previous_unselected.article_link])
            continue
        new.append([row["article_headline"], row["subject"], row["article_link"]])
    # Dataframe creation
    previous_selected_df = pd.DataFrame(previous_selected_list, columns=[
                                        "article_headline", "subject", "article_link", "sentiment", "summary"])
    previous_unselected_df = pd.DataFrame(previous_unselected_list, columns=[
                                          "article_headline", "subject", "article_link"])
    new_df = pd.DataFrame(new, columns=["article_headline", "subject", "article_link"])
    return previous_selected_df, previous_unselected_df, new_df
