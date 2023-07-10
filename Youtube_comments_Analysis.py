#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Importing the Libraries

import pandas as pd
import numpy as np
from googleapiclient.discovery import build
import os
from googleapiclient.errors import HttpError
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
import json
import requests
from bs4 import BeautifulSoup
import pandas as pd 
import regex as re


# In[187]:


# Arguments that need to be passed to the build function
DEVELOPER_KEY = '#########'


# In[3]:


links = pd.read_csv('vdoLinks.csv')
links.shape


# In[4]:


links.head()


# In[5]:


links = list(links['youtubeId'])


# In[188]:


youtube_object = build("youtube", "v3", developerKey = DEVELOPER_KEY)


# In[155]:


df_data = pd.DataFrame(columns=['title','description','view_count','like_count','dislike_count','comment_count',
                                'duration','favorite_count'])
for link in links:
    data = {}
    video_details = youtube_object.videos().list(
            part='snippet, statistics, contentDetails',
            id=link).execute()
    
    if len(video_details['items']) != 0:
        title = video_details['items'][0]['snippet']['title']
        data['title'] = title

        description = video_details['items'][0]['snippet'].get('description', 'NULL')
        data['description'] = description

        view_count = video_details['items'][0]['statistics'].get('viewCount', 0)
        data['view_count'] = view_count

        like_count = video_details['items'][0]['statistics'].get('likeCount', 0)
        data['like_count'] = like_count

        dislike_count = video_details['items'][0]['statistics'].get('dislikeCount', 0)
        data['dislike_count'] = dislike_count

        comment_count = video_details['items'][0]['statistics'].get('commentCount', 0)
        data['comment_count'] = comment_count

        duration = video_details['items'][0]['contentDetails'].get('duration', 0)
        data['duration'] = duration

        favorite_count = video_details['items'][0]['statistics'].get('favoriteCount', 0)
        data['favorite_count'] = favorite_count

        df_data = df_data.append(data, ignore_index=True)
df_data.head()


# In[156]:


df_data.shape


# In[157]:


df_data.to_csv('video_df.csv')


# In[192]:


df_comments = pd.DataFrame(columns=['comments', 'link'])

for link in links:
    comments_data = {}
    try:
        video_comments = []
        comments_data['link'] = link
        comments = youtube_object.commentThreads().list(
                    part='snippet',
                    videoId=link,
                    maxResults=100
                ).execute()
        
        for comment in comments['items']:
            video_comments.append(comment['snippet']['topLevelComment']['snippet']['textDisplay'])
        comments_data['comments'] = video_comments
        df_comments = df_comments.append(comments_data, ignore_index=True)
        
    except HttpError as error:
            if error.resp.status == 404:
                pass
df_comments.head()


# In[193]:


df_comments.shape


# In[194]:


df_comments.to_csv('comments_df.csv')


# In[ ]:





# ## Importing libraries

# In[6]:


import pandas as pd
import matplotlib.pyplot as plt


# ## Top 10 videos based on view count

# In[7]:


video_df = pd.read_csv('video_df.csv', usecols=range(1,9))
video_df.head()


# In[5]:


videos_sorted_df = video_df.sort_values(by=['view_count'], ascending=False)
videos_sorted_df[:10]


# In[4]:


videos_sorted_df[['title', 'view_count']][:10]


# In[5]:


fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(videos_sorted_df[:10]['title'], videos_sorted_df[:10]['view_count'])
 
plt.xlabel("Movie names")
plt.ylabel("Number of views")
plt.xticks(rotation= 90)
plt.title("Top 10 videos based on total views")
plt.show()


# ## Bottom 10 videos based on view count

# In[6]:


videos_sorted_df[['title', 'view_count']][-10:]


# In[8]:


bottom_videos = videos_sorted_df[(videos_sorted_df['view_count'] == 1) | (videos_sorted_df['view_count'] == 2)][-10:]
bottom_videos[['title', 'view_count']][-10:]


# In[8]:


fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(bottom_videos['title'], bottom_videos['view_count'])
 
plt.xlabel("Movie names")
plt.ylabel("Number of views")
plt.xticks(rotation= 90)
plt.title("Bottom 10 videos based on total views")
plt.show()


# ## Most liked video

# In[9]:


video_df.head()


# In[10]:


most_liked_video = video_df[video_df['like_count'] == video_df['like_count'].max()]
most_liked_video


# ## Least liked video

# In[11]:


not_liked_video = video_df[video_df['like_count'] == video_df['like_count'].min()]
not_liked_video


# In[12]:


least_liked_video = video_df[video_df['like_count'] == 1]
least_liked_video


# ## Video with highest duration

# In[8]:


import isodate

def duration_to_seconds(df):
    dur = isodate.parse_duration(df)
    return (dur.total_seconds())


# In[9]:


#converting duration to seconds
video_df['duration'] = video_df['duration'].apply(duration_to_seconds)
video_df.head()


# In[10]:


high_duration_video = video_df[video_df['duration'] == video_df['duration'].max()]
high_duration_video


# ## Sentiment Analysis

# In[28]:


comments_df = pd.read_csv('comments_df.csv', usecols=range(1,3))


# In[29]:


links_df = pd.read_csv('vdoLinks.csv')
links_df.rename(columns={'youtubeId':'link'}, inplace=True)
links_df.head()


# In[23]:


links_df.info()


# In[30]:


comments_df.info()


# In[31]:


import numpy as np
comments_df = pd.merge(comments_df, links_df[['link', 'title']], on='link', how='left' )
comments_df.head()


# In[32]:


comments_df.shape


# In[33]:


comments_df.info()


# ## Applying VADER sentiment analysis on comments

# In[34]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from bs4 import BeautifulSoup
import unicodedata


# In[35]:


analyser = SentimentIntensityAnalyzer()


# In[36]:


comments_df['comments'][0]


# In[37]:


def remove_punctuation_func(text):
    return re.sub(r'[^a-zA-Z0-9]', ' ', text)

def remove_numbers(text):
    return re.sub(r'[0-9]', ' ', text)

def remove_html_tags_func(text):
    return BeautifulSoup(text, 'html.parser').get_text()

def remove_url_func(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def remove_accented_chars_func(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

def remove_extra_whitespaces_func(text):
    return re.sub(r'^\s*|\s\s*', ' ', text).strip()


# In[38]:


comments_df.info()


# In[48]:


df_scores = pd.DataFrame(columns=['title', 'scores', 'sentiment'])
for i in range(comments_df.shape[0]):
    comments_list = re.findall("\'([^\\']*)\'", comments_df['comments'][i])
    if len(comments_list)>0:
        scores = []
        title = comments_df['title'][i]
        for comment in comments_list:
            comment = comment.lower()
            comment = remove_punctuation_func(comment)
            comment = remove_numbers(comment)
            comment = remove_html_tags_func(comment)
            comment = remove_url_func(comment)
            comment = remove_accented_chars_func(comment)
            comment = remove_extra_whitespaces_func(comment)
            score = analyser.polarity_scores(comment)
            scores.append(-score['neg'] if score['neg']>score['pos'] else score['pos'])
        sentiment = sum(scores) / len(scores)
        df_scores = pd.concat([df_scores, pd.DataFrame.from_records([{'title':title,'scores':scores, 'sentiment':sentiment }])], ignore_index=True)
    
df_scores.head()


# In[46]:


df_scores.shape


# In[49]:


df_scoressort = df_scores.sort_values(by=['sentiment'], ascending=False)
df_scoressort[:10]


# In[50]:


fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(df_scoressort['title'][:10], df_scoressort['sentiment'][:10])
 
plt.xlabel("Movie names")
plt.ylabel("sentiment")
plt.xticks(rotation= 90)
plt.title("Top 10 videos having positive sentiment")
plt.show()


# In[ ]:




