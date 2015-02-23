# -*- coding: utf-8 -*-
import pickle
from twython import Twython
import datetime
import time
import pandas as pd
import numpy as np
from pandas.io.pickle import read_pickle

TWTR_APP_KEY = ""
TWTR_APP_SECRET = ""

def get_access_token():    
    # Get the OAuth2 access token and pickle it for future use

    twitter = Twython(TWTR_APP_KEY, TWTR_APP_SECRET, oauth_version=2)
    TWTR_ACCESS_TOKEN = twitter.obtain_access_token()
    pickle.dump(TWTR_ACCESS_TOKEN, open("acc_token.pkl", "wb"))
    
def get_twitter_conn():    
    # Returns an OAuth2 authenticated twitter connection
    # which is used for read-only access but in exchange we get a large rate limit
    
    TWTR_ACCESS_TOKEN = pickle.load(open("acc_token.pkl", "rb"))
    return Twython(TWTR_APP_KEY, access_token = TWTR_ACCESS_TOKEN)     
    
def check_twitter_timeout(twitter):    
    # Makes sure we avoid the Twitter timeout error by checking
    # the number of calls remaining and sleeping if necessary.
    
    if twitter.get_lastfunction_header(header='x-rate-limit-remaining') is not None:
        calls_remaining = int(twitter.get_lastfunction_header(header='x-rate-limit-remaining'))
        next_reset_time = float(twitter.get_lastfunction_header(header='x-rate-limit-reset'))
        if (calls_remaining == 1) and (next_reset_time > time.time()):               
            remainder = next_reset_time - time.time() + 120.0
            print "Hit Rate Limit at {} and need to sleep for {} minutes.".format(datetime.datetime.now().strftime("%H:%M:%S"), str(remainder / 60.0))
            time.sleep(remainder) 

def get_twt_history():    
    # Get the last 3200 tweets on the Chase account and pickle the results.
    # Total amount of tweets is limited by Twitter, unless you're accessing your own account.
    
    twitter = get_twitter_conn()
    
    last_max_id = None
    chase_timeline = []
    
    while True:
        if last_max_id is None:
            timeline_results = twitter.get_user_timeline(screen_name = 'Chase', count = 200, include_rts = True)
        else:
            timeline_results = twitter.get_user_timeline(screen_name = 'Chase', count = 200, include_rts = True, max_id = last_max_id-1)
        
        if len(timeline_results) == 0:
            break
        else:
            chase_timeline.extend(timeline_results)
            last_max_id = timeline_results[-1]['id']
            check_twitter_timeout(twitter)
        
    pickle.dump(chase_timeline, open("chase_timeline.pkl", "wb"))

def get_rwtwt_users_and_data():    
    # Find all of the tweets by Chase that were retweeted
    # and then get the names of all of retweeters
    
    twitter = get_twitter_conn()    
    chase_timeline = pickle.load(open("chase_timeline.pkl", "rb"))
    
    # Get id and number of retweets for each tweet
    rtwt_data = []
    for twt in chase_timeline:
        if 'retweeted_status' not in twt:   # Original tweet by Chase
            if twt['retweet_count'] >= 5:  # Getting timeout errors if we look for every retweeted tweet
                rtwt_data.append([twt['id'], twt['retweet_count']])
    
    # Get the ids of the retweeters
    rtwt_user_data = []
    rtwt_user_set = set()
     
    for rtwt in rtwt_data:
        user_list = []
        user_results = twitter.cursor(twitter.get_retweeters_ids, id = rtwt[0])        
        for result in user_results:            
            user_list.append(result)
            rtwt_user_set.add(result)
            check_twitter_timeout(twitter)
             
        rtwt_user_data.append([rtwt[0], rtwt[1], user_list])
         
    pickle.dump(rtwt_user_data, open("retweet_user_data.pkl", "wb"))
        
    # Get the profiles of the retweeters
    rtwt_user_length = len(rtwt_user_set)
    retweet_users = []
    rtwt_user_list = []
    
    for n, rtwt_user in enumerate(rtwt_user_set):
        rtwt_user_list.append(rtwt_user)       
        if (len(rtwt_user_list) == 100) or (n == rtwt_user_length - 1):
            retweet_users.extend(twitter.lookup_user(user_id = rtwt_user_list))
            check_twitter_timeout(twitter)
            rtwt_user_list = [] 
        
    pickle.dump(retweet_users, open("retweet_users.pkl", "wb"))
        
def get_retweet_factors_impressions():    
    # Use the Twitter search API to retrieve recent tweets that contain high retweet scoring reference data 
    # (User Mentions and Hashtags) as well as Chase mentions. To be run after Twitter_Analysis.analyze_retweet_factors.
    
    twitter = get_twitter_conn()    
    
    score_table = read_pickle('k_score_table.pkl')
    sorted_score_table = score_table[score_table.Type.isin(['User Mention', 'Hashtag'])].sort(columns = ['Score'], ascending = False)
            
    # Retrieve names of top scoring reference data as well as anything that has 'chase' or 'sapphire' in the name
    top_score_table = sorted_score_table[sorted_score_table.Score > 1000.0]    
    chase_tags = np.array(['chase' in name_val for name_val in sorted_score_table.Name.values.ravel()]) 
    sapphire_tags = np.array(['sapphire' in name_val for name_val in sorted_score_table.Name.values.ravel()])
    chase_score_table = sorted_score_table[chase_tags | sapphire_tags]
    
    table_array = [top_score_table, chase_score_table]     
    retweet_tag_tweets = {}   
    
    # Loop through each tag and retrieve the most recent 1000 results
    for data_table in table_array:     
        for _, score_row in data_table.iterrows():          
            tag_results = []
            results = twitter.cursor(twitter.search, q=score_row[0], count=100)
            
            n = 1
            for result in results:
                tag_results.append(result)
                check_twitter_timeout(twitter)
                n += 1
                if n > 1000:
                    break
                                
            retweet_tag_tweets[score_row[0]] = tag_results            
    
    pickle.dump(retweet_tag_tweets, open("retweet_tag_tweets.pkl", "wb"))
            
def update_retweet_factor_data(data, data_type, factor_data_types, rtwt_factor):
    # Updates the dicts we are using to store reference association data 
    # and the reference entities for each tweet
    
    # Input
    # data: reference data to store
    # data_type: Name of reference data (Hashtag, URL, etc)
    # factor_data_types: dict mapping data to the data type
    # rtwt_factor: dict of reference data for a single tweet
    
    if data not in factor_data_types:
        factor_data_types[data] = data_type
    if data not in rtwt_factor:
        rtwt_factor[data] = 0
    rtwt_factor[data] += 1

def clean_tweet_text(tweet, use_utf):
    # Converts a UTF-8 tweet to ascii for easier reading, 
    # and then removes all reference data (Hashtags, etc).
    
    # Input
    # tweet: tweet object
    # use_utf: if we can't convert to ascii, whether to keep it
    
    # Output
    # twt_text: original tweet text converted to ascii, with reference data removed
        
    twt_text = tweet['text'].lower()
    
    try:
        twt_text = twt_text.decode("utf-8")
        twt_text = twt_text.encode("ascii", "ignore")
    except:
        if use_utf:
            twt_text = tweet['text'].lower()
        else:
            twt_text = ''
        
    twt_words = twt_text.split()
    for twt_word in twt_words:
        if twt_word.startswith('#') or twt_word.startswith('@') or twt_word.startswith('http'):
            twt_text = twt_text.replace(twt_word, '')
                    
    return twt_text             
            
def get_retweet_factors():
    # Collect statistics on each original tweet by Chase - word counts, hashtag references, etc. 
    # and writes it to a datatable.
    
    chase_timeline = pickle.load(open("chase_timeline.pkl", "rb"))
                     
    twt_char_types = ['NumRtwts', 'NumSymbols', 'NumHashtags', 'NumMedia', 'NumUrls', 'NumUserMents', 'IsReply', 'NumWords', 'NumChars']
    factor_data_types = { char_type : 'Twt Characteristic' for char_type in twt_char_types }
    
    rtwt_factor_dict = {}   
    for twt in chase_timeline:
        if 'retweeted_status' not in twt:   # Tweets written by Chase             
            twt_entities = twt['entities']
            text_split = clean_tweet_text(twt, True).split()
            
            rtwt_factor = {}
            rtwt_factor['NumRtwts'] = twt['retweet_count'] 
            rtwt_factor['NumSymbols'] = len(twt_entities['symbols'])
            rtwt_factor['NumHashtags'] = len(twt_entities['hashtags'])            
            rtwt_factor['NumMedia'] = len(twt_entities['media']) if 'media' in twt_entities else 0
            rtwt_factor['NumUrls'] = len(twt_entities['urls'])
            rtwt_factor['NumUserMents'] = len(twt_entities['user_mentions'])
            rtwt_factor['IsReply'] = 0 if twt['in_reply_to_status_id'] is None else 1            
            rtwt_factor['NumWords'] = len(text_split)
            rtwt_factor['NumChars'] = sum(len(s) for s in text_split)
            
            for twt_sym in twt_entities['symbols']:
                update_retweet_factor_data('$' + twt_sym['text'].lower(), 'Stock Symbol', factor_data_types, rtwt_factor)
                
            for twt_hash in twt_entities['hashtags']:
                update_retweet_factor_data('#' + twt_hash['text'].lower(), 'Hashtag', factor_data_types, rtwt_factor)
                                
            for user_mention in twt_entities['user_mentions']:
                update_retweet_factor_data('@' + user_mention['screen_name'].lower(), 'User Mention', factor_data_types, rtwt_factor)
                
            for twt_url in twt_entities['urls']:
                update_retweet_factor_data(twt_url['display_url'].lower().split('/')[0], 'URL', factor_data_types, rtwt_factor)
                
            if 'media' in twt_entities:
                for twt_media in twt_entities['media']:
                    update_retweet_factor_data(twt_media['type'], 'Media', factor_data_types, rtwt_factor)            
                            
            rtwt_factor_dict[twt['id']] = rtwt_factor
        
    rtwt_table = pd.DataFrame.from_dict(rtwt_factor_dict, orient = 'index') 
    rtwt_table.fillna(0, inplace = True)
    rtwt_table.to_pickle('retweet_table.pkl')
    pickle.dump(factor_data_types, open("factor_data_types.pkl", "wb"))
    
def write_data_to_csv():
    # Write user profile data, retweet connection data, and the retweet factor data to csv/txt files
    # that can be used by Spark.
        
    # Gather user data as (User Id, Screen Name, Number of Followers, Time Zone).
    # Using Time Zone instead of Location because it is slightly more accurate (Location is a self-reported string).
    user_list = []
    retweet_users = pickle.load(open("retweet_users.pkl", "rb"))
    chase_timeline = pickle.load(open("chase_timeline.pkl", "rb"))
    chase_user_data = chase_timeline[0]['user']
    chase_user_id = chase_user_data['id']
    
    user_ids = set()
    user_ids.add(chase_user_id)
    user_list.append([chase_user_id, chase_user_data['screen_name'], chase_user_data['followers_count'], chase_user_data['time_zone']])
    
    for retweet_user in retweet_users:
        user_list.append([retweet_user['id'], retweet_user['screen_name'], retweet_user['followers_count'], retweet_user['time_zone']])
        user_ids.add(retweet_user['id'])
    
    # Gather retweet connection data as (Retweeting User Id, Original User Id, Retweeted Tweet Id).
    connection_list = []
    rtwt_user_data = pickle.load(open("retweet_user_data.pkl", "rb"))
    for retweet_data in rtwt_user_data:
        tweet_id = retweet_data[0]
        for retweeting_user in retweet_data[2]:
            connection_list.append([retweeting_user, chase_user_id, tweet_id])
    
    for twt in chase_timeline:
        if 'retweeted_status' in twt:   # Retweeted by Chase
            orig_tweet_user_data = twt['retweeted_status']['user']
            connection_list.append([chase_user_id, orig_tweet_user_data['id'], twt['id']])
            if orig_tweet_user_data['id'] not in user_ids:
                user_list.append([orig_tweet_user_data['id'], orig_tweet_user_data['screen_name'], orig_tweet_user_data['followers_count'], orig_tweet_user_data['time_zone']])
            
    user_table = pd.DataFrame.from_records(user_list, index = 'UserId', columns = ['UserId', 'UserName', 'NumFollowers', 'Location'])        
    user_table.loc[user_table.Location.isnull(), 'Location'] = 'Unknown'
    user_table.to_csv('UserData.csv', header = False)
    
    connection_table = pd.DataFrame.from_records(connection_list, index = 'RetweetUserId', columns = ['RetweetUserId', 'OrigUserId', 'TweetId'])           
    connection_table.to_csv('ConnectionData.csv', header = False)
    
def pull_twitter_data():
    
    get_access_token()  # Twitter access key
    get_twt_history()   # Get last 3200 tweets from the Chase account
    get_retweet_factors()   # Collect reference data on tweets by Chase that were retweeted
    get_rwtwt_users_and_data()  # Collect user data on tweets by Chase that were retweeted
    write_data_to_csv() # Write the collected data to a csv file for use by Spark.