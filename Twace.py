from Twitter_Data import pull_twitter_data
from Twitter_Analysis import analyze_tweet_factors

def run_twace():
    
    pull_twitter_data() # Pull in most recent tweets for Chase, pulling out relevant factors
    analyze_tweet_factors() # Analyze the data that we've pulled in

run_twace()