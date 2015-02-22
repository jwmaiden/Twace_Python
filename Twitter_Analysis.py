import pickle
import xlwt
import matplotlib.pylab as py
import numpy as np
from sklearn import linear_model, metrics, ensemble, grid_search, neighbors, tree
import sklearn.cross_validation as cv
import sklearn.feature_selection as fs
from sklearn.decomposition import PCA
import pandas as pd
import nltk
from pandas.io.pickle import read_pickle
from Twitter_Data import clean_tweet_text

def print_histogram(y, num_bins):   
    # Prints a histogram of input array with equally spaced bins
    
    # Input
    # y: array
    # num_bins: number of bins in histogram
    
    _, _, patches = py.hist(y, num_bins, histtype='stepfilled')
    py.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
    py.show()
    
def get_tweet_distribution(show_histogram = False):    
    # Prints distribution for tweet collection. Excluding posts that are not original by Chase.
    # Also displays histogram of number of retweets and favorites for each tweet.
    # Writes output to spreadsheet for analysis. To be run after Twitter_Data.get_twt_history.
    
    chase_timeline = pickle.load(open("chase_timeline.pkl", "rb"))
    
    hashtag_map = {}
    retweet_dist = []
    favorite_dist = []
        
    for twt in chase_timeline:         
        if 'retweeted_status' not in twt:   # Want to look at original tweets, not retweets 
                   
            twt_entities = twt['entities']       
            hashtag_set = set([twt_hashs['text'].lower() for twt_hashs in twt_entities['hashtags']])
            favs_count = twt['favorite_count']
            rtwts_count = twt['retweet_count']
            
            # Store hashtag counts
            if len(hashtag_set) > 0:
                for hashtag in hashtag_set:
                    if hashtag not in hashtag_map:
                        hashtag_map[hashtag] = {}
                        hashtag_map[hashtag]['Count'] = 0
                        hashtag_map[hashtag]['Retweets'] = 0
                        hashtag_map[hashtag]['Favorites'] = 0
                    hashtag_map[hashtag]['Count'] += 1
                    hashtag_map[hashtag]['Retweets'] += rtwts_count
                    hashtag_map[hashtag]['Favorites'] += favs_count        
            
            # Favorites
            if favs_count > 0:
                favorite_dist.append(favs_count)
            
            # Retweets
            if rtwts_count > 0:
                retweet_dist.append(rtwts_count)
    
    twt_dist_book = xlwt.Workbook()
    hash_tab = twt_dist_book.add_sheet('Hashtags')    
    
    # Print occurances, retweets, and favorites for each hashtag
    hash_tab.write(0, 0, "Hashtag")
    hash_tab.write(0, 1, "Occurances")
    hash_tab.write(0, 2, "Retweeted")
    hash_tab.write(0, 3, "Favorited")
    hash_tab.write(0, 4, "Retweets Per Occurance")
    hash_tab.write(0, 5, "Favorites Per Occurance")
       
    for n, (hashtag, hash_values) in enumerate(hashtag_map.items(), 1):
        hash_tab.write(n, 0, hashtag)
        hash_tab.write(n, 1, hash_values['Count'])
        hash_tab.write(n, 2, hash_values['Retweets'])
        hash_tab.write(n, 3, hash_values['Favorites'])
        hash_tab.write(n, 4, float(hash_values['Retweets']) / float(hash_values['Count']))
        hash_tab.write(n, 5, float(hash_values['Favorites']) / float(hash_values['Count']))
    
    # Display a histogram of the retweet and favorites distribution
    if show_histogram:
        print_histogram(retweet_dist, max(retweet_dist) - min(retweet_dist) + 1)
        print_histogram(favorite_dist, max(favorite_dist) - min(favorite_dist) + 1)
    
    # Write the histogram to a spreadsheet
    rwtwt_hist, rwtwt_bins = np.histogram(retweet_dist, max(retweet_dist) - min(retweet_dist) + 1)
    fav_hist, fav_bins = np.histogram(favorite_dist, max(favorite_dist) - min(favorite_dist) + 1)
    
    hist_tab = twt_dist_book.add_sheet("Retwts + Favs")
    
    hist_tab.write(0, 0, "Num Retweets")
    hist_tab.write(0, 1, "Count")
    
    hist_tab.write(0, 3, "Num Favorites")
    hist_tab.write(0, 4, "Count")    
    
    for x in range(np.size(rwtwt_hist)):
        hist_tab.write(x + 1, 0, int(rwtwt_bins[x+1]))
        hist_tab.write(x + 1, 1, int(rwtwt_hist[x]))
        
    for x in range(np.size(fav_hist)):
        hist_tab.write(x + 1, 3, int(fav_bins[x+1]))
        hist_tab.write(x + 1, 4, int(fav_hist[x]))       
    
    twt_dist_book.save("Tweet Distribution.xls")
    
def get_model_values(model, x_train, y_train, x_test, y_test):
    # Fit a model and return the score and mse
    
    # Input
    # model: scikit-learn model
    # x_train: independent variables training set
    # y_train: dependent variable training set
    # x_test: independent variables test set
    # y_test: dependent variable test set
    
    # Output
    # train_score: training score
    # test_score: test score
    # train_mse: training mse
    # test_mse: test mse        
    
    model.fit(x_train, y_train)
    
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    train_mse = metrics.mean_squared_error(model.predict(x_train), y_train)
    test_mse = metrics.mean_squared_error(model.predict(x_test), y_test)
    
    return train_score, test_score, train_mse, test_mse    

def get_grid_search_values(model, grid_params, x_train, y_train, x_test, y_test, scoring_criteria = 'mean_squared_error'):  
    # Run a grid search on a model, and return the train / test score and MSE on the best result
    
    # Input
    # model: scikit-learn model
    # grid_params: dict of parameter space
    # x_train: independent variables training set
    # y_train: dependent variable training set
    # x_test: independent variables test set
    # y_test: dependent variable test set
    # scoring_criteria: model scoring criteria
    
    # Output
    # best_model: model that produced the best results
    # para_search.best_params_: best grid parameters
    # train_score: training score
    # test_score: test score
    # train_mse: training mse
    # test_mse: test mse
    
    para_search = grid_search.GridSearchCV(model, grid_params, scoring = scoring_criteria, cv = 5).fit(x_train, y_train)
    best_model = para_search.best_estimator_
    train_score = best_model.score(x_train, y_train)
    test_score = best_model.score(x_test, y_test)
    train_mse = metrics.mean_squared_error(best_model.predict(x_train), y_train)
    test_mse = metrics.mean_squared_error(best_model.predict(x_test), y_test)
    
    return best_model, para_search.best_params_, train_score, test_score, train_mse, test_mse

def convert_retweet_factors_to_csv(rtwt_table, score_table):    
    # Gather the retweet factor data and organize into a csv format that can be converted into the libsvm 
    # format used by Spark. To convert, call "python csv2libsvm.py RetweetFactors.csv rtwtsvm.txt 0 True"
    
    # Input
    # rtwt_table: Pandas dataframe of all tweet reference data vs. number of retweets
    # score_table: Pandas dataframe of k-scores of dependent data vs. number of retweets
        
    rtwt_cols = rtwt_table.columns.values.tolist()    
    sorted_score_table = score_table.sort(columns = ['Score'], ascending = False)
    
    sorted_rtwt_cols = ['NumRtwts']
    sorted_rtwt_cols += [rtwt_factor for rtwt_factor in sorted_score_table.Name.values.ravel()]
    if len(rtwt_cols) != len(sorted_rtwt_cols):
        raise Exception('Unable to reorder the retweet columns.')
    
    sorted_rtwt_table = rtwt_table[sorted_rtwt_cols]
    sorted_rtwt_table.to_csv('RetweetFactors.csv', index = False)    

def analyze_retweet_factors():
    # Look at the reference data pulled in for each tweet by Chase that was retweeted. 
    # Score the best performing factors and model "Number of Retweets" vs. reference data.
    # Must be run after Twitter_Data.get_retweet_factors.
    
    rtwt_table = read_pickle('retweet_table.pkl')
    factor_data_types = pickle.load(open("factor_data_types.pkl", "rb"))
    
    depd_vals_table = rtwt_table.drop('NumRtwts', axis = 1)
    x = np.array(depd_vals_table)
    y = rtwt_table.NumRtwts
    
    pca = PCA().set_params(n_components = 0.9)
    x2 = pca.fit_transform((x - x.mean()) / x.std()) 
    
    # Use a chi-squared test to determine the relevance of each dependent variable
    # against the number of retweets.
    best_k = fs.SelectKBest(fs.chi2, k = 10).fit(x, y.values.ravel())
    
    # Write scores to spreadsheet so that we can determine which factors have the most importance
    rtwt_test_book = xlwt.Workbook()
    
    k_tab = rtwt_test_book.add_sheet('Data') 
    k_tab.write(0, 0, 'Name')
    k_tab.write(0, 1, 'Type')
    k_tab.write(0, 2, 'Score')
    
    k_score_array = []
    data_table_cols = depd_vals_table.columns.values.tolist()
    for n, k_score in enumerate(best_k.scores_, 0):
        k_tab.write(n+1, 0, data_table_cols[n])
        k_tab.write(n+1, 1, factor_data_types[data_table_cols[n]])
        k_tab.write(n+1, 2, k_score)
        k_score_array.append([data_table_cols[n], factor_data_types[data_table_cols[n]], k_score])
    
    score_table = pd.DataFrame.from_records(k_score_array, columns = ['Name', 'Type', 'Score'])
    score_table.to_pickle('k_score_table.pkl')
    
    # Use the k-scores to reorder the retweet factor table, to be used later by Spark.
    convert_retweet_factors_to_csv(rtwt_table, score_table)
    
    x3 = best_k.transform(x)
    x4 = best_k.set_params(k = 20).fit_transform(x, y.values.ravel())
    
    # Model the data against a collection of regression models and write the results to a spreadsheet
    model_data = [#("Original Data", x, y),    # Original data takes ages to run - 2k features
                  ("PCA Transform Data", x2, y),
                  ("K (10) Best Data", x3, y),
                  ("K (20) Best Data", x4, y)]
    
    linear_models = [("OLS", linear_model.LinearRegression()),
              ("Ridge", linear_model.RidgeCV(normalize = True, fit_intercept = False, scoring = 'mean_squared_error', cv = 5)),
              ("Lasso", linear_model.LassoCV(normalize = True, fit_intercept = False, cv = 5))]    
    
    tree_models = [("KNN", neighbors.KNeighborsClassifier(), {'n_neighbors' : np.arange(3, 9), 'weights' : ['uniform', 'distance'], 'p' : [1, 2]}),
              ("Decision Tree", tree.DecisionTreeClassifier(), {'criterion' : ['gini', 'entropy'], 'max_features' : [None, 'auto', 'log2'], 'max_depth' : [None, 3, 4, 5]}),
              ("Random Forest", ensemble.RandomForestClassifier(), {'criterion': ['gini', 'entropy'], 'max_features' : [None, 'auto', 'log2'], 'n_estimators': np.arange(10, 50, 10), 'max_depth' : [None, 3, 4, 5]})
              ]
    
    test_tab = rtwt_test_book.add_sheet('Model Results')
    test_tab.write(0, 0, 'Dataset Name')
    test_tab.write(0, 1, 'Model Name')
    test_tab.write(0, 2, 'Training Score')
    test_tab.write(0, 3, 'Test Score')
    test_tab.write(0, 4, 'Training MSE')
    test_tab.write(0, 5, 'Test MSE')     
    
    row = 1
    for m_data in model_data:
                
        x_train, x_test, y_train, y_test = cv.train_test_split(m_data[1], m_data[2], train_size = 0.8, random_state = 0)
                             
        for model_name, model in linear_models:
            train_score, test_score, train_mse, test_mse = get_model_values(model, x_train, y_train, x_test, y_test)
            test_tab.write(row, 0, m_data[0])
            test_tab.write(row, 1, model_name)
            test_tab.write(row, 2, train_score)
            test_tab.write(row, 3, test_score)
            test_tab.write(row, 4, train_mse)
            test_tab.write(row, 5, test_mse)
            row += 1        
        
        for model in tree_models:
            _, _, train_score, test_score, train_mse, test_mse = get_grid_search_values(model[1], model[2], x_train, y_train, x_test, y_test, 'accuracy')
            test_tab.write(row, 0, m_data[0])
            test_tab.write(row, 1, model[0])
            test_tab.write(row, 2, train_score)
            test_tab.write(row, 3, test_score)
            test_tab.write(row, 4, train_mse)
            test_tab.write(row, 5, test_mse)
            row += 1
            
    rtwt_test_book.save('Retweet Analysis.xls')

def get_word_tokenize(text):
    # Tokenize a string of input text
    
    # Input
    # text: input text
    
    # Output
    # list of tokenized words 
       
    sentences = [s for s in nltk.sent_tokenize(text)]
    normalized_sentences = [s.lower() for s in sentences]
    return [w.lower() for sentence in normalized_sentences for w in nltk.word_tokenize(sentence)]  

def get_top_n_words(words, n, stopwords): 
    # Return the top n most frequent words from a tokenized list of words, using the input stopwords
    
    # Input
    # words: tokenized words
    # n: Top N words to return
    # stopwords: List of stopwords
    
    # Output
    # top_n_words: Top N most frequent words
      
    fdist = nltk.FreqDist(words)
    top_n_words = [w[0] for w in fdist.items() if w[0] not in stopwords][:n]    
    return top_n_words
            
def test_retweet_impression_distribution():
    # Go through the tweets that are associated with the most popular
    # retweet factors and look for frequent words. To be run aftr Twitter_Data.get_retweet_factors_impressions.
        
    retweet_tag_tweets = pickle.load(open("retweet_tag_tweets.pkl", "rb"))
    
    stopwords = nltk.corpus.stopwords.words('english')     

    for twt_tag, twt_txt_array in retweet_tag_tweets.iteritems(): 
        clean_twt_words = [] 
        for twt in twt_txt_array:              
            for clean_twt_word in get_word_tokenize(clean_tweet_text(twt, False)):
                clean_twt_words.extend(clean_twt_word)    
        top_n_summ_words = get_top_n_words(clean_twt_words, 10, stopwords)
      
        print 'Retweet Factor:\t{}'.format(twt_tag)
        print 'Top 10 words:\t{}'.format(top_n_summ_words)              
        print '\n'
        
def analyze_tweet_factors():
    
    get_tweet_distribution()    # Show histogram of retweets/favorites for Chase's original tweets
    analyze_retweet_factors()   # Analyze the factors that led to retweets of Chase's original tweets