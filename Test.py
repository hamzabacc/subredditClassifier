import sklearn
import nltk 
from nltk import word_tokenize
from nltk.stem.porter import *
import pandas as pd

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', 25)
stemmer = PorterStemmer()

def main():
    print("Reading and processing training texts")

    all_subreddit_df = pd.read_csv("all_subreddits.csv")
    train_df = all_subreddit_df[all_subreddit_df.subreddit.isin(["askreddit", "askscience"])]
   # df["random_rank"] = df.groupby("subreddit", as_index=False)["random_number"].rank()
    #df["training_data"] = df["random_rank"] % 10 == 0
   # train_df = df[df.training_data == True]
    #test_df = df[df.training_data == False]
    #train_df=pd.read_csv('aww_politics_comments_late_2012_train.csv', nrows=5000)
    train_df=train_df[pd.notnull(train_df['body'])]
    print('Training data has {} rows, {} columns'.format(train_df.shape[0],train_df.shape[1]))
    print('Training data:')
    print(train_df)

    print('Body column:')
    print(train_df['body'])
   # print(train_df['body'])

    print("First Row")
    print(train_df.iloc[0])

    #nltk.download('punkt')

    train_df['target'] = (train_df['subreddit']=='aww').astype(int)
    print("Target column:")
    print(train_df[['subreddit','target']])

    train_df['processed_text'] = train_df['body'].apply(process_text)
    print("Procesed text:")
    print(train_df[['body','processed_text']])



def process_text(text):
    return text.lower()

stemer = PorterStemmer()
def process_single_comment(comment):
    comment = comment.decode('utf8')
    tokens = word_tokenize(comment)
    tokens = [token.lower(token) for token in tokens]
    processed_comment = " ".join(tokens)
    return processed_comment

if __name__ == '__main__':
    main()