from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as mt
import sklearn
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
import pandas as pd
import numpy as np
pd.set_option('display.expand_frame_repr', False) #Make it so we display full dataframes
pd.set_option('display.max_columns', 25)


def main():

	print('Reading and processing training texts')
	df = pd.read_csv('all_subreddits.csv')
	df = df[df.subreddit.isin(["askreddit", "askscience"])]
	#test_df = df.sample(frac=1).iloc[10000:20000].reset_index()
	df = df.sample(frac=1).iloc[0:10000].reset_index()
	df = df[pd.notnull(df['body'])]

	df['random'] = np.random.rand(df.shape[0],1)
	df_train = df[df['random'] <=0.8]
	df_test = df[df['random'] > 0.8]
	df_train=df_train.reset_index()
	df_test=df_test.reset_index()


	print('Training data has {} rows, {} columns'.format(df_train.shape[0], df_train.shape[1]))
	print('Training data:')
	print(df_train)

	print('Body column:')
	print(df_train['body'])

	print('First row')
	print(df_train.iloc[0])

	nltk.download('punkt')

	df_train['target'] = (df_train['subreddit']=='askreddit').astype(int)
	df_test['target'] = (df_test['subreddit']=='askreddit').astype(int)

	#repeat
	print('Target column:')
	print(df_train[['subreddit','target']])

	df_train['processed_text'] = df_train['body'].apply(process_single_comment)
	df_test['processed_text'] = df_test['body'].apply(process_single_comment)
	print('Processed text:')
	print(df_train[['body','processed_text']])

	print("Converting training texts to vectors")
	count_vect = CountVectorizer()
	train_X = count_vect.fit_transform(df_train['processed_text'])
	test_X = count_vect.transform(df_test['processed_text'])

	clf=LogisticRegression()
	clf.fit(train_X,df_train["target"])

	test_predictions = clf.predict(test_X)
	# train_predictions = clf.predict(train_X)
	true_labels = df_test['target']
	most_common_label = df_train['target'].mode()[0]
	most_common_predictions = [most_common_label] * len(true_labels)

	print('\nClassifier performance:')
	print ('\tMost-common class test set accuracy score: {}'.format(mt.accuracy_score(true_labels, most_common_predictions)))
	print ('\tTest set accuracy score: {}'.format(mt.accuracy_score(true_labels, test_predictions)))

	print ('\tTest set classification report:\n{}'.format(mt.classification_report(true_labels, test_predictions)))

	import sys
	#sys.exit()
	
	#Look at the coefficients of the model
	print('\n\n**********************************************************\n\n')
	print('Top coefficients in model:')
	top_coef_indices = np.argsort(np.abs(clf.coef_))[0,-30:] #Find the indices of the top 30 coefficients by absolute value
	top_coefs = list(clf.coef_[0,top_coef_indices])
	top_word_ids = list(top_coef_indices)

	word_list = count_vect.get_feature_names()
	top_words = [word_list[id] for id in top_word_ids]

	for i in range(len(top_coefs)-1,-1,-1):
		print ('\t{}: {}'.format(top_words[i], top_coefs[i]))
  
	
	#Look at the coefficients of the model
	print("\n\n**********************************************************\n\n")
	print('Top coefficients in model:')
	top_coef_indices = np.argsort(np.abs(clf.coef_))[0,-30:] #Find the indices of the top 30 coefficients by absolute value
	top_coefs = list(clf.coef_[0,top_coef_indices])
	top_word_ids = list(top_coef_indices)


	word_list = count_vect.get_feature_names()
	top_words = [word_list[id] for id in top_word_ids]

	for i in range(len(top_coefs)-1,-1,-1):
		print('\t{}: {}'.format(top_words[i], top_coefs[i]))



			

	# prediction_differences = test_predictions - true_labels
	# correct_prediction_indices = np.where(prediction_differences == 0)[0]
	# false_positive_indices = np.where(prediction_differences == -1)[0]
	# false_negative_indices = np.where(prediction_differences == 1)[0]
	# print('\n\n**********************************************************\n\n')
	# print('\nLooking at prediction for a few sample comments:')
	# for index in np.concatenate((correct_prediction_indices[0:2],false_positive_indices[0:2],false_negative_indices[0:2])):
	# 	original_comment = test_df['body'][index]
	# 	comment = train_df['processed_text'][index]
	# 	true_label = train_df['target'][index]
	# 	predicted_label = test_predictions[index]
	# 	words = comment.split()
	# 	coefficients = [clf.coef_[0,count_vect.vocabulary_[word]] if word in count_vect.vocabulary_ else 0.0 for word in words]

	# 	coefficient_text = ' '.join(['{} ({:.2f})'.format(word, coefficient) for word, coefficient in zip(words, coefficients)])


	# 	print('\nUnprocessed comment: {}'.format(original_comment))
	# 	print('Processed comment: {}'.format(comment))
	# 	print('True target value: {}'.format(true_label))
	# 	print('Predicted target value: {}'.format(predicted_label))
	# 	print('Processed comment with coefficients: {}'.format(coefficient_text))

	# 	print('\n------------------------------\n')


	#Look at some of the comments and try to figure out why the modl
	prediction_differences = test_predictions - true_labels
	correct_prediction_indices = np.where(prediction_differences == 0)[0]
	false_positive_indices = np.where(prediction_differences == -1)[0]
	false_negative_indices = np.where(prediction_differences == 1)[0]
	print('\n\n**********************************************************\n\n')

	incorrect_prediction_indices = np.where(prediction_differences != 0)[0]
	display_indices = incorrect_prediction_indices[0:30]
	

	print('\nLooking at prediction for {} sample comments:'.format(display_indices.shape[0])) 

	print('\nLooking at prediction for a few sample comments:')
	for index in np.concatenate((correct_prediction_indices[0:0],false_positive_indices[0:10],false_negative_indices[0:10])):
		original_comment = df_test['body'][index]
		comment = df_test['processed_text'][index]
		true_label = df_test['target'][index]
		predicted_label = test_predictions[index]
		words = comment.split()
		coefficients = [clf.coef_[0,count_vect.vocabulary_[word]] if word in count_vect.vocabulary_ else 0.0 for word in words]

		coefficient_text = ' '.join(['{} ({:.2f})'.format(word, coefficient) for word, coefficient in zip(words, coefficients)])


		print('\nUnprocessed comment: {}'.format(original_comment))
		print('Processed comment: {}'.format(comment))
		print('True target value: {}'.format(true_label))
		print('Predicted target value: {}'.format(predicted_label))
		print('Processed comment with coefficients: {}'.format(coefficient_text))

		print('\n------------------------------\n')



stemmer  = PorterStemmer()
def process_single_comment(comment):
	#comment = comment.decode('utf8')
	tokens = word_tokenize(comment)
	tokens = [token.lower() for token in tokens]
	tokens = [stemmer.stem(token) for token in tokens]
	processed_comment = ' '.join(tokens)
	return processed_comment

if __name__ == '__main__':
	main()