import tweepy
from tweepy import Cursor
from tweepy import OAuthHandler
import sentiment_module as s

#consumer key, consumer secret, access token, access secret.
ckey="XXXXXXXXXXXXXX"
csecret="XXXXXXXXXXXXXXXX"
atoken="XXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
asecret="XXXXXXXXXXXXXXXXXXXXXXXX"

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)
api = tweepy.API(auth)

cursor = tweepy.Cursor(api.search, q='#Trump', result_type="recent", lang='en')

tweets = [tweet.text for tweet in cursor.items(1000)]

sentiments = []

for t in tweets:
	sentiment_value, confidence = s.sentiment(t)
	if confidence > 0.80:
		sentiments.append(sentiment_value)
		print t, sentiment_value, confidence

x = 0.00
for sent in sentiments:
	if sent == "pos":
		x += 1.00

print len(sentiments)
try:
	print "%"+str(x / len(sentiments) * 100)
except ZeroDivisionError:
	pass

