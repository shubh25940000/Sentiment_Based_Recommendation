from flask import Flask, request, render_template
from model import ProductRecommendationEngine


app = Flask(__name__)

sentiment_model = ProductRecommendationEngine()


@app.route('/')
def home():
    return render_template('index_2.html')


@app.route('/predict', methods=['POST'])
def prediction():
    # get user from the html form
    user = request.form['userName']
    # convert text to lowercase
    user = user.lower()
    items = sentiment_model.product_recommendations(user)
    print("inside")

    if(not(items is None)):
        print(f"retrieving items....{len(items)}")
        list(items.values.tolist())
        return render_template("index_2.html", column_names=items.columns.values, row_data=list(items.values.tolist()), zip=zip)
    else:
        return render_template("index_2.html", message="User Name doesn't exists, No product recommendations at this point of time!")


@app.route('/predictSentiment', methods=['POST'])
def predict_sentiment():
    # get the review text from the html form
    review_text = request.form["reviewText"]
    review_title = request.form["reviewTitle"]
    if (review_text != "") & (review_title != ""):
        pred_sentiment = sentiment_model.predict_sentiment(review_text, review_title)[0]
        pred_sentiment_value = sentiment_model.predict_sentiment(review_text, review_title)[1]
        return render_template("index_2.html", sentiment=pred_sentiment, sentiment_prob = pred_sentiment_value)
    else:
        pred_sentiment = "Please enter valid review text and title"
        return render_template("index_2.html", message=pred_sentiment)

if __name__ == '__main__':
    app.run()