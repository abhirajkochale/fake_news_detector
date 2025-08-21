📰 Fake News Detection System
This is a small project I built to classify news articles as Real or Fake using Machine Learning + NLP.
It comes with a simple Streamlit app, so you can paste in any news article and instantly check if it’s fake or not.

✨ What it does
Cleans and processes the dataset (fake + real news)
Converts text into numerical features using TF-IDF
Trains a Naive Bayes classifier
Lets you test any custom news in a web interface
Shows the model’s accuracy after training

📂 What’s inside
fake_news.py → the main app (run this with Streamlit)
requirements.txt → list of dependencies
FakeNewsDataset.rar → dataset (compressed file with Fake.csv and True.csv)
README.md → this file

🚀 How to run it
Clone the repo
git clone https://github.com/abhirajkochale/Fake-News-Detection.git
cd Fake-News-Detection
Extract the dataset
Inside this repo there’s a file: FakeNewsDataset.rar
Extract it with WinRAR or 7zip
You’ll get two files:
Fake.csv
True.csv
Keep them in the same folder as fake_news.py
Install dependencies
pip install -r requirements.txt
Run the Streamlit app
streamlit run fake_news.py

🖊️ Example usage
Try entering:
Government launches free healthcare program for all citizens.
Result → ✅ REAL News
Or:
Scientists confirm aliens secretly living among us.
Result → ❌ FAKE News

🛠️ Tech used
Python
Streamlit
Scikit-learn
Pandas
TF-IDF Vectorizer
Naive Bayes Classifier
💡 This project is just a starting point. You can improve it by:

Adding more data

Trying other ML models (Logistic Regression, Random Forest, etc.)
