import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

@st.cache_data
def load_data():
    fake = pd.read_csv("Fake.csv")
    true = pd.read_csv("True.csv")
    min_len = min(len(fake), len(true))
    fake = fake.sample(min_len)
    true = true.sample(min_len)
    fake["label"] = 0
    true["label"] = 1
    df = pd.concat([fake, true], axis=0)
    df = df.sample(frac=1).reset_index(drop=True)
    return df

@st.cache_resource
def train_model():
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42
    )
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_tfidf))
    return model, vectorizer, acc

def predict_news(news, model, vectorizer):
    news_tfidf = vectorizer.transform([news])
    prediction = model.predict(news_tfidf)[0]
    return "✅ REAL News" if prediction == 1 else "❌ FAKE News"

def main():
    st.title("📰 Fake News Detection System")
    st.write("Enter a news article below to check if it’s **Real or Fake**.")
    model, vectorizer, acc = train_model()
    st.success(f"Model trained successfully! ✅ Accuracy: {acc*100:.2f}%")
    user_input = st.text_area("Enter News Article:")
    if st.button("Check"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some news text.")
        else:
            result = predict_news(user_input, model, vectorizer)
            st.subheader("Result:")
            st.write(result)

if __name__ == "__main__":
    main()
