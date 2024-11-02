
import numpy as np
import pandas as pd
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
import streamlit as st

# ตรวจสอบว่ามีการดาวน์โหลดข้อมูล nltk ที่จำเป็น
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# โหลดข้อมูล
data = pd.read_csv("train.csv")  # ตรวจสอบเส้นทางไฟล์ให้ถูกต้อง
data.dropna(inplace=True)

# แบ่งข้อมูลออกเป็นชุดฝึกและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(data['selected_text'], data['sentiment'], test_size=0.2, random_state=42)

# สร้างและฝึกโมเดล
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# สร้าง UI ใน Streamlit
st.title("Tweet Sentiment Analysis")
st.write("ใส่ข้อความทวีตที่ต้องการวิเคราะห์ด้านล่าง:")

# รับข้อความจากผู้ใช้
user_input = st.text_area("ข้อความทวีต:", "")

if st.button("วิเคราะห์"):
    if user_input:
        # ทำการทำนาย
        prediction = model.predict([user_input])
        sentiment = prediction[0]

        st.write(f"ผลการวิเคราะห์: {sentiment}")

        # แสดง confusion matrix
        predictions_test = model.predict(X_test)  # ทำนายผลสำหรับชุดทดสอบ
        cm = confusion_matrix(y_test, predictions_test)
        
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        st.pyplot(plt)
    else:
        st.warning("กรุณาใส่ข้อความทวีตที่ต้องการวิเคราะห์")
