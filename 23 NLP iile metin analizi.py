import pandas as pd
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re
import numpy as np 
from sklearn.model_selection import train_test_split
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# NLTK stopwords'lerini indir
nltk.download('stopwords')

# Veriyi yükle
data = pd.read_csv("C:/Users/ASAF/Desktop/Datasets/NLPlabeledData.tsv", sep='\t', quoting=3)
df = data.copy()  

# İlk örneği kontrol et
ex_text = data.review[0]

# Metin işleme: HTML etiketlerini kaldır, harf ve boşluk dışındaki karakterleri kaldır, küçük harfe çevir ve kelimeleri ayır
ex_text = BeautifulSoup(ex_text, "lxml").get_text()  # HTML'den metin çıkar
ex_text = re.sub("[^a-zA-Z]", " ", ex_text)  # Harfler ve boşluklar dışındaki her şeyi kaldır
ex_text = ex_text.lower()  # Küçük harfe çevir
ex_text = ex_text.split()  # Kelimeleri ayır

print(f"Örnek işlem: {ex_text[:10]}")  # İlk 10 kelimeyi yazdır

# Stopwords setini oluştur
swords = set(stopwords.words("english"))

# Stopwords'leri kaldır
ex_text = [w for w in ex_text if w not in swords]

print(f"Stopwords'ten arındırılmış: {ex_text[:10]}...")

def islem(review):
    review = BeautifulSoup(review, "lxml").get_text()  # HTML'den metin çıkar
    review = re.sub("[^a-zA-Z]", " ", review)  # Harfler ve boşluklar dışındaki her şeyi kaldır
    review = review.lower()  # Küçük harfe çevir
    review = review.split()  # Kelimeleri ayır
    swords = set(stopwords.words("english"))  # Stopwords setini oluştur
    review = [w for w in review if w not in swords]  # Stopwords'leri kaldır
    return " ".join(review)  # Kelimeleri tekrar birleştir

# Tüm veri için metin işleme fonksiyonunu uygula
train_x_tum = []

for i in range(len(data["review"])):
    if (i + 1) % 1000 == 0:
        print(f"İşlem adımı: {i + 1}")
    train_x_tum.append(islem(data["review"][i]))

# İşlenmiş örnekleri yazdır
print(f"İşlenmiş örnekler: {train_x_tum[:5]}")  # İlk 5 örneği yazdır

# Bağımsız ve bağımlı değişkenleri ayır
x = train_x_tum
y = np.array(data["sentiment"])

# Veriyi eğitim ve test setlerine böl
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=20)

# Metin verisini sayısal verilere dönüştür
cv = CountVectorizer(max_features=5000)

x_train1 = cv.fit_transform(x_train)  # Eğitim verisini dönüştür

y1_train = y_train

# Rastgele Orman sınıflandırıcısı oluştur ve eğit
rf = RandomForestClassifier(n_estimators=100)
model = rf.fit(x_train1, y1_train)

# Test verisini dönüştür
x1_test = cv.transform(x_test)

# Tahmin yap
tahmin = rf.predict(x1_test)

# Modelin doğruluğunu hesapla
score = model.score(x1_test, y_test)
print(score)
