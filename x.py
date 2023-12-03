from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Veri setini yükle
data = fetch_20newsgroups()

# Eğitim ve test verilerini ayır
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=11)

# TF-IDF vektörleştirici ve Naive Bayes sınıflandırıcı içeren bir boru hattı oluştur
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Modeli eğit
model.fit(X_train, y_train)

# Test verileri üzerinde modelin performansını değerlendir
predicted = model.predict(X_test)
print(classification_report(y_test, predicted))

# Modeli dosyaya kaydet
joblib.dump(model, 'text_classifier.model')