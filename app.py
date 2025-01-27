import tkinter as tk
from PIL import Image, ImageTk
import re
import pandas as pd
import numpy as np
import random
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.model_selection import train_test_split



nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

negation_words = {"not", "no", "never"}
stop_words -= negation_words
lemmatizer = WordNetLemmatizer()

# Przykładowe recenzje
positive_reviews = [
    "This movie was amazing! The story was so touching.",
    "Excellent film, loved every moment!",
    "One of the best movies I've ever seen.",
    "The acting was phenomenal. Highly recommend!",
    "Great story, great cast, absolutely loved it."
]

negative_reviews = [
    "This was a complete waste of time.",
    "Terrible acting and a boring plot.",
    "I regret watching this. Not worth it.",
    "One of the worst movies I've seen in years.",
    "The script was awful and the pacing was terrible."
]

# Funkcja do wczytania danych z pliku CSV
def load_data(filename, limit=1000):
    print(f"Ładuje dane {filename} z limitem {limit}...")
    data = pd.read_csv(filename, nrows=limit)
    print(f"Załadowano {len(data)} wierszy z pliku.")
    original_reviews = data['review'].tolist()
    data['review'] = data['review'].apply(preprocess_text)
    data['sentiment'] = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    positive_count = data['sentiment'].sum()
    negative_count = len(data) - positive_count
    print(f"Pozytywne recenzje: {positive_count}, Negatywne recenzje: {negative_count}")
    return original_reviews, data['review'].tolist(), data['sentiment'].tolist()

# Funkcja przetwarzania tekstu
def preprocess_text(text):
    print(f"Przetwarzam tekst: {text[:30]}...")
    text = re.sub(r'<.*?>', '', text)  # Usuwa znaczniki HTML
    text = re.sub(r'http\S+|www\.\S+', '', text)  # Usuwa linki
    text = re.sub(r'[^a-zA-Z ]', '', text).lower()  # Usuwa znaki nieliterowe i konwertuje na małe litery
    text = re.sub(r'([.,!?])', r' \1 ', text)  # Wstawia spacje po znakach przestankowych
    text = re.sub(r'\s+', ' ', text).strip()
    words = word_tokenize(text)  # Tokenizacja za pomocą NLTK
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lematyzacja i usuwanie stopwords
    # Użycie bigramów
    bigrams = ['_'.join(gram) for gram in ngrams(words, 2)]
    words.extend(bigrams)
    processed_text = ' '.join(words)
    print(f"Tekst po przetworzeniu: {processed_text[:30]}...")
    return processed_text


# Funkcja do ograniczenia słownika do najczęściej występujących słów
def build_vocab(reviews, max_vocab_size=5000):
    print("Buduję słownik...")
    all_words = ' '.join(reviews).split()
    word_counts = pd.Series(all_words).value_counts()
    vocab = {word: i for i, word in enumerate(word_counts.head(max_vocab_size).index)}
    print(f"Słownik zbudowany z {len(vocab)} słów.")
    return vocab

# Funkcja do konwersji recenzji na wektor cech
def vectorize_reviews(reviews, vocab):
    print("Wektoryzacja recenzji...")
    vectors = np.zeros((len(reviews), len(vocab)))
    for i, review in enumerate(reviews):
        for word in review.split():
            if word in vocab:
                vectors[i, vocab[word]] += 1
        if i % 100 == 0:
            print(f"Zwektoryzowano {i + 1}/{len(reviews)} recenzji...")
    return vectors

# Funkcja generująca chmurę słów
def generate_wordcloud(reviews, title):
    print(f"Generowanie chmury słów dla {title}...")
    all_words = ' '.join(reviews)
    wordcloud = WordCloud(width=800, height=400, background_color='black').generate(all_words)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('on')
    plt.title(title, fontsize=16)
    plt.show()

# Implementacja MLP
class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights_input_hidden = np.random.uniform(-0.5, 0.5, (input_size, hidden_size))
        self.weights_hidden_output = np.random.uniform(-0.5, 0.5, (hidden_size, output_size))
        self.bias_hidden = np.random.uniform(-0.5, 0.5, hidden_size)
        self.bias_output = np.random.uniform(-0.5, 0.5, output_size)

    def sigmoid(self, x):
        x = np.clip(x, -700, 700)
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, inputs):
        self.hidden_layer = self.sigmoid(np.dot(inputs, self.weights_input_hidden) + self.bias_hidden)
        self.output_layer = self.sigmoid(np.dot(self.hidden_layer, self.weights_hidden_output) + self.bias_output)
        return self.output_layer

    def backward(self, inputs, targets, learning_rate):
        output_errors = targets - self.output_layer
        hidden_errors = np.dot(output_errors, self.weights_hidden_output.T) * self.sigmoid_derivative(self.hidden_layer)

        self.weights_hidden_output += learning_rate * np.dot(self.hidden_layer[:, np.newaxis], output_errors[np.newaxis, :])
        self.weights_input_hidden += learning_rate * np.dot(inputs[:, np.newaxis], hidden_errors[np.newaxis, :])
        self.bias_hidden += learning_rate * hidden_errors
        self.bias_output += learning_rate * output_errors

    def train(self, inputs, targets, epochs, learning_rate):
        print("Trenuje sieć...")
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}...")
            for i in range(len(inputs)):
                self.forward(inputs[i])
                self.backward(inputs[i], targets[i], learning_rate)
            print(f"Epoch {epoch + 1} zakończony.")

# Wczytanie danych
original_reviews, reviews, sentiments = load_data('imdb.csv', limit=15000)
vocab = build_vocab(reviews, max_vocab_size=20000)
input_vectors = vectorize_reviews(reviews, vocab)

train_vectors, test_vectors, train_sentiments, test_sentiments = train_test_split(
    input_vectors, sentiments, test_size=0.2, random_state=42
)

# Inicjalizacja i trenowanie modelu
mlp = MLP(train_vectors.shape[1], 10, 1)
mlp.train(train_vectors, np.array(train_sentiments).reshape(-1, 1), epochs=10, learning_rate=0.1)

# Obliczanie precyzji na zbiorze testowym
print("Obliczam precyzyjność sieci...")
predictions = []
for vector, target in zip(test_vectors, test_sentiments):
    output = mlp.forward(vector)[0]
    predictions.append(1 if output > 0.5 else 0)
correct = sum(p == t for p, t in zip(predictions, test_sentiments))
print(f"Liczba poprawnych klasyfikacji: {correct} z {len(test_sentiments)}")
accuracy = correct / len(test_sentiments)
print(f"Precyzyjność sieci na danych testowych: {accuracy:.2%}")


# Funkcja analizy recenzji
def analyze_sentiment():
    review = entry.get("1.0", tk.END).strip()  # Pobiera tekst z pola
    if review:
        print("Analizuje recenzje...")
        corrected_review = str(TextBlob(review).correct())
        print(f"Poprawiony tekst: {corrected_review}")
        preprocessed_review = preprocess_text(corrected_review)
        vector = vectorize_reviews([preprocessed_review], vocab)[0]
        prediction = mlp.forward(vector)[0]
        hidden_activations = mlp.hidden_layer
        np.set_printoptions(suppress=True, precision=10)
        print(f"Warstwa ukryta - wartości: {hidden_activations}")
        np.set_printoptions(suppress=True, precision=10)
        print(f"Wynik sieci neuronowej przed zastosowaniem progu decyzyjnego: {prediction}")
        sentiment = "Pozytywna" if prediction > 0.5 else "Negatywna"
        print(f"Wynik: {sentiment}")


        # Wyczyszczenie poprzedniego wyniku i wyświetlenie nowego
        result_label.config(text=f"Wynik: {sentiment}", fg="green" if sentiment == "Pozytywna" else "red")
    else:
        result_label.config(text="", fg="black")



# Funkcja analizy z animacją ładowania
def show_loading():
    result_label.config(text="")  # Wyczyść poprzedni wynik
    loading_label = tk.Label(
        text_frame, text="Analizuje...", font=("Arial", 12, "bold"), bg=text_frame.cget("bg"), fg="#007bff"
    )
    loading_label.pack(pady=(5, 0))
    root.update_idletasks()
    root.after(700, lambda: (loading_label.destroy(), analyze_sentiment()))




########### UI

# Tworzenie głównego okna
root = tk.Tk()
root.title("Analizator sentymentu recenzji filmowych")
root.configure(bg="#f0f0f5")

# Tworzenie ramki dla listy recenzji
list_frame = tk.Frame(root, bg="#f0f0f5")
list_frame.pack(side=tk.LEFT, padx=10, pady=10)

positive_icon_image = Image.open("pos.png").resize((20, 20), Image.Resampling.LANCZOS)
positive_icon = ImageTk.PhotoImage(positive_icon_image)

negative_icon_image = Image.open("neg.png").resize((20, 20), Image.Resampling.LANCZOS)
negative_icon = ImageTk.PhotoImage(negative_icon_image)

# Nagłówki dla list
positive_label = tk.Label(
    list_frame, text="Pozytywne recenzje", font=("Arial", 10, "bold"), bg="#f0f0f5",
    image=positive_icon, compound="left", padx=10
)
positive_label.pack(anchor="w")

def insert_review(review):
    entry.delete("1.0", tk.END)
    entry.insert(tk.END, review)

def load_random_positive():
    positive_reviews_loaded = [original_reviews[i] for i, sentiment in enumerate(sentiments) if sentiment == 1]
    if positive_reviews_loaded:
        review = random.choice(positive_reviews_loaded)
        insert_review(review)

def load_random_negative():
    negative_reviews_loaded = [original_reviews[i] for i, sentiment in enumerate(sentiments) if sentiment == 0]
    if negative_reviews_loaded:
        review = random.choice(negative_reviews_loaded)
        insert_review(review)

for review in positive_reviews:
    btn = tk.Button(
        list_frame, text=review, wraplength=200, anchor="w", justify="left",
        font=("Arial", 9), bg="#e6e6e6", relief=tk.GROOVE,
        command=lambda r=review: insert_review(r)
    )
    btn.pack(fill="x", pady=2)

negative_label = tk.Label(
    list_frame, text="Negatywne recenzje", font=("Arial", 10, "bold"), bg="#f0f0f5",
    image=negative_icon, compound="left", padx=10
)
negative_label.pack(anchor="w", pady=(10, 0))

for review in negative_reviews:
    btn = tk.Button(
        list_frame, text=review, wraplength=200, anchor="w", justify="left",
        font=("Arial", 9), bg="#e6e6e6", relief=tk.GROOVE,
        command=lambda r=review: insert_review(r)
    )
    btn.pack(fill="x", pady=2)

text_frame = tk.Frame(root, bg="#f0f0f5")
text_frame.pack(side=tk.RIGHT, padx=10, pady=10)

label = tk.Label(text_frame, text="Wprowadź recenzję:", font=("Arial", 12, "bold"), bg="#f0f0f5")
label.pack()

entry = tk.Text(text_frame, height=10, width=50, font=("Arial", 10), bg="#ffffff")
entry.pack()

analyze_button = tk.Button(
    text_frame, text="Analizuj recenzje", font=("Arial", 12, "bold"), bg="#007bff", fg="white", relief=tk.RAISED,
    command=show_loading
)
analyze_button.pack(pady=10)

random_positive_button = tk.Button(
    text_frame, text="Załaduj losową pozytywną recenzję", font=("Arial", 10, "bold"), bg="#28a745", fg="white",
    command=load_random_positive
)
random_positive_button.pack(pady=5)

random_negative_button = tk.Button(
    text_frame, text="Załaduj losową negatywną recenzję", font=("Arial", 10, "bold"), bg="#dc3545", fg="white",
    command=load_random_negative
)
random_negative_button.pack(pady=5)

result_label = tk.Label(text_frame, text="", font=("Arial", 12, "bold"), bg="#f0f0f5")
result_label.pack(pady=10)

# Generowanie chmur słów
generate_wordcloud([review for review, sentiment in zip(reviews, sentiments) if sentiment == 1], "Chmura słów pozytwynych")
generate_wordcloud([review for review, sentiment in zip(reviews, sentiments) if sentiment == 0], "Chmura słów negatywnych")

root.mainloop()
