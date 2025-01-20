import tkinter as tk
from PIL import Image, ImageTk
import re
import pandas as pd
import numpy as np
import math
import random
from nltk.corpus import stopwords
from nltk.util import ngrams
import nltk

# Upewnij się, że stopwords zostały pobrane
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
# Usunięcie negacji z stopwords
negation_words = {"not", "no", "never", "wasn't"}
stop_words -= negation_words

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
    data = pd.read_csv('imdb.csv', nrows=limit)
    data['review'] = data['review'].apply(preprocess_text)
    data['sentiment'] = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    return data['review'].tolist(), data['sentiment'].tolist()

# Funkcja przetwarzania tekstu
def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)  # Usuwa znaczniki HTML
    text = re.sub(r'http\S+|www\.\S+', '', text)  # Usuwa linki
    text = re.sub(r'[^a-zA-Z ]', '', text).lower()  # Usuwa znaki nieliterowe i konwertuje na małe litery
    text = re.sub(r'\s+', ' ', text).strip()  # Usuwa dodatkowe białe znaki
    words = text.split()
    # Użycie bigramów
    bigrams = ['_'.join(gram) for gram in ngrams(words, 2)]
    words.extend(bigrams)
    return ' '.join([word for word in words if word not in stop_words])

# Funkcja do ograniczenia słownika do najczęściej występujących słów
def build_vocab(reviews, max_vocab_size=5000):
    all_words = ' '.join(reviews).split()
    word_counts = pd.Series(all_words).value_counts()
    vocab = {word: i for i, word in enumerate(word_counts.head(max_vocab_size).index)}
    return vocab

# Funkcja do konwersji recenzji na wektor cech
def vectorize_reviews(reviews, vocab):
    vectors = np.zeros((len(reviews), len(vocab)))
    for i, review in enumerate(reviews):
        for word in review.split():
            if word in vocab:
                vectors[i, vocab[word]] += 1
    return vectors

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
        for epoch in range(epochs):
            for i in range(len(inputs)):
                self.forward(inputs[i])
                self.backward(inputs[i], targets[i], learning_rate)

# Wczytanie danych
reviews, sentiments = load_data('imdb.csv', limit=15000)
vocab = build_vocab(reviews, max_vocab_size=20000)
input_vectors = vectorize_reviews(reviews, vocab)

# Inicjalizacja i trenowanie modelu
mlp = MLP(input_vectors.shape[1], 10, 1)
mlp.train(input_vectors, np.array(sentiments).reshape(-1, 1), epochs=10, learning_rate=0.1)

# Funkcja analizy recenzji
def analyze_sentiment():
    review = entry.get("1.0", tk.END).strip()  # Pobiera tekst z pola
    if review:
        preprocessed_review = preprocess_text(review)
        vector = vectorize_reviews([preprocessed_review], vocab)[0]
        prediction = mlp.forward(vector)[0]
        sentiment = "Positive" if prediction > 0.5 else "Negative"
        print(f"Review sentiment: {sentiment}")
    else:
        print("Please enter or select a review.")

# Funkcja analizy z animacją ładowania
def show_loading():
    loading_label = tk.Label(
        text_frame, text="Analizuje...", font=("Arial", 12, "bold"), bg=text_frame.cget("bg"), fg="#007bff"
    )
    loading_label.pack(pady=(5, 0))
    root.update_idletasks()
    root.after(2000, lambda: loading_label.destroy())
    analyze_sentiment()

# Tworzenie głównego okna
root = tk.Tk()
root.title("Movie Review Sentiment Analyzer")
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

for review in positive_reviews:
    btn = tk.Button(
        list_frame, text=review, wraplength=200, anchor="w", justify="left",
        font=("Arial", 9), bg="#e6e6e6", relief=tk.GROOVE,

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

root.mainloop()
