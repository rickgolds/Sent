import tkinter as tk
from PIL import Image, ImageTk
import re
import csv
import math
import random
from collections import Counter

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
    reviews, sentiments = [], []
    with open('imdb.csv', 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader):
            if i >= limit:
                break
            reviews.append(row['review'])
            sentiments.append(1 if row['sentiment'] == 'positive' else 0)
    return reviews, sentiments

# Funkcja przetwarzania tekstu
def preprocess_text(text):
    stopwords = {"a", "an", "the", "and", "is", "in", "it", "of", "to", "was", "with"}
    text = re.sub(r'[^a-zA-Z ]', '', text).lower()
    words = text.split()
    return [word for word in words if word not in stopwords]

# Funkcja do konwersji recenzji na wektor cech
def vectorize_reviews(reviews, vocab):
    vectors = []
    for review in reviews:
        vector = [0] * len(vocab)
        words = preprocess_text(review)
        for word in words:
            if word in vocab:
                vector[vocab[word]] += 1
        vectors.append(vector)
    return vectors

# Funkcja do ograniczenia słownika do najczęściej występujących słów
def build_vocab(reviews, max_vocab_size=5000):
    word_counter = Counter()
    for review in reviews:
        words = preprocess_text(review)
        word_counter.update(words)
    most_common = word_counter.most_common(max_vocab_size)
    return {word: i for i, (word, _) in enumerate(most_common)}

# Implementacja MLP
class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights_input_hidden = [[random.uniform(-0.5, 0.5) for _ in range(hidden_size)] for _ in range(input_size)]
        self.weights_hidden_output = [[random.uniform(-0.5, 0.5) for _ in range(output_size)] for _ in range(hidden_size)]
        self.bias_hidden = [random.uniform(-0.5, 0.5) for _ in range(hidden_size)]
        self.bias_output = [random.uniform(-0.5, 0.5) for _ in range(output_size)]

    def sigmoid(self, x):
        x = max(-700, min(700, x))
        return 1 / (1 + math.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, inputs):
        self.hidden_layer = [0] * self.hidden_size
        for i in range(self.hidden_size):
            self.hidden_layer[i] = sum(inputs[j] * self.weights_input_hidden[j][i] for j in range(self.input_size)) + self.bias_hidden[i]
            self.hidden_layer[i] = self.sigmoid(self.hidden_layer[i])

        self.output_layer = [0] * self.output_size
        for i in range(self.output_size):
            self.output_layer[i] = sum(self.hidden_layer[j] * self.weights_hidden_output[j][i] for j in range(self.hidden_size)) + self.bias_output[i]
            self.output_layer[i] = self.sigmoid(self.output_layer[i])

        return self.output_layer

    def backward(self, inputs, targets, learning_rate):
        output_errors = [targets[i] - self.output_layer[i] for i in range(self.output_size)]
        hidden_errors = [0] * self.hidden_size

        for i in range(self.hidden_size):
            for j in range(self.output_size):
                hidden_errors[i] += output_errors[j] * self.weights_hidden_output[i][j]

        for i in range(self.hidden_size):
            for j in range(self.output_size):
                self.weights_hidden_output[i][j] += learning_rate * output_errors[j] * self.hidden_layer[i]

        for i in range(self.input_size):
            for j in range(self.hidden_size):
                self.weights_input_hidden[i][j] += learning_rate * hidden_errors[j] * inputs[i]

        for i in range(self.hidden_size):
            self.bias_hidden[i] += learning_rate * hidden_errors[i]
        for i in range(self.output_size):
            self.bias_output[i] += learning_rate * output_errors[i]

    def train(self, inputs, targets, epochs, learning_rate):
        for epoch in range(epochs):
            for i in range(len(inputs)):
                self.forward(inputs[i])
                self.backward(inputs[i], targets[i], learning_rate)

# Funkcja generowania raportu klasyfikacji
def classification_report(mlp, input_vectors, targets):
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for i in range(len(input_vectors)):
        prediction = mlp.forward(input_vectors[i])[0]
        predicted_label = 1 if prediction > 0.5 else 0
        true_label = targets[i][0]

        if predicted_label == 1 and true_label == 1:
            true_positives += 1
        elif predicted_label == 1 and true_label == 0:
            false_positives += 1
        elif predicted_label == 0 and true_label == 0:
            true_negatives += 1
        elif predicted_label == 0 and true_label == 1:
            false_negatives += 1

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("Classification Report:")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1_score:.2f}")

# Wczytanie danych
reviews, sentiments = load_data('imdb.csv', limit=5000)  # Ograniczenie do 1000 recenzji
vocab = build_vocab(reviews, max_vocab_size=10000)  # Ograniczenie słownika
input_vectors = vectorize_reviews(reviews, vocab)

# Inicjalizacja i trenowanie modelu
mlp = MLP(len(vocab), 10, 1)
mlp.train(input_vectors, [[s] for s in sentiments], epochs=10, learning_rate=0.1)

# Wygenerowanie raportu klasyfikacji
classification_report(mlp, input_vectors, [[s] for s in sentiments])

def analyze_sentiment():
    review = entry.get("1.0", tk.END).strip()  # Pobiera tekst z pola
    if review:
        vector = vectorize_reviews([review], vocab)[0]
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
