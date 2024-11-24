import tkinter as tk

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

# Funkcja do wklejania recenzji do pola tekstowego
def insert_review(review):
    entry.delete("1.0", tk.END)  # Wyczyść pole tekstowe
    entry.insert(tk.END, review)  # Wklej recenzję

# Funkcja obsługująca przycisk
def analyze_sentiment():
    review = entry.get("1.0", tk.END).strip()  # Pobiera tekst z pola
    if review:
        print(f"Analyzing sentiment for review: {review}")  # Na razie tylko wypisuje tekst
    else:
        print("Please enter or select a review.")

# Tworzenie głównego okna
root = tk.Tk()
root.title("Sentiment Analysis")

# Tworzenie ramki dla listy recenzji
list_frame = tk.Frame(root)
list_frame.pack(side=tk.LEFT, padx=10, pady=10)

# Nagłówki dla list
positive_label = tk.Label(list_frame, text="Pozytywne recenzje", font=("Arial", 10, "bold"))
positive_label.pack(anchor="w")

# Lista pozytywnych recenzji
for review in positive_reviews:
    btn = tk.Button(list_frame, text=review, wraplength=200, anchor="w", justify="left",
                    command=lambda r=review: insert_review(r))
    btn.pack(fill="x", pady=2)

negative_label = tk.Label(list_frame, text="Negatywne recenzje", font=("Arial", 10, "bold"))
negative_label.pack(anchor="w", pady=(10, 0))

# Lista negatywnych recenzji
for review in negative_reviews:
    btn = tk.Button(list_frame, text=review, wraplength=200, anchor="w", justify="left",
                    command=lambda r=review: insert_review(r))
    btn.pack(fill="x", pady=2)

# Tworzenie ramki dla pola tekstowego i przycisku
text_frame = tk.Frame(root)
text_frame.pack(side=tk.RIGHT, padx=10, pady=10)

# Pole tekstowe do wpisania recenzji
label = tk.Label(text_frame, text="Wprowadź recenzje:")
label.pack()

entry = tk.Text(text_frame, height=10, width=50)
entry.pack()

# Przycisk do analizy sentymentu
analyze_button = tk.Button(text_frame, text="ANALIZUJ", command=analyze_sentiment)
analyze_button.pack(pady=10)

root.mainloop()
