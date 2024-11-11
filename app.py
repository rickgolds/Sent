import tkinter as tk

# Funkcja obsługująca przycisk
def analyze_sentiment():
    review = entry.get("1.0", tk.END)  # Pobiera tekst z pola
    print(f"Wprowadź recenzje: {review}")  # Na razie tylko wypisuje tekst

# Tworzenie głównego okna
root = tk.Tk()
root.title("Analiza sentymentu")

# Pole tekstowe do wpisania recenzji
label = tk.Label(root, text="Wprowadz recenzje:")
label.pack()

entry = tk.Text(root, height=10, width=50)
entry.pack()

# Przycisk do analizy sentymentu
analyze_button = tk.Button(root, text="Analizuj", command=analyze_sentiment)
analyze_button.pack()

root.mainloop()
