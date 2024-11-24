import tkinter as tk
from PIL import Image, ImageTk

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

# Funkcja analizy z animacją ładowania
def show_loading():
    # Tworzymy etykietę "Analyzing..." i umieszczamy ją pod przyciskiem
    loading_label = tk.Label(
        text_frame, text="Analizuje...", font=("Arial", 12, "bold"), bg=text_frame.cget("bg"), fg="#007bff"
    )
    loading_label.pack(pady=(5, 0))  # Odstęp 5 pikseli od przycisku

    # Symulujemy czas analizy i usuwamy etykietę po 2 sekundach
    root.update_idletasks()
    root.after(2000, lambda: loading_label.destroy())


def analyze_sentiment():
    review = entry.get("1.0", tk.END).strip()  # Pobiera tekst z pola
    if review:
        print(f"Analyzing sentiment for review: {review}")  # Na razie tylko wypisuje tekst
    else:
        print("Please enter or select a review.")

def interpolate_color(start_color, end_color, alpha):
    """Interpoluje między dwoma kolorami"""
    start_r, start_g, start_b = start_color
    end_r, end_g, end_b = end_color
    new_r = int(start_r + (end_r - start_r) * alpha)
    new_g = int(start_g + (end_g - start_g) * alpha)
    new_b = int(start_b + (end_b - start_b) * alpha)
    return f"#{new_r:02x}{new_g:02x}{new_b:02x}"


def toggle_theme_animated():
    steps = 20  # Liczba kroków animacji
    delay = 10  # Opóźnienie między krokami (w milisekundach)

    # Kolory dla jasnego i ciemnego trybu
    light_bg = (240, 240, 245)
    dark_bg = (0, 0, 0)
    light_text = (0, 0, 0)
    dark_text = (255, 255, 255)
    light_button_bg = (230, 230, 230)
    dark_button_bg = (51, 51, 51)
    light_entry_bg = (255, 255, 255)
    dark_entry_bg = (30, 30, 30)

    if root.cget("bg") == "#f0f0f5":  # Jasny -> Ciemny
        for i in range(steps):
            alpha = i / steps
            # Interpolacja kolorów
            bg_color = interpolate_color(light_bg, dark_bg, alpha)
            text_color = interpolate_color(light_text, dark_text, alpha)
            button_bg = interpolate_color(light_button_bg, dark_button_bg, alpha)
            entry_bg = interpolate_color(light_entry_bg, dark_entry_bg, alpha)

            # Aktualizacja tła
            root.configure(bg=bg_color)
            list_frame.configure(bg=bg_color)
            text_frame.configure(bg=bg_color)

            # Aktualizacja etykiet
            positive_label.configure(bg=bg_color, fg=text_color)
            negative_label.configure(bg=bg_color, fg=text_color)
            label.configure(bg=bg_color, fg=text_color)

            # Aktualizacja przycisków
            for widget in list_frame.winfo_children():
                if isinstance(widget, tk.Button):
                    widget.configure(bg=button_bg, fg=text_color)
            analyze_button.configure(bg=button_bg, fg=text_color)
            theme_button.configure(bg=button_bg, fg=text_color)

            # Aktualizacja pola tekstowego
            entry.configure(bg=entry_bg, fg=text_color, insertbackground=text_color)

            root.update_idletasks()
            root.after(delay)

        # Końcowe ustawienia dla trybu ciemnego
        root.configure(bg="#000000")
        list_frame.configure(bg="#000000")
        text_frame.configure(bg="#000000")
        positive_label.configure(bg="#000000", fg="#ffffff")
        negative_label.configure(bg="#000000", fg="#ffffff")
        label.configure(bg="#000000", fg="#ffffff")
        entry.configure(bg="#1e1e1e", fg="#ffffff", insertbackground="#ffffff")
        analyze_button.configure(bg="#007bff", fg="#ffffff", activebackground="#0056b3")
        theme_button.configure(bg="#007bff", fg="#ffffff", activebackground="#0056b3")
        for widget in list_frame.winfo_children():
            if isinstance(widget, tk.Button):
                widget.configure(bg="#333333", fg="#ffffff", activebackground="#444444")
    else:  # Ciemny -> Jasny
        for i in range(steps):
            alpha = i / steps
            # Interpolacja kolorów
            bg_color = interpolate_color(dark_bg, light_bg, alpha)
            text_color = interpolate_color(dark_text, light_text, alpha)
            button_bg = interpolate_color(dark_button_bg, light_button_bg, alpha)
            entry_bg = interpolate_color(dark_entry_bg, light_entry_bg, alpha)

            # Aktualizacja tła
            root.configure(bg=bg_color)
            list_frame.configure(bg=bg_color)
            text_frame.configure(bg=bg_color)

            # Aktualizacja etykiet
            positive_label.configure(bg=bg_color, fg=text_color)
            negative_label.configure(bg=bg_color, fg=text_color)
            label.configure(bg=bg_color, fg=text_color)

            # Aktualizacja przycisków
            for widget in list_frame.winfo_children():
                if isinstance(widget, tk.Button):
                    widget.configure(bg=button_bg, fg=text_color)
            analyze_button.configure(bg=button_bg, fg=text_color)
            theme_button.configure(bg=button_bg, fg=text_color)

            # Aktualizacja pola tekstowego
            entry.configure(bg=entry_bg, fg=text_color, insertbackground=text_color)

            root.update_idletasks()
            root.after(delay)

        # Końcowe ustawienia dla trybu jasnego
        root.configure(bg="#f0f0f5")
        list_frame.configure(bg="#f0f0f5")
        text_frame.configure(bg="#f0f0f5")
        positive_label.configure(bg="#f0f0f5", fg="#000000")
        negative_label.configure(bg="#f0f0f5", fg="#000000")
        label.configure(bg="#f0f0f5", fg="#000000")
        entry.configure(bg="#ffffff", fg="#000000", insertbackground="#000000")
        analyze_button.configure(bg="#007bff", fg="#ffffff", activebackground="#0056b3")
        theme_button.configure(bg="#007bff", fg="#ffffff", activebackground="#0056b3")
        for widget in list_frame.winfo_children():
            if isinstance(widget, tk.Button):
                widget.configure(bg="#e6e6e6", fg="#000000", activebackground="#d9d9d9")





# Funkcja podświetlania przycisków
def hover_effect(widget, enter_color_light, enter_color_dark, leave_color_light, leave_color_dark):
    def on_enter(e):
        if root.cget("bg") == "#f0f0f5":  # Jasny motyw
            e.widget.config(bg=enter_color_light)
        else:  # Ciemny motyw
            e.widget.config(bg=enter_color_dark)
        e.widget.config(cursor="hand2")

    def on_leave(e):
        if root.cget("bg") == "#f0f0f5":  # Jasny motyw
            e.widget.config(bg=leave_color_light)
        else:
            e.widget.config(bg=leave_color_dark)

    widget.bind("<Enter>", on_enter)
    widget.bind("<Leave>", on_leave)


# Tworzenie głównego okna
root = tk.Tk()
root.title("Movie Review Sentiment Analyzer")
root.configure(bg="#f0f0f5")  # Tło aplikacji

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
    image=positive_icon, compound="left", padx=10  # Ikona po lewej stronie tekstu
)
positive_label.pack(anchor="w")

# Lista pozytywnych recenzji
for review in positive_reviews:
    btn = tk.Button(
        list_frame, text=review, wraplength=200, anchor="w", justify="left",
        font=("Arial", 9), bg="#e6e6e6", relief=tk.GROOVE,
        command=lambda r=review: insert_review(r)
    )
    btn.pack(fill="x", pady=2)

negative_label = tk.Label(
    list_frame, text="Negatywne recenzje", font=("Arial", 10, "bold"), bg="#f0f0f5",
    image=negative_icon, compound="left", padx=10  # Ikona po lewej stronie tekstu
)
negative_label.pack(anchor="w", pady=(10, 0))

# Lista negatywnych recenzji
for review in negative_reviews:
    btn = tk.Button(
        list_frame, text=review, wraplength=200, anchor="w", justify="left",
        font=("Arial", 9), bg="#e6e6e6", relief=tk.GROOVE,
        command=lambda r=review: insert_review(r)
    )
    btn.pack(fill="x", pady=2)

# Tworzenie ramki dla pola tekstowego i przycisku
text_frame = tk.Frame(root, bg="#f0f0f5")
text_frame.pack(side=tk.RIGHT, padx=10, pady=10)

# Pole tekstowe do wpisania recenzji
label = tk.Label(text_frame, text="Wprowadź recenzję:", font=("Arial", 12, "bold"), bg="#f0f0f5")
label.pack()

entry = tk.Text(text_frame, height=10, width=50, font=("Arial", 10), bg="#ffffff")
entry.pack()

# Przycisk do analizy sentymentu
analyze_button = tk.Button(
    text_frame, text="Analizuj recenzje", font=("Arial", 12, "bold"), bg="#007bff", fg="white", relief=tk.RAISED,
    command=show_loading
)
analyze_button.pack(pady=10)

# Przycisk do przełączania trybu
theme_button = tk.Button(
    root, text="Zmień tryb", font=("Arial", 12, "bold"), bg="#007bff", fg="white", relief=tk.RAISED,
    command=toggle_theme_animated
)
theme_button.pack(side=tk.BOTTOM, pady=10)

# Wczytanie i przeskalowanie logo
logo_image = Image.open("logo.png").resize((50, 50), Image.Resampling.LANCZOS)
logo = ImageTk.PhotoImage(logo_image)

# Dodanie logo w prawym górnym rogu
logo_label = tk.Label(root, image=logo, bg="#f0f0f5")
logo_label.place(relx=1.0, rely=0.0, anchor="ne", x=-10, y=10)  # Prawy górny róg z marginesem

# Zastosowanie efektu hover PO stworzeniu wszystkich przycisków
for widget in list_frame.winfo_children():
    if isinstance(widget, tk.Button):
        hover_effect(
            widget,
            enter_color_light="#d9d9d9",  # Jasne podświetlenie w jasnym motywie
            enter_color_dark="#444444",   # Subtelne podświetlenie w ciemnym motywie
            leave_color_light="#e6e6e6",  # Domyślne tło w jasnym motywie
            leave_color_dark="#333333"    # Domyślne tło w ciemnym motywie
        )

hover_effect(
    analyze_button,
    enter_color_light="#0056b3",  # Jasny niebieski w jasnym motywie
    enter_color_dark="#555555",   # Lekki szary w ciemnym motywie
    leave_color_light="#007bff",  # Domyślne tło w jasnym motywie
    leave_color_dark="#333333"    # Domyślne tło w ciemnym motywie
)

hover_effect(
    theme_button,
    enter_color_light="#0056b3",  # Jasny niebieski w jasnym motywie
    enter_color_dark="#555555",   # Lekki szary w ciemnym motywie
    leave_color_light="#007bff",  # Domyślne tło w jasnym motywie
    leave_color_dark="#333333"    # Domyślne tło w ciemnym motywie
)

root.mainloop()
