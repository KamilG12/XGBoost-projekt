Ten projekt wykorzystuje algorytm XGBoost do przewidywania ryzyka kredytowego na podstawie niemieckich danych kredytowych. Model klasyfikuje kredyty jako "dobry" (niski poziom ryzyka) lub "zły" (wysoki poziom ryzyka) na podstawie cech klienta.

Funkcjonalności
Wstępne przetwarzanie danych (mapowanie wartości kategorycznych, usuwanie braków danych).
Trenowanie modelu XGBoost z optymalnymi hiperparametrami.
Ocena modelu za pomocą:
Dokładności (accuracy),
Raportu klasyfikacji (classification report).
Analiza ważności cech wpływających na ryzyko kredytowe.
Predykcja dla nowych klientów.
Dane wejściowe
Cechy klientów, takie jak wiek, płeć, cel kredytu, kwota kredytu, oszczędności, itp.
Wymagania
Python
Biblioteki: pandas, scikit-learn, xgboost
Sposób działania
Wczytanie i wstępne przetwarzanie danych:

Usunięcie brakujących danych.
Zamiana wartości tekstowych na numeryczne (np. "male" → 0, "female" → 1).
Trenowanie modelu:

Model XGBoost jest trenowany na zbiorze treningowym.
Ocena modelu:

Obliczenie dokładności oraz wygenerowanie raportu klasyfikacji.
Predykcja:

Przewidywanie ryzyka kredytowego na podstawie nowych danych wejściowych.
Wyniki
Dokładność modelu: Wyświetlana w konsoli.
Ważność cech: Analiza, które cechy najbardziej wpływają na ryzyko kredytowe.
Przykładowe predykcje: Model przewiduje, czy kredyt jest "dobry" czy "zły" dla nowych przypadków
