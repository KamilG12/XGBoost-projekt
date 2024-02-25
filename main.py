import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Wczytanie danych z pliku CSV
df_credit = pd.read_csv("german_credit_data.csv", sep=',', index_col=0)
df_credit = df_credit.dropna()
df_credit['Risk'] = df_credit['Risk'].map({'good': 0, 'bad': 1})
df_credit['Sex'] = df_credit['Sex'].map({'male': 0, 'female': 1})
df_credit['Housing'] = df_credit['Housing'].map({'own': 0, 'free': 1, 'rent': 2})
df_credit['Saving accounts'] = df_credit['Saving accounts'].map({'little': 0, 'quite rich': 1, 'rich': 2, 'moderate': 3})
df_credit['Checking account'] = df_credit['Checking account'].map({'little': 0, 'quite rich': 1, 'rich': 2, 'moderate': 3})
df_credit['Purpose'] = df_credit['Purpose'].map({'radio/TV': 0, 'education': 1, 'furniture/equipment': 2, 'car': 3, 'business': 4,
                                                 'domestic appliances': 5, 'repairs': 6, 'vacation/others': 7})

# Podział danych na cechy (X) i etykiety (y)
X = df_credit.drop('Risk', axis=1)
y = df_credit['Risk']

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

# Inicjalizacja modelu XGBoost
model = xgb.XGBClassifier(learning_rate= 0.1, max_depth= 8, n_estimators= 300, objective='binary:logistic', random_state=12)

# Trenowanie modelu na danych treningowych
model.fit(X_train, y_train)

# Predykcja na danych testowych
y_pred = model.predict(X_test)

# Ocena dokładności modelu
accuracy = accuracy_score(y_test, y_pred)
print(f'Dokładność modelu: {accuracy}')

# Wyświetlenie raportu klasyfikacji
print('Raport klasyfikacji:')
print(classification_report(y_test, y_pred))

# Wyświetlenie ważności cech
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.sort_values(ascending=False, inplace=True)
print("Ważność cech:")
print(feature_importances)

# Przykładowe predykcje dla nowych danych
nowe_dane = pd.DataFrame({'Age': [45], 'Sex': [0], 'Job': [3], 'Housing': [0], 'Saving accounts': [1],
                           'Checking account': [0], 'Credit amount': [8000], 'Duration': [9], 'Purpose': [2]})
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
nowe_dane_scaled = scaler.transform(nowe_dane)
przewidywane_etykiety = model.predict(nowe_dane_scaled)

# Wyświetl przewidziane etykiety
print("Przewidziane etykiety dla nowych danych:")
print(przewidywane_etykiety)
# Wyświetl etykietę jako "zły" lub "dobry"
if przewidywane_etykiety[0] == 1:
    print('Zły kredyt')
else:
    print('Dobry kredyt')

nowe_dane = pd.DataFrame({'Age': [24, 36, 27, 45, 65, 35, 38, 42, 70, 52],
                          'Sex': [0, 1, 0, 0, 0, 1, 0, 0, 1, 1],
                          'Job': [1, 2, 3, 0, 2, 1, 2, 0, 1, 3],
                          'Housing': [0, 2, 0, 1, 1, 1, 0, 2, 0, 0],
                          'Saving accounts': [3, 0, 2, 2, 0, 1, 1, 0, 2, 0],
                          'Checking account': [0, 1, 2, 0, 0, 2, 1, 1, 2, 1],
                          'Credit amount': [1200, 5900, 12000, 8200, 4870, 9900, 2835, 6800, 2100, 6300],
                          'Duration': [6, 48, 12, 42, 24, 30, 24, 48, 12, 3],
                          'Purpose': [0, 0, 1, 2, 3, 1, 2, 3, 0, 3]})
nowe_dane_scaled = scaler.transform(nowe_dane)
przewidywane_etykiety = model.predict(nowe_dane_scaled)

# Wyświetlenie przewidywanych etykiet
for i, dane in enumerate(nowe_dane_scaled):
    przewidywana_etykieta = model.predict([dane])
    print(f'Nowy przypadek danych {i+1}: Kredyt jest {"zły" if przewidywana_etykieta[0] == 1 else "dobry"}')

