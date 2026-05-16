Zadanie polega na stworzeniu modelu rekurencyjnego który przewidywał będzie kompozytora danego utworu muzyki klasycznej w oparciu o jego zapis w formie sekwencji akordów. Akordy znormalizowane zostały do klucza C-dur lub a-mol w zależności od skali utworu (durowa/molowa).
Dane przygotowane są w postaci Pickli - https://docs.python.org/3/library/pickle.html w których znajduje się lista krotek z sekwencjami i odpowiadającymi im klasami - autorami odpowiednio: {0: 'bach', 1: 'beethoven', 2: 'debussy', 3: 'scarlatti', 4: 'victoria'} (train.pkl). W pliku test_no_target znajdują się testowe sekwencje, dla których predykcje mają Państwo przewidzieć.

Uwaga, utwory mogą być oczywiście różnych długości. W celu stworzenia batcha danych różnej długości, muszą je Państwo odpowiednio przygotować stosując tzw. padding. Przykładowo można się posiłkować tym tutorialem: https://suzyahyah.github.io/pytorch/2019/07/01/DataLoader-Pad-Pack-Sequence.html. W Państwa przypadku będzie to trochę łatwiejsze bo dotyczy problemu klasyfikacji sekwencji, a nie tłumaczenia sequence-to-sequence.

UWAGA! Proszę dokładnie zastosować się do poniższej instrukcji. Proszę sprawdzić czy każdy podpunkt się zgadza!

- W ramach rozwiązania, proszę oddać poprzez Teamsy jeden plik .zip zawierający: kod (w formie notebooka, lub skryptu/skryptów .py) oraz plik .csv z predykcjami na zbiorze test_data.csv. BEZ dodatkowych podfolderów i BEZ danych.
- Bardzo proszę nazwać plik .zip nazwiskami i imionami obu autorów z grupy ALFABETYCZNIE. Nazwę głównego archiwum .zip proszę dodatkowo rozpocząć od przedrostka poniedzialek_ lub piatek_ lub sroda_ (NIE pon/pia/śr /inne wersje). Przykład: sroda_KowalAndrzej_ZowalHanna.zip
- Proszę nie umieszczać plików w dodatkowych podfolderach tylko bezpośrednio.
- Proszę plik z predykcjami nazwać pred.csv
- Testowe predykcje powinny mieć kolejność zgodną z kolejnością sekwencji do których się odnoszą w picklu. Plik csv. nie powinien mieć nagłówka ani indeksów.
- W MS Teams wszystkim przydzieliłam zadanie, ale bardzo proszę, żeby tylko jeden (dowolny) członek zespołu je zwrócił.
