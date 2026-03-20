# Treść zadania:

Załóżmy, że chcemy kupić mieszkanie. Do dyspozycji mamy 100 000$, możemy też wziąć kredyt na kolejne 250 000 $, co da nam w sumie budżet w wysokości 350 000 $. Stwórzmy model który pomoże nam przewidzieć, czy mieszkanie o zadanych parametrach możemy kupić za własne pieniądze (cheap), z kredytem (average), czy jest poza naszym zasięgiem (expensive).

W oparciu o dostępne atrybuty zbuduj model, który pomoże oszacować, czy dana nieruchomość należy do klasy cheap, average czy expensive. Do dyspozycji mają Państwo dane treningowe (train_data.csv) z oryginalnymi cenami nieruchomości (SalePrice), oraz zbiór testowy (test_data.csv).

Końcowe wyniki obliczane będą w oparciu o średnią dokładność (accuracy) dla każdej klasy. Proszę zwrócić uwagę na fakt, że klasy są mocno niezbalansowane! W pliku evaluation.py znajduje się funkcja, przy użyciu której obliczana będzie skuteczność Państwa modelu.

# Uwaga!

Proszę dokładnie zastosować się do poniższej instrukcji i sprawdzić czy przesyłane przez Państwa rozwiązanie jest zgodne z każdym podpunktem:
- Jako rozwiązanie proszę oddać poprzez MS Teams Assignment jeden plik .zip zawierający: kod (w formie notebooka lub skryptu .py) oraz plik .csv z predykcjami wykonanymi na zbiorze test_data.csv.
- Proszę nazwać plik .zip nazwiskami i imionami obu autorów z grupy ALFABETYCZNIE. Nazwę archiwum zip proszę dodatkowo rozpocząć od przedrostka poniedzialek_ lub sroda_ lub piatek_ (NIE pon/pia/śr /inne wersje).  Przykład: sroda_MałyszAdam_ŚwiątekIga.zip
- Proszę plik z predykcjami nazwać "pred.csv". W pliku z predykcjami powinna się znajdować dokładnie jedna kolumna, oznaczająca przewidywaną przez Państwa klasę (int) ceny mieszkania (0: cheap, 1: average, 2: expensive). Koniecznie proszę sprawdzić format zwracanych przez Państwa predykcji (tyle predykcji ile elementów w zbiorze testowym, brak nagłówków, jedna kolumna, itd.). W plikach zamieściłem przykładowy plik "pred.csv" w oczekiwanym formacie.
- Proszę nie umieszczać plików w dodatkowych podfolderach tylko bezpośrednio w archiwum. Struktura przykładowego pliku zip:
```
sroda_MałyszAdam_ŚwiątekIga.zip
├── pred.csv
└── rozwiazanie.ipynb
```
- W MS Teams zadanie jest przydzielone każdemu, ale proszę, żeby tylko jeden (dowolny) członek zespołu je zwrócił.

Niezastosowanie się do w/w instrukcji może skutkować obniżeniem punktacji, ponieważ ewaluacja wyników jest automatyczna, a niespójne nazwy i pliki mogą spowodować złe wczytanie plików do testowania.

W razie pytań zapraszam na konsultacje.