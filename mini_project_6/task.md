Zadanie polega na stworzeniu modelu który klasyfikował będzie mowę nienawiści na podstawie otagowanych komentarzy w języku polskim.
Do rozwiązania zadania mogą Państwo skorzystać z dowolnego przetrenowanego modelu językowego (np. takiego jaki był pokazywany na zajęciach), mogą Państwo sami swój model wytrenować, podejście do rozwiązania jest dowolne*, byleby osiągnąć jak najlepszą dokładność :)
*Istnieją dwa podejścia niedozwolone, proszę ich nie robić :)
Wykonywanie predykcji samemu, ale wątpię że chciałoby się Państwu to robić pod koniec semestru:)
Wrzucenie pliku do LLMa, np ChatGPT z instrukcją wykonania predykcji

Dane:
Do dyspozycji mają Państwo dane treningowe w pliku `hate_train.csv` (zwracam uwagę że nie jest ich za dużo). Każda próbka danych zapisana jest w oddzielnej linijce.

Instrukcje:
- W ramach rozwiązania proszę o zwrócenie poprzez Teams pojedynczego archiwum zip, zawierający:
    1. Kod (w formie notebooka/ skryptu python)
    2. Plik `pred.csv` z predykcjami na zbiorze testowym (`hate_test_data.txt`). 
UWAGA! Plik powinien mieć tyle linijek ile jest próbek w zbiorze testowym, powinien on nie zawierać nagłówka i mieć dokładnie jedna kolumnę. Każda linijka pliku powinna zawierać klasę (liczbę, nie nazwę klasy!!) jako predykcję modelu.
- Bardzo proszę nazwać plik .zip nazwiskami i imionami obu autorów z grupy ALFABETYCZNIE. Nazwę głównego archiwum .zip proszę dodatkowo rozpocząć od przedrostka poniedzialek_ lub piatek_ lub sroda_ (NIE pon/pia/śr /inne wersje). Przykład: sroda_KowalAndrzej_ZowalHanna.zip
- Proszę nie umieszczać plików w dodatkowych podfolderach tylko bezpośrednio.
- W MS Teams wszystkim przydzieliłam zadanie, ale bardzo proszę, żeby tylko jeden (dowolny) członek zespołu je zwrócił.

Niezastosowanie się do instrukcji może skutkować obniżeniem punktacji - ewaluacja wyników jest automatyczna, niespójne nazwy i pliki mogą spowodować złe wczytanie plików do testowania.
W związku z końcem semestru, nie będzie możliwości dosyłania rozwiązań po upływie terminu.

Powodzenia!