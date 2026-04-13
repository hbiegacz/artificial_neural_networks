**Treść zadania.**
Zadanie polega na stworzeniu klasyfikatora obrazków przypisującego zadanym wejściom jedną z 50 klas, takich jak przedmioty czy zwierzęta. Do dyspozycji mają Państwo:
- `train.zip` - archiwum ze zbiorem treningowym, w którym obrazom przyporządkowane zostały ich klasy na podstawie podfolderów, w których się one znajdują. Zbiór treningowy przygotowany jest tak, żeby można było go łatwo załadować za pomocą klasy `ImageFolder`, przykładowo jako:
```
import torchvision

trainset = torchvision.datasets.ImageFolder("train/")

map_class_to_idx = dataset_clean.class_to_idx  # map nazwy klasy na index
```
- `test.zip` - zbiór testowy z obrazami, bez przypisanych klas. 

Skuteczność stworzonego przez Państwa modelu będę sprawdzał w oparciu o **średnie accuracy na wszystkich klasach**, którego kod zamieściłem w pliku `helpers.py`. 

**Uwaga!**
Proszę dokładnie zastosować się do poniższej instrukcji i sprawdzić czy przesłane przez Państwa rozwiązanie jest zgodne z każdym podpunktem:
- W ramach rozwiązania zabronione jest używanie gotowych modeli i ich architektur. Celem tego projektu jest opracowanie własnej architektury i wytrenowanie jej od początku, wyłącznie w oparciu o dostarczone dane (lub ich augmentacje). **Skorzystanie w JAKIKOLWIEK sposób z gotowych modeli będzie wiązać się z wyzerowaniem Państwa punktów w ramach tego projektu!** 
- Jako rozwiązanie proszę oddać poprzez MS Teams Assignment jeden plik .zip zawierający: kod (w formie notebooka lub skryptu .py) oraz plik `pred.csv` z predykcjami wykonanymi na zbiorze testowym.
- Proszę nazwać plik .zip nazwiskami i imionami obu autorów z grupy ALFABETYCZNIE. Nazwę archiwum zip proszę dodatkowo rozpocząć od przedrostka poniedzialek_ lub sroda_ lub piatek_ (NIE pon/pia/śr /inne wersje).  Przykład: sroda_MałyszAdam_ŚwiątekIga.zip
- Proszę plik z predykcjami nazwać `pred.csv`. W pliku tym powinny znajdować się dokładnie dwie kolumny: nazwy plików wskazujących zdjęcia ze zbioru testowego (str) oraz przypisane im klasy w zakresie 0-49 (int). Koniecznie proszę sprawdzić format zwracanych przez Państwa predykcji (tyle predykcji, ile elementów w zbiorze testowym, brak nagłówków, dwie kolumny itd.). W plikach zamieściłem przykładowy plik "pred.csv" w oczekiwanym formacie. W szczególności, proszę zwrócić uwagę, żeby pliki nazywały się dokładnie tak samo jak w zbiorze testowym, czyli np. IMG_000134.JPEG (a nie: 000134.JPEG, IMG_000134.jpeg, IMG_000134).
- Proszę nie umieszczać plików w dodatkowych podfolderach, tylko bezpośrednio w archiwum. Struktura przykładowego pliku zip:
```
sroda_MałyszAdam_ŚwiątekIga.zip
├── pred.csv
└── rozwiazanie.ipynb
```
  - Prosiłbym o nieprzesyłanie do mnie plików zbioru treningowego czy testowego.
  - W MS Teams zadanie jest przydzielone każdemu, ale proszę, żeby tylko jeden (dowolny) członek zespołu je zwrócił.
  
**Niezastosowanie się do instrukcji może skutkować obniżeniem punktacji, ponieważ ewaluacja wyników jest automatyczna, a niespójne nazwy i pliki mogą spowodować złe wczytanie plików do testowania.**

W razie pytań zapraszam na konsultacje.