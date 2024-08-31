# Predviđanje vrste i kvaliteta vina iz njegovih fizičkih svojstava
Kvalitet vina je ključan za potrošače i proizvođače u vinskoj industriji. Tradicionalno testiranje kvaliteta vina je vremenski zahtevno i resursno intenzivno, često uključujući subjektivne procene koje mogu varirati među stručnjacima. Ovaj projekat se fokusira na unapređenje efikasnosti i tačnosti ocenjivanja kvaliteta vina korišćenjem mašinskog učenja.

## Cilj Projekta
Cilj ovog projekta je da se koriste objektivne metode mašinskog učenja za identifikaciju ključnih karakteristika vina koje utiču na njegov kvalitet i vrstu. Korišćenjem skupa podataka o portugalskom vinu vinho verde, razvijaćemo modele za predviđanje kvaliteta vina na osnovu njegovih fizičko-hemijskih svojstava.

## Skup Podataka
Podaci su dostupni na [sledećem linku](https://archive.ics.uci.edu/dataset/186/wine+quality). Skup podataka sadrži 6497 uzoraka portugalskog vina sa sledećim karakteristikama: fiksna kiselost, hlapiva kiselost, koncentracija limunske kiseline, sadržaj rezidualnog šećera, količina hlorida, sadržaj slobodnog sumpor-dioksida, ukupan sadržaj sumpor-dioksida, gustina, pH, koncentracija sulfata, sadržaj alkohola. Svaki uzorak ima dodeljene oznake za stil vina (crveno ili belo) i subjektivnu ocenu kvaliteta (od 0 do 10).

## Instalacija
Da biste instalirali sve potrebne biblioteke i pakete, koristite sledeću komandu:

```bash
pip install seaborn ucimlrepo plotly
```
## Korišćene metode

U ovom projektu koristimo različite metode mašinskog učenja za predviđanje kvaliteta vina, uključujući linearnu regresiju, Naivni Bayes, SVM, slučajnu sumu i AdaBoost. Od svih ovih metoda, slučajna suma je pokazala najbolje rezultate, pružajući najvišu tačnost i stabilnost u analizi kvaliteta vina.

## Finalni model
Finalni model, zajedno sa pripadajućim skalarem i PCA transformatorom, smešten je u folderu **models**.

## Literatura 
[Predicting Style and Quality of Vinho Verde from
its Physical Properties](https://archive.ics.uci.edu/dataset/186/wine+quality)https://cs229.stanford.edu/proj2019aut/data/assignment_308832_raw/26646202.pdf

## Članovi tima
Marina Vasiljević 1061/2023  
Isidora Burmaz 1057/2023
