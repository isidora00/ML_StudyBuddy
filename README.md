# Predviđanje vrste i kvaliteta vina iz njegovih fizičkih svojstava
Kvalitet vina je ključan za potrošače i proizvođače u vinskoj industriji. Tradicionalno testiranje kvaliteta vina je vremenski zahtevno i resursno intenzivno, često uključujući subjektivne procene koje mogu varirati među stručnjacima. Ovaj projekat se fokusira na unapređenje efikasnosti i tačnosti ocenjivanja kvaliteta vina korišćenjem mašinskog učenja.

## Cilj Projekta
Cilj ovog projekta je da se koriste objektivne metode mašinskog učenja za identifikaciju ključnih karakteristika vina koje utiču na njegov kvalitet i vrstu. Korišćenjem skupa podataka o portugalskom vinu vinho verde, razvijaćemo modele za predviđanje kvaliteta vina na osnovu njegovih fizičko-hemijskih svojstava.

## Skup podataka
Skup podataka koji ćemo koristiti za ovaj problem je javno dostupan na [sledećem linku](https://archive.ics.uci.edu/dataset/186/wine+quality). Ovaj skup podataka sadrži 6497 primeraka portugalskog vina *vinho verde*. Podaci su nebalansirano raspodeljeni između dve vrste vina: 75% uzoraka su bela vina (4898), dok 25% čine crvena vina (1599). Svaki uzorak je opisan sa 11 fizičko-hemijskih karakteristika, koje su sve neprekidne veličine: fiksna kiselost, hlapiva kiselost, koncentracija limunske kiseline, sadržaj rezidualnog šećera, koncentracije hlorida, sadržaj slobodnog sumpor-dioksida, ukupan sadržaj sumpor-dioksida, gustina, pH, koncentracije sulfata i sadržaj alkohola. Svakom uzorku dodeljene su i dve oznake: stil vina (crveno ili belo) i subjektivna ocena (celi brojevi između 0 i 10) koju je odredio panel od tri somelijera.

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
