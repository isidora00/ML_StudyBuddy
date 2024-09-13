# Predviđanje vrste i kvaliteta vina iz njegovih fizičkih svojstava
Kvalitet vina je ključan za potrošače i proizvođače u vinskoj industriji. Tradicionalno testiranje kvaliteta vina je vremenski zahtevno i resursno intenzivno, često uključujući subjektivne procene koje mogu varirati među stručnjacima. Ovaj projekat se fokusira na unapređenje efikasnosti i tačnosti ocenjivanja kvaliteta vina korišćenjem mašinskog učenja.

## Cilj Projekta
Cilj ovog projekta je da se koriste metode mašinskog učenja za identifikaciju ključnih karakteristika vina koje utiču na njegov kvalitet i vrstu. Korišćenjem skupa podataka o portugalskom vinu vinho verde, razvijaćemo modele za predviđanje kvaliteta vina na osnovu njegovih fizičko-hemijskih svojstava, a na osnovu skupa podataka o recenzijama vina model za klasifikovaje vina po senzornim opisima, poput ukusa, mirisa i teksture. 

## Skup podataka
Skupovi podataka koji ćemo koristiti za ovaj problem su javno dostupni na sledećim linkovima: [vinho verde](https://archive.ics.uci.edu/dataset/186/wine+quality) [wine review](https://www.kaggle.com/datasets/zynicide/wine-reviews). 

Prvi skup podataka sadrži 6497 primeraka portugalskog vina *vinho verde*. Podaci su nebalansirano raspodeljeni između dve vrste vina: 75% uzoraka su bela vina (4898), dok 25% čine crvena vina (1599). Svaki uzorak je opisan sa 11 fizičko-hemijskih karakteristika, koje su sve neprekidne veličine: fiksna kiselost, hlapiva kiselost, koncentracija limunske kiseline, sadržaj rezidualnog šećera, koncentracije hlorida, sadržaj slobodnog sumpor-dioksida, ukupan sadržaj sumpor-dioksida, gustina, pH, koncentracije sulfata i sadržaj alkohola. Svakom uzorku dodeljene su i dve oznake: stil vina (crveno ili belo) i subjektivna ocena (celi brojevi između 0 i 10) koju je odredio panel od tri somelijera.

Drugi skup podataka sadrži preko 130.000 recenzija različitih vrsta vina iz celog sveta sa sajta https://wineenthusiast.com/. Svaka recenzija sadrži tekstualni opis vina, kao i niz metapodataka kao što su ime vina, poreklo, vrsta grožđa, cena, i ime recenzenta. Podaci uključuju senzorne opise, poput ukusa, mirisa i teksture, koji se koriste za dalje analize i klasifikaciju stilova vina. Podaci su nebalansirani, s velikim brojem recenzija za neke vrsta vina i znatno manje za druge.

## Instalacija
Da biste instalirali sve potrebne biblioteke i pakete, koristite sledeću komandu:

```bash
conda env create -f environment.yml
```
## Korišćene metode

U ovom projektu koristimo različite metode mašinskog učenja za predviđanje kvaliteta vina, uključujući linearnu regresiju, Naivni Bayes, SVM, slučajnu sumu i AdaBoost. Od svih ovih metoda, slučajna suma je pokazala najbolje rezultate, pružajući najvišu tačnost i stabilnost u analizi kvaliteta vina.

## Finalni modeli
Finalni modeli, zajedno sa pripadajućim skalarima i PCA transformatorima, smešteni su u folderu **models**.

## Literatura 
[Predicting Style and Quality of Vinho Verde from
its Physical Properties](https://archive.ics.uci.edu/dataset/186/wine+quality)https://cs229.stanford.edu/proj2019aut/data/assignment_308832_raw/26646202.pdf

## Članovi tima
Marina Vasiljević 1061/2023  
Isidora Burmaz 1057/2023
