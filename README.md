# conv-alt-nn

Progetto per tentare di sviluppare un layer convoluzionale alternativo che porti a dei filtri e attivazioni
maggiormente 'spiegabili'.

Nei layer convolutivi tradizionali, le attivazioni successive al primo layer non sono più interpretabili
come immagini in quanto il pattern di connettività tra i layer porta a 'mischiare' le informazioni contenute
tra i vari "strati" del volume di attivazione in input in una sola mappa di attivazione in output. 
Al primo layer questo è poco problematico perchè lo  "spessore" è quello dell'immagine stessa: 1 se greyscale,
3 se rgb. Ma visto che il numero di filtri applicati nel layer convoluzionale standard è un iperparametro, 
al secondo layer convoluzionale avrò un output formato da un qualche numero di mappe di attivazione (lo stesso
iperparametro) in cui ognuna di queste dipende dal numero di filtri applicati al layer precedente. Le mappe
di attivazione risultanti da queste non sono più, quindi, delle immagini, ma dei pattern di attivazione di feature
estratte dal primo livello convolutivo; e così via nei layer successivi.

Esiste un modo per far sì che i layer convolutivi abbiano in input immagini e in output ancora immagini?
Questo progetto si pone tale obiettivo. I benefici potrebbero essere di due tipi diversi:

* ad immagini in input ad un layer corrispondono immagini in output. questo permette di verificare almeno visivamente
quale tipo di processing è stato applicato ad una immagine, e ad utilizzare la letteratura già presente riguardo l'image
processing per interpretarlo (potenzialmente)
* se è vero che i layer convoluzionali successivi ricombinano feature visive estratte dal primo, avere più layer che
funzionano tramite lo stesso principio ('immagini in input; immagini in output') potrebbe aiutare la rete a svolgere 
un qualche preprocessing necessario  in più passi (più layer) invece di uno solo. per fare un esempio, se fosse possibile 
sostituire tutti i layer convolutivi con i layer che immaginiamo, si otterrebbe una "ricetta" di image processing che permette
di rendere le immagini facilmente classificabili.

## il layer

se ```[]``` è uno slice di una immagine ed f è un filtro, il layer convolutivo tradizionale funziona così sulle greyscale:

```
[] -(f1)-> []
   -(f2)-> []
    ...   
   -(fn)-> []
```

dove il numero di mappe di attivazioni in output (n) è un iperparametro. di contro, in rgb:

```
[] -(f11)-> 
[] -(f12)->  [+]
[] -(f13)-> 
   -(f21)->
   -(f22)->  [+]
   -(f23)->
     ...
   -(fn1)->
   -(fn2)->  [+]
   -(fn3)->
```

qui si nota che:
* il numero 3 non è un iperparametro e deve essere uguale alla profondità dell'input. qui è 3 perchè siamo nel primo layer, ma al secondo layer
sarà n...
* n, a questo layer, è un iperparametro. fissa invece il numero di matrici di convoluzione che dovranno essere apprese al layer successivo per 
produrre ognuna delle mappe di attivazione.
* ognuna delle mappe di attivazione in output ```[+]``` dipende sì da una convoluzione indipendente del canale rgb, **ma ognuna di queste viene 
poi sommata alle altre**. questo causa la "mescolanza di feature" di cui si parlava all'inizio e la difficoltà ad interpretare le mappe
di attivazione come immagini ai layer successivi al primo. infatti n è sempre scelto per ottimizzare la performance e non l'interpretabilità.

banalmente, facciamo uso di layer convoluzionali tradizionali con numero di filtri n=3, assieme a qualche altro accorgimento.

il nostro layer su immagine greyscale, primo livello:

```
[] -(f1)-> [1]
   -(f2)-> [2]
    ...
   -(fn)-> [n]
```

secondo livello:

```
[1] -(g11)-> [111]
    -(g12)-> [112]
    -(g13)-> [113]
[1] -(g21)-> [121]
    -(g22)-> [122]
    -(g23)-> [123]
      ...
[1] -(gm1)-> [1m1]
    -(gm2)-> [1m2]
    -(gm3)-> [1m3]

[2] -(g11)-> [211]
    -(g12)-> [212]
    -(g13)-> [213]
[2] -(g21)-> [221]
    -(g22)-> [222]
    -(g23)-> [223]
      ...
[2] -(gm1)-> [2m1]
    -(gm2)-> [2m2]
    -(gm3)-> [2m3]
...

[n] -(g11)-> [n11]
    -(g12)-> [n12]
    -(g13)-> [n13]
[n] -(g21)-> [n21]
    -(g22)-> [n22]
    -(g23)-> [n23]
      ...
[n] -(gm1)-> [nm1]
    -(gm2)-> [nm2]
    -(gm3)-> [nm3]
```

dove 3 ed m sono iperparametri. il numero di mappe di attivazione/immagini in uscita sarà ```m*n*3```.

## osservazioni
* si può immaginare m come il numero di livelli convolutivi tradizionali (con input=1 e output=3) utilizzabili in una implementazione ingenua
* in questo momento il codice non è parallelizzato

