# LU Dekompozicija s Višenitnim Programiranjem

## Opis

Ovaj program implementira LU dekompoziciju matrice koristeći Doolittle algoritam u dvije varijante:
1. **Jednonitna implementacija** - sekvencijalno izvršavanje
2. **Višenitna implementacija** - paralelno izvršavanje s `std::thread`

Program testira performanse obaju pristupa na različitim veličinama matrica i prikazuje ubrzanje.

## LU Dekompozicija

LU dekompozicija razlaže matricu **A** na umnožak dviju matrica:
- **L** (Lower) - donje trokutasta matrica s jedinicama na dijagonali
- **U** (Upper) - gornje trokutasta matrica

Takda da: **A = L × U**

### Doolittle Algoritam

Program koristi Doolittle algoritam:
```
Za i = 0 do n-1:
  1. Računaj elementi U[i][k] za k ≥ i (gornji red)
     U[i][k] = A[i][k] - Σ(L[i][j] × U[j][k]) za j < i
     
  2. Računaj elementi L[k][i] za k > i (donji stupac)  
     L[k][i] = (A[k][i] - Σ(L[k][j] × U[j][i])) / U[i][i] za j < i
```

## Paralelizacija

### Strategija
- **Sekvencijalni dio**: Računanje pivot elementa U[i][i]
- **Paralelni dio**: Podijela ostatka reda/stupca među nitima
- **Sinkronizacija**: Čekanje završetka svih niti prije prelaska na sljedeći korak

### Implementacija
```cpp
// Za svaki red i
for(int i = 0; i < n; i++) {
    // 1. Sekvencijalno: računaj U[i][i]
    U[i][i] = A[i][i] - suma;
    
    // 2. Paralelno: podijeli ostatak reda među nitima
    vector<thread> threads;
    for(int t = 0; t < numThreads; t++) {
        // Svaka nit računa svoj dio reda
        threads.emplace_back(computeURow, ...);
    }
    
    // 3. Čekaj sve niti
    for(auto& t : threads) t.join();
    
    // 4. Isto za stupac L matrice...
}
```

## Kompajliranje i Pokretanje

### Osnovni način:
```bash
make           # Kompajlira program
./lu_decomposition  # Pokreće test
```

### Čišćenje:
```bash
make clean     # Briše izvršne datoteke
```

## Rezultati Performansi

Na 8-jezgarnom sustavu (stvarni rezultati):

| Veličina  | Jedna nit (ms) | Više niti (ms) | Ubrzanje |
|-----------|----------------|-----------------|----------|
| 100×100   | 0              | 17              | 0.00     |
| 500×500   | 44             | 66              | 0.67     |  
| 750×750   | 124            | 110             | **1.13** |
| 1200×1200 | 545            | 266             | **2.05** |

### Analiza

**Zašto nema ubrzanja za manje matrice?**
1. **Overhead stvaranja niti** - košta vremena
2. **Nedovoljno posla po niti** - niti završavaju prebrzo
3. **Cache efekata** - manje matrice stanu u cache, pa su već brze

**Kada paralelizacija postaje korisna?**
- Od ~750×750 matrica počinje pokazivati korist (1.13× ubrzanje)
- 1200×1200 matrica pokazuje odličo skaliranje (**2.05× ubrzanje**)
- Kada je posao po niti veći od overhead-a
- Za velike matrice (1200+) ubrzanje može doseći **2× ili više** što je odličo za 8 jezgri!

## Algoritamska Složenost

- **Vremenska složenost**: O(n³) za obje verzije
- **Prostorna složenost**: O(n²) za matrice L, U, A
- **Paralelno ubrzanje**: Teorijski maksimalno ~2× zbog sekvencijalnih dijelova

## Struktura Koda

```
LUDecomposition klasa:
├── generateRandomMatrix()     - Generira test matrice
├── singleThreadedLU()        - Jednonitna dekompozicija
├── multiThreadedLU()         - Višenitna dekompozicija  
├── computeURow()             - Paralelna funkcija za U red
├── computeLColumn()          - Paralelna funkcija za L stupac
├── verifyDecomposition()     - Provjera točnosti (L×U = A)
└── printMatrices()           - Ispis matrica
```

## Numerička Stabilnost

Program koristi toleranciju od 1×10⁻⁸ za provjeru točnosti zbog:
- Grešaka zaokruživanja u floating-point aritmetici
- Akumulacije grešaka kroz iteracije
- Tipične greške su reda ~10⁻¹⁵ što je odlično

## Mogućnosti Poboljšanja

1. **Block LU algoritam** - bolje cache performanse
2. **Pivoting** - bolja numerička stabilnost  
3. **SIMD instrukcije** - vektorska paralelizacija
4. **OpenMP** - lakša paralelizacija (alternativna implementacija uključena)
5. **GPU ubrzanje** - CUDA/OpenCL za masivni paralelizam 