#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <random>
#include <iomanip>
#include <mutex>
#include <atomic>

using namespace std;
using namespace std::chrono;

class LUDecomposition {
private:
    vector<vector<double>> L, U, A;
    int n;
    
public:
    LUDecomposition(int size) : n(size) {
        L.resize(n, vector<double>(n, 0.0));
        U.resize(n, vector<double>(n, 0.0));
        A.resize(n, vector<double>(n, 0.0));
    }
    
    // Getter za matricu A
    vector<vector<double>>& getA() { return A; }
    const vector<vector<double>>& getA() const { return A; }
    
    // Generiranje random matrice za testiranje
    void generateRandomMatrix() {
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(1.0, 100.0);
        
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                A[i][j] = dis(gen);
            }
        }
    }
    
    // Kopiranje matrice A u radnu matricu
    void copyMatrix(vector<vector<double>>& dest, const vector<vector<double>>& src) {
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                dest[i][j] = src[i][j];
            }
        }
    }
    
    // Jednostavna (jednonačelna) LU dekompozicija - Doolittle algoritam
    void singleThreadedLU() {
        // Inicijaliziramo L i U
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                L[i][j] = (i == j) ? 1.0 : 0.0; // L je jedinična na dijagonali
                U[i][j] = 0.0;
            }
        }
        
        // Doolittle LU dekompozicija
        for(int i = 0; i < n; i++) {
            // Računamo U matricu za red i (gornji trokut)
            for(int k = i; k < n; k++) {
                double sum = 0.0;
                for(int j = 0; j < i; j++) {
                    sum += L[i][j] * U[j][k];
                }
                U[i][k] = A[i][k] - sum;
            }
            
            // Računamo L matricu za stupac i (donji trokut)
            for(int k = i + 1; k < n; k++) {
                double sum = 0.0;
                for(int j = 0; j < i; j++) {
                    sum += L[k][j] * U[j][i];
                }
                L[k][i] = (A[k][i] - sum) / U[i][i];
            }
        }
    }
    
    // Paralelizirana funkcija za računanje U reda
    void computeURow(int i, int startK, int endK) {
        for(int k = startK; k < endK; k++) {
            double sum = 0.0;
            for(int j = 0; j < i; j++) {
                sum += L[i][j] * U[j][k];
            }
            U[i][k] = A[i][k] - sum;
        }
    }
    
    // Paralelizirana funkcija za računanje L stupca
    void computeLColumn(int i, int startK, int endK) {
        for(int k = startK; k < endK; k++) {
            double sum = 0.0;
            for(int j = 0; j < i; j++) {
                sum += L[k][j] * U[j][i];
            }
            L[k][i] = (A[k][i] - sum) / U[i][i];
        }
    }
    
    // Višenitna LU dekompozicija - jednostavniji pristup
    void multiThreadedLU(int numThreads) {
        // Inicijaliziramo L i U
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                L[i][j] = (i == j) ? 1.0 : 0.0; // L je jedinična na dijagonali
                U[i][j] = 0.0;
            }
        }
        
        // Doolittle LU dekompozicija s paralelizacijom unutar svakog koraka
        for(int i = 0; i < n; i++) {
            // Računamo U[i][i] sekvencijalno (pivot element)
            double sum = 0.0;
            for(int j = 0; j < i; j++) {
                sum += L[i][j] * U[j][i];
            }
            U[i][i] = A[i][i] - sum;
            
            // Paralelno računanje ostatka i-tog reda U matrice
            if(i + 1 < n) {
                vector<thread> threads;
                int elementsPerThread = max(1, (n - i - 1) / numThreads);
                
                for(int t = 0; t < numThreads; t++) {
                    int startK = i + 1 + t * elementsPerThread;
                    int endK = (t == numThreads - 1) ? n : min(n, startK + elementsPerThread);
                    
                    if(startK < endK) {
                        threads.emplace_back([this, i, startK, endK]() {
                            computeURow(i, startK, endK);
                        });
                    }
                }
                
                for(auto& t : threads) {
                    t.join();
                }
            }
            
            // Paralelno računanje i-tog stupca L matrice
            if(i + 1 < n) {
                vector<thread> threads;
                int elementsPerThread = max(1, (n - i - 1) / numThreads);
                
                for(int t = 0; t < numThreads; t++) {
                    int startK = i + 1 + t * elementsPerThread;
                    int endK = (t == numThreads - 1) ? n : min(n, startK + elementsPerThread);
                    
                    if(startK < endK) {
                        threads.emplace_back([this, i, startK, endK]() {
                            computeLColumn(i, startK, endK);
                        });
                    }
                }
                
                for(auto& t : threads) {
                    t.join();
                }
            }
        }
    }
    
    // Provjera točnosti dekompozicije (L * U = A)
    bool verifyDecomposition() {
        const double EPSILON = 1e-8; // Povećana tolerancija za numeričke greške
        
        double maxError = 0.0;
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                double sum = 0.0;
                for(int k = 0; k < n; k++) {
                    sum += L[i][k] * U[k][j];
                }
                double error = abs(sum - A[i][j]);
                maxError = max(maxError, error);
                if(error > EPSILON) {
                    if(n <= 4) { // Samo za manje matrice
                        cout << "Greška na poziciji [" << i << "," << j << "]: ";
                        cout << "L*U = " << sum << ", A = " << A[i][j];
                        cout << ", razlika = " << error << endl;
                    }
                    return false;
                }
            }
        }
        
        if(n <= 4) {
            cout << "Maksimalna greška: " << scientific << maxError << fixed << endl;
        }
        return true;
    }
    
    // Ispis matrica (za manje matrice)
    void printMatrices() {
        if(n > 10) {
            cout << "Matrice su prevelike za ispis (n > 10)" << endl;
            return;
        }
        
        cout << fixed << setprecision(2);
        
        cout << "\nMatrica A:" << endl;
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                cout << setw(8) << A[i][j];
            }
            cout << endl;
        }
        
        cout << "\nMatrica L:" << endl;
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                cout << setw(8) << L[i][j];
            }
            cout << endl;
        }
        
        cout << "\nMatrica U:" << endl;
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                cout << setw(8) << U[i][j];
            }
            cout << endl;
        }
    }
};

int main() {
    cout << "=== LU DEKOMPOZICIJA S VIŠE NITI ===" << endl;
    
    // Dohvaćamo broj jezgri
    int numCores = thread::hardware_concurrency();
    cout << "Broj dostupnih jezgri: " << numCores << endl;
    
    // Testiramo različite veličine matrica
    vector<int> matrixSizes = {100, 500, 750, 1200, 1500};
    
    cout << "\n" << setw(15) << "Veličina" 
         << setw(20) << "Jedna nit (ms)" 
         << setw(20) << "Više niti (ms)" 
         << setw(15) << "Ubrzanje" 
         << setw(15) << "Točno" << endl;
    cout << string(90, '-') << endl;
    
    for(int size : matrixSizes) {
        LUDecomposition lu(size);
        lu.generateRandomMatrix();
        
        // Kreiramo kopiju za multithreaded test da koristimo istu matricu
        LUDecomposition luMulti(size);
        luMulti.copyMatrix(luMulti.getA(), lu.getA());
        
        // Test jedne niti
        auto start = high_resolution_clock::now();
        lu.singleThreadedLU();
        auto end = high_resolution_clock::now();
        auto singleTime = duration_cast<milliseconds>(end - start).count();
        
        bool isCorrect1 = lu.verifyDecomposition();
        
        // Test više niti
        start = high_resolution_clock::now();
        luMulti.multiThreadedLU(numCores);
        end = high_resolution_clock::now();
        auto multiTime = duration_cast<milliseconds>(end - start).count();
        
        bool isCorrect2 = luMulti.verifyDecomposition();
        
        double speedup = (multiTime > 0) ? (double)singleTime / multiTime : 0.0;
        
        cout << setw(12) << size 
             << setw(20) << singleTime 
             << setw(20) << multiTime 
             << setw(15) << fixed << setprecision(2) << speedup 
             << setw(15) << (isCorrect1 && isCorrect2 ? "DA" : "NE") << endl;
    }

    return 0;
}
