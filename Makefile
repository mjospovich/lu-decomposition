CXX = g++
CXXFLAGS = -std=c++20 -O3 -pthread -Wall -Wextra
TARGET = lu_decomposition
SOURCE = LU_dekompozicija.cpp

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCE)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: clean run 