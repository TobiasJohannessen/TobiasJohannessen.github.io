# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -I/usr/include

# SFML libraries
LDFLAGS = -lsfml-graphics -lsfml-window -lsfml-system

# Executable name
TARGET = sfml-app

# Source files
SRC = main.cpp

# Build rule
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

# Clean rule
clean:
	rm -f $(TARGET)
