all: neuralnetwork.a

neuralnetwork.o: src/neuralnetwork.cpp inc/neuralnetwork.hpp
	g++ -std=c++11 -c src/neuralnetwork.cpp -o obj/neuralnetwork.o -I./inc -I/usr/local/inc

activation.o: src/activation.cpp inc/activation.hpp
	g++ -std=c++11 -c src/activation.cpp -o obj/activation.o -I./inc

neuralnetwork.a: neuralnetwork.o activation.o
	ar rcs lib/neuralnetwork.a obj/neuralnetwork.o obj/activation.o

clean:
	clean rm -f obj/*.o lib/*.a
