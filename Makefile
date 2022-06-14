compile: *.cpp *.h
	/usr/bin/g++  -g /home/jeremydohmann/projects/zeta_surgical/*.cpp -fopenmp -lpthread -mavx -o /home/jeremydohmann/projects/zeta_surgical/main.o `pkg-config --cflags --libs opencv4`

clean:
	rm *.o *.png

test: *.cpp *.h tests/*.cpp clean
	/usr/bin/g++  -g /home/jeremydohmann/projects/zeta_surgical/image_utils.cpp /home/jeremydohmann/projects/zeta_surgical/tests/*.cpp  -fopenmp -lpthread -mavx -o /home/jeremydohmann/projects/zeta_surgical/test.o `pkg-config --cflags --libs opencv4`
	./test.o
	make clean