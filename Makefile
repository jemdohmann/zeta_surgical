compile: *.cpp *.h
	/usr/bin/g++  -g /home/jeremydohmann/projects/zeta_surgical/*.cpp -fopenmp -lpthread -mavx -o /home/jeremydohmann/projects/zeta_surgical/test.o `pkg-config --cflags --libs opencv4`

clean:
	rm *.o *.png