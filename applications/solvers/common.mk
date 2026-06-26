# --------------------------------------------------------------------------- #
#                                                                             #
#  HermiteLBM: CUDA-based moment representation Lattice Boltzmann Method      #
#  Developed at UDESC - State University of Santa Catarina                    #
#  Website: https://www.udesc.br                                              #
#  Github: https://github.com/Geoenergia-Lab/HermiteLBM                       #
#                                                                             #
# --------------------------------------------------------------------------- #

default: clean $(EXECUTABLE)

$(EXECUTABLE):
	$(NVCXX) $(NVCXX_FLAGS) -Xptxas -v $(SOURCE) -o $@

install: clean uninstall $(EXECUTABLE)
	@ mkdir -p $(HERMITELBM_BIN_DIR)
	@ cp $(EXECUTABLE) $(HERMITELBM_BIN_DIR)/
	@ rm -f $(EXECUTABLE)

clean:
	@ rm -f $(EXECUTABLE)

uninstall:
	@ rm -f $(HERMITELBM_BIN_DIR)/$(EXECUTABLE)

# --------------------------------------------------------------------------- #