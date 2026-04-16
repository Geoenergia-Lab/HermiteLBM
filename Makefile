# Top-level Makefile

# Check if required environment variables are set
ifeq ($(HERMITELBM_BUILD_DIR),)
$(error HERMITELBM_BUILD_DIR is not set. Please run "source bashrc" in the project directory first)
endif

ifeq ($(HERMITELBM_BIN_DIR),)
$(error HERMITELBM_BIN_DIR is not set. Please run "source bashrc" in the project directory first)
endif

ifeq ($(HERMITELBM_INCLUDE_DIR),)
$(error HERMITELBM_INCLUDE_DIR is not set. Please run "source bashrc" in the project directory first)
endif

ifeq ($(HERMITELBM_CUDA_DIR),)
$(error HERMITELBM_CUDA_DIR is not set. Please run "source bashrc" in the project directory first)
endif

TOOL_SUBDIRS = applications/computeVersion
GPU_SUBDIRS = applications/solvers/momentBasedD3Q19 applications/solvers/momentBasedD3Q27 applications/solvers/isothermalD3Q19 applications/solvers/isothermalD3Q27 applications/postProcessing/fieldConvert applications/postProcessing/fieldCalculate
SUBDIRS = $(TOOL_SUBDIRS) $(GPU_SUBDIRS)

.PHONY: all clean install uninstall $(SUBDIRS) directories

all: directories $(SUBDIRS)

# Create build directories
directories:
	@ mkdir -p $(HERMITELBM_BUILD_DIR)
	@ mkdir -p $(HERMITELBM_BIN_DIR)
	@ mkdir -p $(HERMITELBM_INCLUDE_DIR)

# Compile and run computeVersion to generate hardware.info
$(HERMITELBM_INCLUDE_DIR)/hardware.info: directories
	@ $(MAKE) -C applications/computeVersion install
	@ computeVersion

# Compile computeVersion (tool subdirectories)
$(TOOL_SUBDIRS): directories
	$(MAKE) -C $@

# Compile GPU applications only after hardware.info is generated
$(GPU_SUBDIRS): $(HERMITELBM_INCLUDE_DIR)/hardware.info
	$(MAKE) -C $@

# Clean all projects
clean:
	@ for dir in $(SUBDIRS); do $(MAKE) -C $$dir clean; done
	@ rm -rf $(HERMITELBM_BUILD_DIR)

# Install all projects
install: directories $(HERMITELBM_INCLUDE_DIR)/hardware.info
	@ for dir in $(SUBDIRS); do $(MAKE) -C $$dir install; done

# Uninstall all projects
uninstall:
	@ for dir in $(SUBDIRS); do $(MAKE) -C $$dir uninstall; done
	@ rm -rf $(HERMITELBM_BIN_DIR) $(HERMITELBM_INCLUDE_DIR)