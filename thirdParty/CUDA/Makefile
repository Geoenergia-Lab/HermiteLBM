# --------------------------------------------------------------------------- #
#                                                                             #
#  HermiteLBM: CUDA-based moment representation Lattice Boltzmann Method      #
#  Developed at UDESC - State University of Santa Catarina                    #
#  Website: https://www.udesc.br                                              #
#  Github: https://github.com/Geoenergia-Lab/HermiteLBM                       #
#                                                                             #
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
#  CUDA Toolkit Installer for Debian‑based distributions
# --------------------------------------------------------------------------- #
#  Usage:
#    (1) Edit bashrc and set HERMITELBM_DISTRO, HERMITELBM_CUDA_VERSION_MAJOR,
#        and HERMITELBM_CUDA_VERSION_MINOR.
#    (2) source bashrc
#    (3) make install
#
#  Or override on command line:
#    make install DISTRO=ubuntu2404 CUDA_VERSION=13-0
# ============================================================

# Read from environment (set by bashrc)
DISTRO ?= $(HERMITELBM_DISTRO)
MAJOR ?= $(HERMITELBM_CUDA_VERSION_MAJOR)
MINOR ?= $(HERMITELBM_CUDA_VERSION_MINOR)

# Construct CUDA version string for package name (major-minor)
ifneq ($(MAJOR),)
    ifneq ($(MINOR),)
        CUDA_VERSION = $(MAJOR)-$(MINOR)
    endif
endif

# Fallback defaults if environment variables are missing
ifeq ($(DISTRO),)
    DISTRO = ubuntu2404
endif
ifeq ($(CUDA_VERSION),)
    CUDA_VERSION = 13-0
endif

ARCH ?= x86_64
CUDA_REPO_BASE = https://developer.download.nvidia.com/compute/cuda/repos
KEYRING_DEB = cuda-keyring_1.1-1_all.deb

.PHONY: help install clean list-distros

help:
	@echo "CUDA Toolkit Installer for Debian‑based distributions"
	@echo ""
	@echo "Usage:"
	@echo "  1. Set HERMITELBM_DISTRO, HERMITELBM_CUDA_VERSION_MAJOR, and"
	@echo "     HERMITELBM_CUDA_VERSION_MINOR in bashrc"
	@echo "  2. source bashrc"
	@echo "  3. make install"
	@echo ""
	@echo "Or override on command line:"
	@echo "  make install DISTRO=<code> CUDA_VERSION=<major-minor>"
	@echo ""
	@echo "Other targets:"
	@echo "  make list-distros       # Show supported Debian‑based codes"
	@echo "  make clean              # Remove downloaded .deb files"
	@echo ""
	@echo "Current settings:"
	@echo "  DISTRO       = $(DISTRO)"
	@echo "  CUDA_VERSION = $(CUDA_VERSION)"

list-distros:
	@echo "Supported Debian‑based distribution codes:"
	@echo "  ubuntu2004, ubuntu2204, ubuntu2404, ubuntu2604"
	@echo "  debian10, debian11, debian12, debian13"
	@echo "  wsl-ubuntu"

install: check-distro check-apt
	@echo "==> Installing CUDA $(CUDA_VERSION) for distribution '$(DISTRO)'"
	wget -q --show-progress $(CUDA_REPO_BASE)/$(DISTRO)/$(ARCH)/$(KEYRING_DEB)
	sudo dpkg -i $(KEYRING_DEB)
	sudo apt update
	sudo apt install -y cuda-toolkit-$(CUDA_VERSION)
	rm -f $(KEYRING_DEB)
	@echo ""
	@echo "==> CUDA $(CUDA_VERSION) installation complete."
	@echo ""
	@echo "==> To activate the CUDA environment, run:"
	@echo "    source bashrc"
	@echo "or start a new shell session."

check-distro:
	@if [ -z "$(DISTRO)" ]; then \
		echo "Error: DISTRO is not set."; \
		echo "Please set HERMITELBM_DISTRO in bashrc or pass DISTRO=<code> on command line."; \
		echo "Run 'make list-distros' to see supported codes."; \
		exit 1; \
	fi

check-apt:
	@command -v apt >/dev/null 2>&1 || { \
		echo "Error: This installer requires apt (Debian/Ubuntu)."; \
		exit 1; \
	}
	@command -v wget >/dev/null 2>&1 || { \
		echo "Error: wget is required but not installed."; \
		echo "Install it with: sudo apt install wget"; \
		exit 1; \
	}
	@command -v dpkg >/dev/null 2>&1 || { \
		echo "Error: dpkg is required (should be present on Debian/Ubuntu)."; \
		exit 1; \
	}

clean:
	rm -f $(KEYRING_DEB)