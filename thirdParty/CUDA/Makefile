#*---------------------------------------------------------------------------*#
#                                                                             #
#  cudaLBM: CUDA-based moment representation Lattice Boltzmann Method         #
#  Developed at UDESC - State University of Santa Catarina                    #
#  Website: https://www.udesc.br                                              #
#  Github: https://github.com/geoenergiaUDESC/cudaLBM                         #
#                                                                             #
#*---------------------------------------------------------------------------*#

# --------------------------------------------------------------------------- #
#  CUDA Toolkit Installer for Debian‑based distributions
# --------------------------------------------------------------------------- #
#  Usage:
#    make install DISTRO=<code> [CUDA_VERSION=<major-minor>]
#    make clean
#
#  Example:
#    make install DISTRO=ubuntu2404 CUDA_VERSION=13-0
#    make install DISTRO=debian12 CUDA_VERSION=12-6
#    make install DISTRO=wsl-ubuntu
# ============================================================

# Required: distribution repository code (e.g., ubuntu2404, debian12, wsl-ubuntu)
DISTRO ?= 

# CUDA version in major-minor format (default 13.0)
CUDA_VERSION ?= 13-0

# Architecture (adjust if not x86_64)
ARCH ?= x86_64

# Repository base URL and keyring package name
CUDA_REPO_BASE = https://developer.download.nvidia.com/compute/cuda/repos
KEYRING_DEB = cuda-keyring_1.1-1_all.deb

.PHONY: help install clean list-distros

help:
	@echo "CUDA Toolkit Installer for Debian‑based distributions"
	@echo ""
	@echo "Usage:"
	@echo "  make install DISTRO=<code> [CUDA_VERSION=<major-minor>]"
	@echo "  make list-distros       # Show supported Debian‑based codes"
	@echo "  make clean              # Remove downloaded .deb files"
	@echo ""
	@echo "Examples:"
	@echo "  make install DISTRO=ubuntu2404 CUDA_VERSION=13-0"
	@echo "  make install DISTRO=debian12"
	@echo "  make install DISTRO=wsl-ubuntu CUDA_VERSION=12-6"
	@echo ""
	@echo "After installation, add the following to your shell profile:"
	@echo "  export PATH=/usr/local/cuda-\$$(echo $(CUDA_VERSION) | tr - .)/bin:\$$PATH"
	@echo "  export LD_LIBRARY_PATH=/usr/local/cuda-\$$(echo $(CUDA_VERSION) | tr - .)/lib64:\$$LD_LIBRARY_PATH"

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
	@cuda_path="/usr/local/cuda-$$(echo $(CUDA_VERSION) | tr '-' '.')"; \
	echo "Add the following to your shell profile (~/.bashrc, ~/.zshrc, etc.):"; \
	echo "  export PATH=$$cuda_path/bin:\$$PATH"; \
	echo "  export LD_LIBRARY_PATH=$$cuda_path/lib64:\$$LD_LIBRARY_PATH"

check-distro:
	@if [ -z "$(DISTRO)" ]; then \
		echo "Error: DISTRO is required."; \
		echo "Run 'make list-distros' to see supported codes."; \
		echo "Usage: make install DISTRO=<code>"; \
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