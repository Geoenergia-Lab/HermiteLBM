/*---------------------------------------------------------------------------*\
|                                                                             |
| HermiteLBM: CUDA-based moment representation Lattice Boltzmann Method       |
| Developed at UDESC - State University of Santa Catarina                     |
| Website: https://www.udesc.br                                               |
| Github: https://github.com/Geoenergia-Lab/HermiteLBM                        |
|                                                                             |
\*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*\

Copyright (C) 2023 UDESC Geoenergia Lab
Authors: Nathan Duggins (Geoenergia Lab, UDESC)

This implementation is derived from concepts and algorithms developed in:
  MR-LBM: Moment Representation Lattice Boltzmann Method
  Copyright (C) 2021 CERNN
  Developed at Universidade Federal do Paraná (UFPR)
  Original authors: V. M. de Oliveira, M. A. de Souza, R. F. de Souza
  GitHub: https://github.com/CERNN/MR-LBM
  Licensed under GNU General Public License version 2

License
    This file is part of HermiteLBM.

    HermiteLBM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

Description
    Functions used to handle errors

Namespace
    LBM

SourceFiles
    signalHandler.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_SIGNALHANDLER_CUH
#define __MBLBM_SIGNALHANDLER_CUH

namespace LBM
{
    /**
     * @brief RAII manager for installing and restoring a SIGINT handler.
     *
     * Installs a custom signal handler for SIGINT upon construction and
     * automatically restores the previous handler upon destruction.
     * Copy and move operations are deleted to prevent multiple restorations.
     **/
    class signalHandler
    {
    public:
        /**
         * @brief Installs the SIGINT handler and saves the previous action.
         *
         * @throws std::runtime_error if sigaction fails.
         **/
        signalHandler()
            : old_action_{} // zero-initialize the struct
        {
            struct sigaction sa{};
            sa.sa_handler = &signalHandler::handleSignal;
            sigemptyset(&sa.sa_mask);
            sa.sa_flags = 0; // no SA_RESTART

            if (sigaction(SIGINT, &sa, &old_action_) != 0)
            {
                throw std::runtime_error("Failed to install SIGINT handler");
            }
        }

        /**
         * @brief Restores the previous SIGINT handler.
         **/
        ~signalHandler()
        {
            sigaction(SIGINT, &old_action_, nullptr);
        }

        // Prevent copying and moving to avoid double restoration.
        signalHandler(const signalHandler &) = delete;
        signalHandler &operator=(const signalHandler &) = delete;

    private:
        /**
         * @brief Static signal handler invoked on SIGINT.
         *
         * @param signal The signal number (unused).
         *
         * @note The function is not fully async‑signal‑safe: it uses std::cout
         *       and std::atomic::store, which are not guaranteed to be safe in
         *       signal context. For production use, prefer write() and
         *       volatile sig_atomic_t.
         **/
        static void handleSignal([[maybe_unused]] int signal)
        {
            std::cout << "Abort signal received" << std::endl;
            runTime::program_status.store(runTime::programStatus::BAD, std::memory_order_relaxed);
        }

        struct sigaction old_action_; ///< Previous SIGINT action saved for restoration.
    };
}

#endif