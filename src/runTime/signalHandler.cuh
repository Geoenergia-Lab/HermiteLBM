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

#ifdef _WIN32
#include <windows.h>
#endif

namespace LBM
{
    class signalHandler
    {
    public:
        signalHandler()
        {
#ifdef _WIN32
            // Register a console control handler for Ctrl+C and Ctrl+Break.
            if (!SetConsoleCtrlHandler(&signalHandler::handleSignal, TRUE))
            {
                throw std::runtime_error("Failed to install console control handler");
            }
#else
            struct sigaction sa{};
            sa.sa_handler = &signalHandler::handleSignal;
            sigemptyset(&sa.sa_mask);
            sa.sa_flags = 0; // no SA_RESTART

            if (sigaction(SIGINT, &sa, &old_action_) != 0)
            {
                throw std::runtime_error("Failed to install SIGINT handler");
            }
#endif
        }

        ~signalHandler()
        {
#ifdef _WIN32
            // Remove our handler from the console control handler chain.
            SetConsoleCtrlHandler(&signalHandler::handleSignal, FALSE);
#else
            sigaction(SIGINT, &old_action_, nullptr);
#endif
        }

        // Prevent copying and moving.
        signalHandler(const signalHandler &) = delete;
        signalHandler &operator=(const signalHandler &) = delete;

    private:
        /**
         * @brief Common implementation of storing the program status for Linux and Windows systems
         **/
        static void handleSignalImpl([[maybe_unused]] int signal)
        {
            std::cout << "Abort signal received" << std::endl;
            runTime::program_status.store(runTime::programStatus::BAD, std::memory_order_relaxed);
        }

#ifdef _WIN32
        /**
         * @brief Windows console control handler.
         *
         * @param dwCtrlType The type of control event.
         * @return TRUE if the event was handled, FALSE to pass to the next handler.
         **/
        __host__ [[nodiscard]] static BOOL WINAPI handleSignal(const DWORD dwCtrlType)
        {
            if (dwCtrlType == CTRL_C_EVENT || dwCtrlType == CTRL_BREAK_EVENT)
            {
                handleSignalImpl(0);
                // Returning TRUE prevents the default handler (which would terminate
                // the process). Remove this line if you want default termination.
                return TRUE;
            }
            // For other events (CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT)
            // pass them on.
            return FALSE;
        }
#else
        /**
         * @brief POSIX signal handler
         **/
        static void handleSignal([[maybe_unused]] int signal)
        {
            handleSignalImpl(signal);
        }

        /**
         * @brief Only used on POSIX
         **/
        struct sigaction old_action_;
#endif
    };
}

#endif