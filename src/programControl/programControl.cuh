/*---------------------------------------------------------------------------*\
|                                                                             |
| cudaLBM: CUDA-based moment representation Lattice Boltzmann Method          |
| Developed at UDESC - State University of Santa Catarina                     |
| Website: https://www.udesc.br                                               |
| Github: https://github.com/geoenergiaUDESC/cudaLBM                          |
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
    This file is part of cudaLBM.

    cudaLBM is free software: you can redistribute it and/or modify it
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
    A class handling the setup of the solver

Namespace
    LBM

SourceFiles
    programControl.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_PROGRAMCONTROL_CUH
#define __MBLBM_PROGRAMCONTROL_CUH

#include "../LBMIncludes.cuh"
#include "../LBMTypedefs.cuh"
#include "../strings.cuh"
#include "../inputControl.cuh"
#include "../fileIO/fileIO.cuh"

namespace LBM
{
    class programControl
    {
    public:
        /**
         * @brief Constructor for the programControl class
         * @param argc First argument passed to main
         * @param argv Second argument passed to main
         **/
        __host__ [[nodiscard]] programControl(const int argc, const char *const argv[]) noexcept
            : input_(inputControl(argc, argv))
        {
            static_assert((std::is_same_v<scalar_t, float>) | (std::is_same_v<scalar_t, double>), "Invalid floating point size: must be either 32 or 64 bit");

            static_assert((std::is_same_v<label_t, uint32_t>) | (std::is_same_v<label_t, uint64_t>), "Invalid label size: must be either 32 bit unsigned or 64 bit unsigned");

            const auto file = string::readFile("programControl");

            caseName_ = string::extractParameter<std::string>(file, "caseName");
            multiphase_ = string::extractParameter<bool>(file, "multiphase");

            const bool hasRe = string::hasParameter(file, "Re");
            const bool hasReA = string::hasParameter(file, "ReA");
            const bool hasReB = string::hasParameter(file, "ReB");

            const bool hasNuA = string::hasParameter(file, "nuA");
            const bool hasNuB = string::hasParameter(file, "nuB");

            if (!multiphase_)
            {
                if (!hasRe)
                {
                    errorHandler(-1, "Single-phase simulation requires 'Re'.");
                }

                Re_ = string::extractParameter<scalar_t>(file, "Re");
                ReA_ = static_cast<scalar_t>(0);
                ReB_ = static_cast<scalar_t>(0);
                nuA_ = static_cast<scalar_t>(0);
                nuB_ = static_cast<scalar_t>(0);
                We_ = static_cast<scalar_t>(0);
                interfaceWidth_ = static_cast<scalar_t>(0);
            }
            else
            {
                const bool usingRe = hasReA && hasReB;
                const bool usingNu = hasNuA && hasNuB;

                if (usingRe && usingNu)
                {
                    errorHandler(-1, "Specify either (ReA/ReB) or (nuA/nuB), not both.");
                }

                if (!usingRe && !usingNu)
                {
                    errorHandler(-1, "Multiphase requires either (ReA/ReB) or (nuA/nuB).");
                }

                if (usingRe)
                {
                    ReA_ = string::extractParameter<scalar_t>(file, "ReA");
                    ReB_ = string::extractParameter<scalar_t>(file, "ReB");
                    nuA_ = static_cast<scalar_t>(0);
                    nuB_ = static_cast<scalar_t>(0);
                }
                else
                {
                    nuA_ = string::extractParameter<scalar_t>(file, "nuA");
                    nuB_ = string::extractParameter<scalar_t>(file, "nuB");
                    ReA_ = static_cast<scalar_t>(0);
                    ReB_ = static_cast<scalar_t>(0);
                }

                Re_ = static_cast<scalar_t>(0);
                We_ = string::extractParameter<scalar_t>(file, "We");
                interfaceWidth_ = string::extractParameter<scalar_t>(file, "interfaceWidth");
            }

            u_inf_ = string::extractParameter<scalar_t>(file, "u_inf");
            L_char_ = string::extractParameter<scalar_t>(file, "L_char");

            nTimeSteps_ = string::extractParameter<label_t>(file, "nTimeSteps");
            saveInterval_ = string::extractParameter<label_t>(file, "saveInterval");
            infoInterval_ = string::extractParameter<label_t>(file, "infoInterval");

            latestTime_ = fileIO::latestTime(caseName_);

            // Get the launch time
            const time_t time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

            // Get current working directory
            const std::filesystem::path launchDirectory = std::filesystem::current_path();

            std::cout << "/*---------------------------------------------------------------------------*\\" << std::endl;
            std::cout << "|                                                                             |" << std::endl;
            std::cout << "| cudaLBM: CUDA-based moment representation Lattice Boltzmann Method          |" << std::endl;
            std::cout << "| Developed at UDESC - State University of Santa Catarina                     |" << std::endl;
            std::cout << "| Website: https://www.udesc.br                                               |" << std::endl;
            std::cout << "| Github: https://github.com/geoenergiaUDESC/cudaLBM                          |" << std::endl;
            std::cout << "|                                                                             |" << std::endl;
            std::cout << "\\*---------------------------------------------------------------------------*/" << std::endl;
            std::cout << std::endl;
            std::cout << "programControl:" << std::endl;
            std::cout << "{" << std::endl;
            std::cout << "    programName: " << input_.commandLine()[0] << ";" << std::endl;
            std::cout << "    launchTime: " << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S") << ";" << std::endl;
            std::cout << "    launchDirectory: " << launchDirectory.string() << ";" << std::endl;
            std::cout << "    deviceList: [";
            if (deviceList().size() > 1)
            {
                for (label_t i = 0; i < deviceList().size() - 1; i++)
                {
                    std::cout << deviceList()[i] << ", ";
                }
            }
            std::cout << deviceList()[deviceList().size() - 1] << "];" << std::endl;
            std::cout << "    caseName: " << caseName_ << ";" << std::endl;
            if (!multiphase_)
            {
                std::cout << "    Re = " << Re_ << ";" << std::endl;
            }
            else
            {
                std::cout << "    ReA = " << ReA_ << ";" << std::endl;
                std::cout << "    ReB = " << ReB_ << ";" << std::endl;
                std::cout << "    We = " << We_ << ";" << std::endl;
                std::cout << "    interfaceWidth = " << interfaceWidth_ << ";" << std::endl;
            }
            std::cout << "    nTimeSteps = " << nTimeSteps_ << ";" << std::endl;
            std::cout << "    saveInterval = " << saveInterval_ << ";" << std::endl;
            std::cout << "    infoInterval = " << infoInterval_ << ";" << std::endl;
            std::cout << "    latestTime = " << latestTime_ << ";" << std::endl;
            std::cout << "    scalarType: " << ((sizeof(scalar_t) == 4) ? "32 bit" : "64 bit") << ";" << std::endl;
            std::cout << "    labelType: " << ((sizeof(label_t) == 4) ? "uint32_t" : "uint64_t") << ";" << std::endl;
            std::cout << "};" << std::endl;
            std::cout << std::endl;

            cudaDeviceSynchronize();
        };

        /**
         * @brief Destructor for the programControl class
         **/
        ~programControl() noexcept {};

        /**
         * @brief Returns the name of the case
         * @return A const std::string
         **/
        __host__ [[nodiscard]] inline constexpr const std::string &caseName() const noexcept
        {
            return caseName_;
        }

        /**
         * @brief Returns multiphase or not
         * @return Multiphase bool
         **/
        __device__ __host__ [[nodiscard]] inline constexpr bool isMultiphase() const noexcept
        {
            return multiphase_;
        }

        /**
         * @brief Returns the array of device indices
         * @return A read-only reference to deviceList_ contained within input_
         **/
        __host__ [[nodiscard]] inline constexpr const std::vector<deviceIndex_t> &deviceList() const noexcept
        {
            return input_.deviceList();
        }

        /**
         * @brief Returns the Reynolds number
         * @return The Reynolds number
         **/
        __device__ __host__ [[nodiscard]] inline constexpr scalar_t Re() const noexcept
        {
            return Re_;
        }

        /**
         * @brief Returns the fluid A Reynolds number
         * @return The Reynolds number for fluid A
         **/
        __device__ __host__ [[nodiscard]] inline constexpr scalar_t ReA() const noexcept
        {
            return ReA_;
        }

        /**
         * @brief Returns the fluid B Reynolds number
         * @return The Reynolds number for fluid B
         **/
        __device__ __host__ [[nodiscard]] inline constexpr scalar_t ReB() const noexcept
        {
            return ReB_;
        }

        /**
         * @brief Returns the fluid A kinematic viscosity
         * @return The kinematic viscosity for fluid A
         **/
        __device__ __host__ [[nodiscard]] inline constexpr scalar_t nuA() const noexcept
        {
            return nuA_;
        }

        /**
         * @brief Returns the fluid B kinematic viscosity
         * @return The kinematic viscosity for fluid B
         **/
        __device__ __host__ [[nodiscard]] inline constexpr scalar_t nuB() const noexcept
        {
            return nuB_;
        }

        /**
         * @brief Returns the Weber number
         * @return The Weber number
         **/
        __device__ __host__ [[nodiscard]] inline constexpr scalar_t We() const noexcept
        {
            return We_;
        }

        /**
         * @brief Returns the interface width
         * @return The interface width
         **/
        __device__ __host__ [[nodiscard]] inline constexpr scalar_t interfaceWidth() const noexcept
        {
            return interfaceWidth_;
        }

        /**
         * @brief Returns the characteristic velocity
         * @return The characteristic velocity
         **/
        __device__ __host__ [[nodiscard]] inline constexpr scalar_t u_inf() const noexcept
        {
            return u_inf_;
        }

        /**
         * @brief Returns the characteristic length
         * @return The characteristic length
         **/
        __device__ __host__ [[nodiscard]] inline constexpr scalar_t L_char() const noexcept
        {
            return L_char_;
        }

        /**
         * @brief Returns the total number of simulation time steps
         * @return The total number of simulation time steps
         **/
        __device__ __host__ [[nodiscard]] inline constexpr label_t nt() const noexcept
        {
            return nTimeSteps_;
        }

        /**
         * @brief Decide whether or not the program should perform a checkpoint
         * @return True if the program should checkpoint, false otherwise
         **/
        __device__ __host__ [[nodiscard]] inline constexpr bool save(const label_t timeStep) const noexcept
        {
            return (timeStep % saveInterval_) == 0;
        }

        /**
         * @brief Decide whether or not the program should perform a checkpoint
         * @return True if the program should checkpoint, false otherwise
         **/
        __device__ __host__ [[nodiscard]] inline constexpr bool print(const label_t timeStep) const noexcept
        {
            return (timeStep % infoInterval_) == 0;
        }

        /**
         * @brief Returns the latest time step of the solution files contained within the current directory
         * @return The latest time step as a label_t
         **/
        __device__ __host__ [[nodiscard]] inline constexpr label_t latestTime() const noexcept
        {
            return latestTime_;
        }

        /**
         * @brief Provides read-only access to the input control
         * @return A const reference to an inputControl object
         **/
        __host__ [[nodiscard]] inline constexpr const inputControl &input() const noexcept
        {
            return input_;
        }

        /**
         * @brief Veriefies if the command line has the argument -type
         * @return A string representing the convertion type passed at the command line
         * @param[in] programCtrl Program control parameters
         **/
        __host__ [[nodiscard]] const std::string getArgument(const std::string &argument) const
        {
            if (input_.isArgPresent(argument))
            {
                for (label_t arg = 0; arg < commandLine().size(); arg++)
                {
                    if (commandLine()[arg] == argument)
                    {
                        if (arg + 1 == commandLine().size())
                        {
                            throw std::runtime_error("Argument " + argument + " not specified: the correct syntax is " + argument + " Arg");
                        }
                        else
                        {
                            return commandLine()[arg + 1];
                        }
                    }
                }
            }

            throw std::runtime_error("Argument " + argument + " not specified: the correct syntax is " + argument + " Arg");
        }

        /**
         * @brief Provides read-only access to the arguments supplied at the command line
         * @return The command line input as a vector of strings
         **/
        __host__ [[nodiscard]] inline constexpr const std::vector<std::string> &commandLine() const noexcept
        {
            return input_.commandLine();
        }

    private:
        /**
         * @brief A reference to the input control object
         **/
        const inputControl input_;

        /**
         * @brief The name of the simulation case
         **/
        std::string caseName_;

        /**
         * @brief Whether the simulation is multiphase
         **/
        bool multiphase_;

        /**
         * @brief The Reynolds number
         **/
        scalar_t Re_;

        /**
         * @brief The fluid A Reynolds number
         **/
        scalar_t ReA_;

        /**
         * @brief The fluid B Reynolds number
         **/
        scalar_t ReB_;

        /**
         * @brief The fluid A kinematic viscosity
         **/
        scalar_t nuA_;

        /**
         * @brief The fluid B kinematic viscosity
         **/
        scalar_t nuB_;

        /**
         * @brief The Weber number
         **/
        scalar_t We_;

        /**
         * @brief The interface width
         **/
        scalar_t interfaceWidth_;

        /**
         * @brief The characteristic velocity
         **/
        scalar_t u_inf_;

        /**
         * @brief The characteristic length scale
         **/
        scalar_t L_char_;

        /**
         * @brief Total number of simulation time steps, the save interval, info output interval and the latest time step at program start
         **/
        label_t nTimeSteps_;
        label_t saveInterval_;
        label_t infoInterval_;
        label_t latestTime_;

        /**
         * @brief Reads a variable from the caseInfo file into a parameter of type T
         * @return The variable as type T
         * @param varName The name of the variable to read
         **/
        template <typename T>
        __host__ [[nodiscard]] T initialiseConst(const std::string varName) const noexcept
        {
            return string::extractParameter<T>(string::readFile("programControl"), varName);
        }
    };
}

#include "streamHandler.cuh"

#endif