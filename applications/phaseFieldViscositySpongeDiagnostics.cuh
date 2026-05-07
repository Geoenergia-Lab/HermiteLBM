// /*---------------------------------------------------------------------------*\
// |                                                                             |
// | cudaLBM: CUDA-based moment representation Lattice Boltzmann Method          |
// | Developed at UDESC - State University of Santa Catarina                     |
// | Website: https://www.udesc.br                                               |
// | Github: https://github.com/geoenergiaUDESC/cudaLBM                          |
// |                                                                             |
// \*---------------------------------------------------------------------------*/

// #ifndef __MBLBM_PHASEFIELDVISCOSITYSPONGEDIAGNOSTICS_CUH
// #define __MBLBM_PHASEFIELDVISCOSITYSPONGEDIAGNOSTICS_CUH

// #include "../src/LBMIncludes.cuh"
// #include "../src/latticeMesh/latticeMesh.cuh"
// #include "../src/momentBasedLBM/phaseFieldViscositySponge.cuh"
// #include "../src/programControl/programControl.cuh"

// namespace LBM
// {
//     namespace phaseFieldApplication
//     {
//         template <const bool ApplySponge>
//         __host__ void validateViscositySpongeSetup(const host::latticeMesh &mesh)
//         {
//             if constexpr (ApplySponge)
//             {
//                 if (mesh.dimension<axis::Y>() < static_cast<host::label_t>(8))
//                 {
//                     throw std::runtime_error("phase-field viscosity sponge requires ny >= 8.");
//                 }
//             }
//             else
//             {
//                 (void)mesh;
//             }
//         }

// #if defined(PHASE_FIELD_SPONGE_DIAGNOSTICS)
//         namespace viscositySpongeDiagnostics
//         {
//             __host__ [[nodiscard]] scalar_t tauA(const programControl &programCtrl) noexcept
//             {
//                 const scalar_t nu = programCtrl.dualCharacteristic()
//                                         ? (programCtrl.u_inf_A() * programCtrl.L_char_A() / programCtrl.Re_A())
//                                         : (programCtrl.u_inf() * programCtrl.L_char() / programCtrl.Re_A());
//                 return static_cast<scalar_t>(0.5) + static_cast<scalar_t>(3) * nu;
//             }

//             __host__ [[nodiscard]] scalar_t tauB(const programControl &programCtrl) noexcept
//             {
//                 const scalar_t nu = programCtrl.dualCharacteristic()
//                                         ? (programCtrl.u_inf_B() * programCtrl.L_char_B() / programCtrl.Re_B())
//                                         : (programCtrl.u_inf() * programCtrl.L_char() / programCtrl.Re_B());
//                 return static_cast<scalar_t>(0.5) + static_cast<scalar_t>(3) * nu;
//             }

//             __host__ [[nodiscard]] scalar_t rampYmax(const host::label_t ny, const host::label_t yGlobal) noexcept
//             {
//                 if (ny < static_cast<host::label_t>(8))
//                 {
//                     return static_cast<scalar_t>(0);
//                 }

//                 const host::label_t cells = ny / static_cast<host::label_t>(8);
//                 const host::label_t yMax = ny - static_cast<host::label_t>(1);
//                 const scalar_t yn = static_cast<scalar_t>(static_cast<double>(yGlobal) / static_cast<double>(yMax));
//                 const scalar_t yStart = static_cast<scalar_t>(static_cast<double>(yMax - cells) / static_cast<double>(yMax));
//                 const scalar_t invSponge = static_cast<scalar_t>(static_cast<double>(yMax) / static_cast<double>(cells));
//                 scalar_t s = (yn - yStart) * invSponge;
//                 s = (s < static_cast<scalar_t>(0)) ? static_cast<scalar_t>(0) : s;
//                 s = (s > static_cast<scalar_t>(1)) ? static_cast<scalar_t>(1) : s;
//                 return s * s * s * (s * (s * static_cast<scalar_t>(6) - static_cast<scalar_t>(15)) + static_cast<scalar_t>(10));
//             }

//             __host__ [[nodiscard]] scalar_t tauEffective(const scalar_t tauPhi, const scalar_t ramp) noexcept
//             {
//                 const scalar_t tauMax = static_cast<scalar_t>(0.5) + (tauPhi - static_cast<scalar_t>(0.5)) * (static_cast<scalar_t>(1) + phaseFieldSponge::K_gain());
//                 return tauPhi + ramp * (tauMax - tauPhi);
//             }

//             __host__ void printSample(const char *label, const scalar_t tauPhi, const scalar_t ramp)
//             {
//                 const scalar_t tau = tauEffective(tauPhi, ramp);
//                 const scalar_t omega = static_cast<scalar_t>(1) / tau;
//                 const scalar_t nu = (tau - static_cast<scalar_t>(0.5)) / static_cast<scalar_t>(3);
//                 std::cout << "    " << label << ": tau=" << tau << ", omega=" << omega << ", nu=" << nu << '\n';
//             }
//         }
// #endif

//         template <const bool ApplySponge>
//         __host__ void printViscositySpongeDiagnostics(
//             const host::latticeMesh &mesh,
//             const programControl &programCtrl)
//         {
// #if defined(PHASE_FIELD_SPONGE_DIAGNOSTICS)
//             const host::label_t ny = mesh.dimension<axis::Y>();
//             const host::label_t cells = ny / static_cast<host::label_t>(8);

//             std::cout << "phase-field viscosity sponge:" << '\n';
//             std::cout << "    physical-Y sponge enabled: " << (ApplySponge ? "yes" : "no") << '\n';
//             std::cout << "    sponge length: " << cells << " cells (ny / 8)" << '\n';

//             if constexpr (ApplySponge)
//             {
//                 const host::label_t yMax = ny - static_cast<host::label_t>(1);
//                 const host::label_t yOutside = ny - cells - static_cast<host::label_t>(1);
//                 const host::label_t yFirstInside = yOutside + static_cast<host::label_t>(1);
//                 const scalar_t outsideRamp = viscositySpongeDiagnostics::rampYmax(ny, yOutside);
//                 const scalar_t insideRamp = viscositySpongeDiagnostics::rampYmax(ny, yFirstInside);
//                 const scalar_t ymaxRamp = viscositySpongeDiagnostics::rampYmax(ny, yMax);
//                 const scalar_t tau0 = viscositySpongeDiagnostics::tauA(programCtrl);
//                 const scalar_t tau1 = viscositySpongeDiagnostics::tauB(programCtrl);

//                 std::cout << "    ramp outside sponge y=" << yOutside << ": " << outsideRamp << '\n';
//                 std::cout << "    ramp first sponge cell y=" << yFirstInside << ": " << insideRamp << '\n';
//                 std::cout << "    ramp at ymax y=" << yMax << ": " << ymaxRamp << '\n';
//                 viscositySpongeDiagnostics::printSample("phi=0 outside", tau0, outsideRamp);
//                 viscositySpongeDiagnostics::printSample("phi=0 ymax", tau0, ymaxRamp);
//                 viscositySpongeDiagnostics::printSample("phi=1 outside", tau1, outsideRamp);
//                 viscositySpongeDiagnostics::printSample("phi=1 ymax", tau1, ymaxRamp);

//                 const host::label_t yPerDevice = ny / mesh.nDevices<axis::Y>();
//                 for (host::label_t dy = 0; dy < mesh.nDevices<axis::Y>(); ++dy)
//                 {
//                     const host::label_t yStart = dy * yPerDevice;
//                     const host::label_t yStop = (dy == (mesh.nDevices<axis::Y>() - static_cast<host::label_t>(1)))
//                                                    ? yMax
//                                                    : (yStart + yPerDevice - static_cast<host::label_t>(1));
//                     std::cout << "    device-y " << dy << " global y range: [" << yStart << ", " << yStop << "], ramp at local ymax: "
//                               << viscositySpongeDiagnostics::rampYmax(ny, yStop) << '\n';
//                 }
//             }
//             else
//             {
//                 (void)programCtrl;
//                 std::cout << "    physical Y is periodic, so the ymax sponge is disabled." << '\n';
//             }
// #else
//             (void)mesh;
//             (void)programCtrl;
// #endif
//         }
//     }
// }

// #endif
