// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include "Kokkos_Version_Info.hpp"

namespace Kokkos {
namespace Impl {

std::string GIT_BRANCH       = R"branch(develop)branch";
std::string GIT_COMMIT_HASH  = "3b9f95d";
std::string GIT_CLEAN_STATUS = "CLEAN";
std::string GIT_COMMIT_DESCRIPTION =
    R"message(Merge pull request #9008 from dalg24/rm_deprecated_code_simd)message";
std::string GIT_COMMIT_DATE = "2026-03-31T10:03:25-06:00";

}  // namespace Impl

}  // namespace Kokkos
