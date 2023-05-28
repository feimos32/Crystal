/*
    Copyright (C) <2023>  <Dezeming>  <feimos@mail.ustc.edu.cn>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or any
    later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

    Github site: <https://github.com/feimos32/Crystal>
*/

#include "Export_dll.cuh"

#include "CrystalAlgrithm/Basic/Transform.cuh"
#include "CrystalAlgrithm/Basic/Spectrum.cuh"
#include "CrystalAlgrithm/Basic/Geometry.cuh"
#include "CrystalAlgrithm/Basic/Common.cuh"

#include <iostream>

namespace CrystalAlgrithm {

void SpectrumTest() {
    Spectrum4 s(0.8f);
    Spectrum4 s1 = s * s;
    Spectrum4 s2 = s + s;

    std::cout << "s = " << s.ToString() << std::endl;
    std::cout << "s1 = " << s1.ToString() << std::endl;
    std::cout << "s2 = " << s2.ToString() << std::endl;
}

}



