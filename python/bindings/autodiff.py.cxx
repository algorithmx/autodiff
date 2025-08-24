//                  _  _
//  _   _|_ _  _|o_|__|_
// (_||_||_(_)(_|| |  |
//
// automatic differentiation made easier in C++
// https://github.com/autodiff/autodiff
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright © 2018–2024 Allan Leal
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// pybind11 includes
#include "pybind11.hxx"

void export_dual1st(py::module& m);
#ifndef AUTODIFF_DISABLE_HIGHER_ORDER
void export_dual2nd(py::module& m);
void export_dual3rd(py::module& m);
void export_dual4th(py::module& m);
#endif

void export_real1st(py::module& m);
#ifndef AUTODIFF_DISABLE_HIGHER_ORDER
void export_real2nd(py::module& m);
void export_real3rd(py::module& m);
void export_real4th(py::module& m);
#endif

void exportVectorXdual0th(py::module& m);
void exportVectorXdual1st(py::module& m);
#ifndef AUTODIFF_DISABLE_HIGHER_ORDER
void exportVectorXdual2nd(py::module& m);
void exportVectorXdual3rd(py::module& m);
void exportVectorXdual4th(py::module& m);
#endif

void exportVectorXreal0th(py::module& m);
void exportVectorXreal1st(py::module& m);
#ifndef AUTODIFF_DISABLE_HIGHER_ORDER
void exportVectorXreal2nd(py::module& m);
void exportVectorXreal3rd(py::module& m);
void exportVectorXreal4th(py::module& m);
#endif

void exportMatrixXdual0th(py::module& m);
void exportMatrixXdual1st(py::module& m);
#ifndef AUTODIFF_DISABLE_HIGHER_ORDER
void exportMatrixXdual2nd(py::module& m);
void exportMatrixXdual3rd(py::module& m);
void exportMatrixXdual4th(py::module& m);
#endif

void exportMatrixXreal0th(py::module& m);
void exportMatrixXreal1st(py::module& m);
#ifndef AUTODIFF_DISABLE_HIGHER_ORDER
void exportMatrixXreal2nd(py::module& m);
void exportMatrixXreal3rd(py::module& m);
void exportMatrixXreal4th(py::module& m);
#endif

void exportArrayXdual0th(py::module& m);
void exportArrayXdual1st(py::module& m);
#ifndef AUTODIFF_DISABLE_HIGHER_ORDER
void exportArrayXdual2nd(py::module& m);
void exportArrayXdual3rd(py::module& m);
void exportArrayXdual4th(py::module& m);
#endif

void exportArrayXreal0th(py::module& m);
void exportArrayXreal1st(py::module& m);
#ifndef AUTODIFF_DISABLE_HIGHER_ORDER
void exportArrayXreal2nd(py::module& m);
void exportArrayXreal3rd(py::module& m);
void exportArrayXreal4th(py::module& m);
#endif

PYBIND11_MODULE(autodiff4py, m)
{
    export_dual1st(m);
#ifndef AUTODIFF_DISABLE_HIGHER_ORDER
    export_dual2nd(m);
    export_dual3rd(m);
    export_dual4th(m);
#endif

    m.attr("dual") = m.attr("dual1st");

    export_real1st(m);
#ifndef AUTODIFF_DISABLE_HIGHER_ORDER
    export_real2nd(m);
    export_real3rd(m);
    export_real4th(m);
#endif

    m.attr("real") = m.attr("real1st");

    exportVectorXdual0th(m);
    exportVectorXdual1st(m);
#ifndef AUTODIFF_DISABLE_HIGHER_ORDER
    exportVectorXdual2nd(m);
    exportVectorXdual3rd(m);
    exportVectorXdual4th(m);
#endif

    m.attr("VectorXdual") = m.attr("VectorXdual1st");

    exportVectorXreal0th(m);
    exportVectorXreal1st(m);
#ifndef AUTODIFF_DISABLE_HIGHER_ORDER
    exportVectorXreal2nd(m);
    exportVectorXreal3rd(m);
    exportVectorXreal4th(m);
#endif

    m.attr("VectorXreal") = m.attr("VectorXreal1st");

    exportMatrixXdual0th(m);
    exportMatrixXdual1st(m);
#ifndef AUTODIFF_DISABLE_HIGHER_ORDER
    exportMatrixXdual2nd(m);
    exportMatrixXdual3rd(m);
    exportMatrixXdual4th(m);
#endif

    m.attr("MatrixXdual") = m.attr("MatrixXdual1st");

    exportMatrixXreal0th(m);
    exportMatrixXreal1st(m);
#ifndef AUTODIFF_DISABLE_HIGHER_ORDER
    exportMatrixXreal2nd(m);
    exportMatrixXreal3rd(m);
    exportMatrixXreal4th(m);
#endif

    m.attr("MatrixXreal") = m.attr("MatrixXreal1st");

    exportArrayXdual0th(m);
    exportArrayXdual1st(m);
#ifndef AUTODIFF_DISABLE_HIGHER_ORDER
    exportArrayXdual2nd(m);
    exportArrayXdual3rd(m);
    exportArrayXdual4th(m);
#endif

    m.attr("ArrayXdual") = m.attr("ArrayXdual1st");

    exportArrayXreal0th(m);
    exportArrayXreal1st(m);
#ifndef AUTODIFF_DISABLE_HIGHER_ORDER
    exportArrayXreal2nd(m);
    exportArrayXreal3rd(m);
    exportArrayXreal4th(m);
#endif

    m.attr("ArrayXreal") = m.attr("ArrayXreal1st");
}
