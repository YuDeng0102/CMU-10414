#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
                                  float *theta, size_t m, size_t n, size_t k,
                                  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): posize_ter to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): posize_ter to y data, of size m
     *     theta (float *): posize_ter to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (size_t): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    float *Z = new float[batch * k]();
    float *one_hot = new float[batch * k]();
    float *grad = new float[n * k]();

    for (size_t t = 0; t < (m + batch - 1) / batch; t++)
    {
        size_t start_idx = t * batch, end_idx = std::min(start_idx + batch, m);
        size_t current_size = end_idx - start_idx;

        for (size_t i = 0; i < std::max(n, batch); i++)
        {
            for (size_t j = 0; j < k; j++)
            {
                if (i < batch)
                {
                    Z[i * k + j] = 0;
                    one_hot[i * k + j] = 0;
                }
                if (i < n)
                {
                    grad[i * k + j] = 0;
                }
            }
        }
        for (size_t i = start_idx; i < end_idx; i++)
        {
            one_hot[(i - start_idx) * k + y[i]] = 1;
        }

        for (size_t i = 0; i < current_size; i++)
        {
            for (size_t j = 0; j < k; j++)
            {

                for (size_t u = 0; u < n; u++)
                {
                    Z[i * k + j] += X[(i + start_idx) * n + u] * theta[u * k + j];
                }
            }
        }

        // soft_max
        for (size_t i = 0; i < current_size; i++)
        {
            float sum = 0;
            for (size_t j = 0; j < k; j++)
            {
                sum += Z[i * k + j] = std::exp(Z[i * k + j]);
            }
            for (size_t j = 0; j < k; j++)
            {
                Z[i * k + j] = Z[i * k + j] / sum - one_hot[i * k + j];
            }
        }

        // gradient
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < k; j++)
                for (size_t u = 0; u < current_size; u++)
                {
                    grad[i * k + j] += X[(u + start_idx) * n + i] * Z[u * k + j];
                }

        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < k; j++)
            {
                theta[i * k + j] -= grad[i * k + j] * lr / batch;
            }
    }
    delete[] Z;
    delete[] one_hot;
    delete[] grad;

    /// END YOUR CODE
}

/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m)
{
    m.def("softmax_regression_epoch_cpp", [](py::array_t<float, py::array::c_style> X, py::array_t<unsigned char, py::array::c_style> y, py::array_t<float, py::array::c_style> theta, float lr, size_t batch)
          { softmax_regression_epoch_cpp(
                static_cast<const float *>(X.request().ptr),
                static_cast<const unsigned char *>(y.request().ptr),
                static_cast<float *>(theta.request().ptr),
                X.request().shape[0],
                X.request().shape[1],
                theta.request().shape[1],
                lr,
                batch); }, py::arg("X"), py::arg("y"), py::arg("theta"), py::arg("lr"), py::arg("batch"));
}
