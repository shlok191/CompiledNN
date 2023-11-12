#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "Model.h"
#include "CompiledNN.h"
#include "SimpleNN.h"
#include "Tensor.h"

namespace py = pybind11;

PYBIND11_MODULE(compiledNN, m) {

    m.def("apply", (void (*)(std::vector<NeuralNetwork::TensorXf>&, std::vector<NeuralNetwork::TensorXf>&, const Model&, const NodeCallback&)) &apply, 
          py::arg("input"), py::arg("output"), py::arg("specification"), py::arg("nodeCallback") = NodeCallback(),
          "Applies the net described by the given specification on the given tensors.");

    m.def("apply", (void (*)(const std::vector<NeuralNetwork::TensorXf>&, std::vector<NeuralNetwork::TensorXf>&, const NerualNetwork::Node&)) &apply,
          py::arg("input"), py::arg("output"), py::arg("node"),
          "Applies a net consisting of only the given node on the given tensors.");

    m.def("apply", (void (*)(std::vector<NeuralNetwork::TensorXf>&, std::vector<NeuralNetwork::TensorXf>&, const std::string&)) &apply,
          py::arg("input"), py::arg("output"), py::arg("filename"),
          "Applies the net from the given file on the given tensors.");

    py::bind_vector<std::vector<std::unique_ptr<Layer>>>(m, "VectorOfUniquePtrLayer");
    py::bind_vector<std::vector<TensorLocation>>(m, "VectorOfTensorLocation");

    py::class_<TensorLocation>(m, "TensorLocation")
        .def(py::init<const Layer*, unsigned int, unsigned int>(),
             py::arg("layer"), py::arg("nodeIndex"), py::arg("tensorIndex"))
        .def_readonly("layer", &TensorLocation::layer)
        .def_readonly("nodeIndex", &TensorLocation::nodeIndex)
        .def_readonly("tensorIndex", &TensorLocation::tensorIndex)
        .def("__eq__", &TensorLocation::operator==, py::is_operator());

    py::class_<NeuralNetwork::Node>(m, "Node")
        .def(py::init<const NeuralNetwork::Layer*>())
        .def_readonly("layer", &NeuralNetwork::Node::layer)
        .def_readwrite("inputs", &NeuralNetwork::Node::inputs)
        .def_readwrite("outputs", &NeuralNetwork::Node::outputs)
        .def_readwrite("inputDimensions", &NeuralNetwork::Node::inputDimensions)
        .def_readwrite("outputDimensions", &NeuralNetwork::Node::outputDimensions)
        .def("setDimensions", &NeuralNetwork::Node::setDimensions);

    py::class_<NeuralNetwork::TensorXf>(m, "TensorXf")
        .def(py::init<>())
        .def(py::init<const std::vector<unsigned int>&, std::size_t>())
        .def("reshape", (void (NeuralNetwork::TensorXf::*)(const std::vector<unsigned int>&, std::size_t)) &NeuralNetwork::TensorXf::reshape)
        .def("size", &NeuralNetwork::TensorXf::size)
        .def("rank", &NeuralNetwork::TensorXf::rank)
        .def("dims", &NeuralNetwork::TensorXf::dims)
        .def("__getitem__", [](const NeuralNetwork::TensorXf &s, std::size_t i) {
            if (i >= s.size()) throw py::index_error();
            return s[i];
        })
        .def("__setitem__", [](NeuralNetwork::TensorXf &s, std::size_t i, float v) {
            if (i >= s.size()) throw py::index_error();
            s[i] = v;
        })
        .def("__iter__", [](const NeuralNetwork::TensorXf &s) {
            return py::make_iterator(s.begin(), s.end());
        }, py::keep_alive<0, 1>() /* Essential: keep object alive while iterator exists */);

    py::class_<Model>(m, "Model")
    .def(py::init<>())
    .def(py::init<const std::string&>())
    .def("getLayers", &Model::getLayers, py::return_value_policy::reference_internal)
    .def("getInputs", &Model::getInputs, py::return_value_policy::reference_internal)
    .def("getOutputs", &Model::getOutputs, py::return_value_policy::reference_internal)
    .def("setInputUInt8", &Model::setInputUInt8)
    .def("isInputUInt8", &Model::isInputUInt8)
    .def("clear", &Model::clear)
    .def("load", &Model::load);

    py::class_<CompiledNN>(m, "CompiledNN")
    .def(py::init<asmjit::JitRuntime*>(), py::arg("runtime") = nullptr)
    .def("compile", (void (CompiledNN::*)(const Model&, const CompilationSettings&)) &CompiledNN::compile, 
            py::arg("specification"), py::arg("settings") = CompilationSettings())
    .def("compile", (void (CompiledNN::*)(const NeuralNetwork::Node&, const CompilationSettings&)) &CompiledNN::compile, 
            py::arg("node"), py::arg("settings") = CompilationSettings())
    .def("compile", (void (CompiledNN::*)(const std::string&, const CompilationSettings&)) &CompiledNN::compile, 
            py::arg("filename"), py::arg("settings") = CompilationSettings())
    .def("valid", &CompiledNN::valid)
    .def("numOfInputs", &CompiledNN::numOfInputs)
    .def("input", &CompiledNN::input, py::return_value_policy::reference_internal)
    .def("numOfOutputs", &CompiledNN::numOfOutputs)
    .def("output", &CompiledNN::output, py::return_value_policy::reference_internal)
    .def("apply", &CompiledNN::apply);
}
