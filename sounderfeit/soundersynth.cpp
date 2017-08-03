
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL sounderfeit_ARRAY_API
#include <numpy/ndarrayobject.h>

#include <boost/python/module.hpp>
#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/list.hpp>

namespace bpy = boost::python;

#include "Stk.h"
#include "RtAudio.h"
#include "Bowed.h"

using namespace stk;

#include <iostream>
#include <memory>

class Soundersynth;

int tick( void *outputBuffer, void *inputBuffer, unsigned int nBufferFrames,
          double streamTime, RtAudioStreamStatus status, void *dataPointer )
{
  Soundersynth *self = static_cast<Soundersynth*>(dataPointer);
  StkFloat *out = static_cast<StkFloat*>(outputBuffer);
  for (unsigned int i=0; i<nBufferFrames; i++)
    out[i] = ((double)i)/nBufferFrames-0.5;
  return 0;
}

class Soundersynth
{
protected:
  int _mode;
  bool _playing;
  double _position;
  double _pressure;
  double _volume;
  double _latent1;
  std::string _dataset;
  std::shared_ptr<RtAudio> _dac;
  std::vector< std::vector<double> > _cycles;
  int _currentCycle;

  // Decoder weight matrices
  std::vector<double> _w3, _w4, _b3, _b4;
  std::vector<double> _hidden, _input;

public:
  Soundersynth()
    : _mode(0), _playing(false), _position(32), _pressure(64)
    , _volume(0.5), _latent1(0.5)
    {
      for (int i=0; i<2; i++) {
        std::vector<double> array;
        array.resize(200);
        _cycles.push_back(array);
        for (int j=0; j<201; j++)
          _cycles[i][j] = j%30/30.;
      }
      _currentCycle = 0;
    }

  virtual ~Soundersynth() {};

  bool start() {
    Stk::setSampleRate(48000);
    _dac = std::make_shared<RtAudio>();
    RtAudioFormat format = ( sizeof(StkFloat) == 8 ) ? RTAUDIO_FLOAT64 : RTAUDIO_FLOAT32;
    RtAudio::StreamParameters parameters;
    parameters.deviceId = _dac->getDefaultOutputDevice();
    parameters.nChannels = 1;
    unsigned int bufferFrames = RT_BUFFER_SIZE;
    try {
      _dac->openStream( &parameters, NULL, format,
                        (unsigned int)Stk::sampleRate(), &bufferFrames,
                        &tick, (void *)this );
      _dac->startStream();
      _playing = true;
    }
    catch ( RtAudioError& error ) {
      error.printMessage();
      _dac.reset();
      return false;
    }
    return true;
  }

  void stop() {
    std::cout << "stop.." << std::endl;
    if (_dac) _dac->closeStream();
    _dac.reset();
    _playing = false;
  }

  bool playing() { return _playing; }

  enum MODE {
    DECODER,
    STK,
    LOOKUP,
  };

  enum PARAM {
    PRESSURE,
    POSITION,
    OTHER,
  };

  int modeCount() { return 2; }

  std::string modeName(int index) {
    if (index==DECODER)
      return "decoder";
    else if (index==STK)
      return "stk";
    else if (index==LOOKUP)
      return "lookup";
    return "";
  }

  double getMode(int index) {
    return 0.0;
  }

  void setMode(int index, double value) {
  }

  int paramCount() { return 4; }

  std::string paramName(int index) {
    if (index==0)
      return "position";
    else if (index==1)
      return "pressure";
    else if (index==2)
      return "volume";
    else if (index==3)
      return "latent1";
    return "";
  }

  double getParam(int index) {
    if (index==0)
      return _position;
    else if (index==1)
      return _pressure;
    else if (index==2)
      return _volume;
    else if (index==3)
      return _latent1;
    return 0.0;
  }

  void setParam(int index, double value) {
    if (index==0)
      _position = value;
    else if (index==1)
      _pressure = value;
    else if (index==2)
      _volume = value;
    else if (index==3)
      _latent1 = value;

    // update the cycle buffer with new parameters
    // TODO: do this in the audio loop instead
    decodeCycle();
  }

  std::string getDataset() {
    return _dataset;
  }

  void setDataset(std::string dataset) {
  }

  bpy::object currentCycle() {
    auto &cyc = _cycles[_currentCycle];
    npy_intp shape[1] = { (npy_int)cyc.size() };
    PyObject* obj = PyArray_New(&PyArray_Type, 1, shape, NPY_DOUBLE,
                                NULL, cyc.data(),
                                0, NPY_ARRAY_CARRAY_RO, NULL);
    bpy::handle<> arr( obj );
    return bpy::object(arr);
  }

  int decoderInputSize() {
    // Input layer is the size of w3 / size of the hidden biases
    if (_b3.size() > 0)
      return _w3.size() / _b3.size();
    else
      return 0;
  }

  int decoderHiddenSize() {
    // Hidden layer is the size of the hidden biases
    return _b3.size();
  }

  int decoderOutputSize() {
    // Output is the size of the output biases
    return _b4.size();
  }

  bool setDecoderWeights(bpy::list& arr_list) {
    if (bpy::len(arr_list) != 4)
      return false;

    double *data;
    npy_intp *dims;

    // w3
    PyArrayObject *arr = (PyArrayObject*)
      PyArray_FROM_OTF(bpy::object(arr_list[0]).ptr(),
                       NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!arr) return false;
    if (PyArray_NDIM(arr) != 2)
      goto nope;
    dims = PyArray_DIMS(arr);
    data = (double*)PyArray_DATA(arr);
    _w3.clear();
    for (npy_intp i=0; i < dims[0]*dims[1]; i++)
      _w3.push_back(data[i]);
    Py_DECREF(arr);

    // w4
    arr = (PyArrayObject*)
      PyArray_FROM_OTF(bpy::object(arr_list[1]).ptr(),
                       NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!arr) return false;
    if (PyArray_NDIM(arr) != 2)
      goto nope;
    dims = PyArray_DIMS(arr);
    data = (double*)PyArray_DATA(arr);
    _w4.clear();
    for (npy_intp i=0; i < dims[0]*dims[1]; i++)
      _w4.push_back(data[i]);

    // b3
    arr = (PyArrayObject*)
      PyArray_FROM_OTF(bpy::object(arr_list[2]).ptr(),
                       NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!arr) return false;
    if (PyArray_NDIM(arr) != 1)
      goto nope;
    dims = PyArray_DIMS(arr);
    data = (double*)PyArray_DATA(arr);
    _b3.clear();
    for (npy_intp i=0; i < dims[0]; i++)
      _b3.push_back(data[i]);

    // b4
    arr = (PyArrayObject*)
      PyArray_FROM_OTF(bpy::object(arr_list[3]).ptr(),
                       NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!arr) return false;
    if (PyArray_NDIM(arr) != 1)
      goto nope;
    dims = PyArray_DIMS(arr);
    data = (double*)PyArray_DATA(arr);
    _b4.clear();
    for (npy_intp i=0; i < dims[0]; i++)
      _b4.push_back(data[i]);

  nope:
    Py_DECREF(arr);
    return true;
  }

  // Use the current network weights to decode the current parameters
  // into a single cycle period.
  bool decodeCycle() {
    // We calculate a single feed forward iteration from scratch,
    // because who needs libraries.

    int inputSize = decoderInputSize();
    int hiddenSize = decoderHiddenSize();
    int outputSize = decoderOutputSize();

    auto &cyc = _cycles[_currentCycle];

    if (_w3.size() == 0 || _w4.size() == 0 || _b3.size() == 0 || _b4.size() == 0)
      return false;

    _hidden.resize(hiddenSize);
    _input.resize(inputSize);
    std::fill(_hidden.begin(), _hidden.end(), 0.0);
    std::fill(cyc.begin(), cyc.end(), 0.0);

    // TODO: depend on configuration
    _input[0] = _latent1;
    _input[1] = _position / 32.0 - 1.0;
    _input[2] = _pressure / 64.0 - 1.0;

    // w3
    for (int i=0; i < inputSize; i++) {
      for (int j=0; j < hiddenSize; j++) {
        _hidden[j] += _input[i] * _w3[i*hiddenSize + j];
      }
    }

    // b3
    for (int i=0; i < hiddenSize; i++) {
      _hidden[i] += _b3[i];

      // relu
      if (_hidden[i] < 0.0)
        _hidden[i] = 0.0;
    }

    // w4
    for (int i=0; i < hiddenSize; i++) {
      for (int j=0; j < outputSize; j++) {
        cyc[j] += _hidden[i] * _w4[i*outputSize + j];
      }
    }

    // b4
    for (int i=0; i < outputSize; i++)
      cyc[i] += _b4[i];

    return true;
  }
};

void* init_numpy()
{
  import_array();
  return NULL;
}

BOOST_PYTHON_MODULE(soundersynth)
{
  init_numpy();
  bpy::class_<Soundersynth>("Soundersynth", bpy::init<>())
    .def("start", &Soundersynth::start)
    .def("stop", &Soundersynth::stop)
    .def("playing", &Soundersynth::playing)
    .def("modeCount", &Soundersynth::modeCount)
    .def("modeName", &Soundersynth::modeName)
    .def("setMode", &Soundersynth::setMode)
    .def("getMode", &Soundersynth::getMode)
    .def("paramCount", &Soundersynth::paramCount)
    .def("paramName", &Soundersynth::paramName)
    .def("setParam", &Soundersynth::setParam)
    .def("getParam", &Soundersynth::getParam)
    .def("setDataset", &Soundersynth::setDataset)
    .def("getDataset", &Soundersynth::getDataset)
    .def("currentCycle", &Soundersynth::currentCycle)
    .def("decoderInputSize", &Soundersynth::decoderInputSize)
    .def("decoderHiddenSize", &Soundersynth::decoderHiddenSize)
    .def("decoderOutputSize", &Soundersynth::decoderOutputSize)
    .def("setDecoderWeights", &Soundersynth::setDecoderWeights)
    .def("decodeCycle", &Soundersynth::decodeCycle)
    ;
}
