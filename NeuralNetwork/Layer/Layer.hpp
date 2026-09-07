// Neural Network
// Copyright 2026 Daynlight
// Licensed under the GNU General.
// See LICENSE file for details.



#include "Layer.h"



// =========================== //
// ======= Constructors ====== //
// =========================== //
// core
template <unsigned int S, unsigned int D>
inline NN::Layer<S, D>::Layer() noexcept {
  setWeights(-1, 1);

  loss = std::make_unique<NN::MSE>();
  activation = std::make_unique<NN::Linear>();
};



template <unsigned int S, unsigned int D>
inline NN::Layer<S, D>::~Layer() noexcept { };



// ============================== //
// ======= Setters/Getters ====== //
// ============================== //
template <unsigned int S, unsigned int D>
inline double *NN::Layer<S, D>::getNodes() noexcept {
  return nodes;
};



template <unsigned int S, unsigned int D>
inline double *NN::Layer<S, D>::getActivatedNodes() noexcept{
  activateNodes();
  return activated;
};



template <unsigned int S, unsigned int D>
inline void NN::Layer<S, D>::setNodes(std::initializer_list<double> nodes) noexcept {
  unsigned int i = 0;
  for (auto it = nodes.begin(); it != nodes.end() && i < S; ++it, ++i) this->nodes[i] = *it;
};



template<unsigned int S, unsigned int D>
inline void NN::Layer<S, D>::setNodes(std::span<const double> nodes) noexcept {
  unsigned int i = 0;
  for(const double& node : nodes){
    if(i >= S) break;
    this->nodes[i++] = node;
  };
};



template <unsigned int S, unsigned int D>
inline double NN::Layer<S, D>::getActivatedNode(unsigned int i) const noexcept {
  std::span<const double> layer(nodes, S);
  return activation->fun(layer, i);

};



template <unsigned int S, unsigned int D>
inline void NN::Layer<S, D>::activateNodes() noexcept {
  if(activated_after_forward) return;
  for(size_t i = 0; i < S; i++) activated[i] = getActivatedNode(i);
  activated_after_forward = true;
};



template <unsigned int S, unsigned int D>
inline double *NN::Layer<S, D>::getWeights() noexcept {
  return weights;
};



template <unsigned int S, unsigned int D>
inline void NN::Layer<S, D>::setWeights(std::initializer_list<double> weights) noexcept {
  unsigned int i = 0;
  for (auto it = weights.begin(); it != weights.end() && i < (S + 1) * D; ++it, ++i) this->weights[i] = *it;
};



template <unsigned int S, unsigned int D>
inline void NN::Layer<S, D>::setWeights(const double* weights) noexcept {
  for (unsigned int i = 0; i < (S + 1) * D; ++i) this->weights[i] = weights[i];
};



template <unsigned int S, unsigned int D>
inline void NN::Layer<S, D>::setWeights(double min, double max) noexcept {  
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(min, max);
  
  for(double &el : weights) el = dist(gen);
  
  for(size_t j = 0; j < D; j++) weights[j * (S + 1) + S] = 0.0;
};



template <unsigned int S, unsigned int D>
inline double NN::Layer<S, D>::getLearningRate() const noexcept {
  return learning_rate;
};



template <unsigned int S, unsigned int D>
inline void NN::Layer<S, D>::setLearningRate(double learning_rate) noexcept {
  this->learning_rate = learning_rate;
};



template <unsigned int S, unsigned int D>
inline const std::unique_ptr<NN::iActivation>& NN::Layer<S, D>::getActivation() const noexcept {
  return activation;
};



template <unsigned int S, unsigned int D>
template<typename T>
inline void NN::Layer<S, D>::setActivation() noexcept {
  activation = std::make_unique<T>();
};



template <unsigned int S, unsigned int D>
inline const std::unique_ptr<NN::iLoss>& NN::Layer<S, D>::getLoss() const noexcept {
  return loss;
};



template <unsigned int S, unsigned int D>
template<typename T>
inline void NN::Layer<S, D>::setLoss() noexcept {
  loss = std::make_unique<T>();
};



template <unsigned int S, unsigned int D>
inline double &NN::Layer<S, D>::operator[](unsigned int i) {
  if(i > S) throw std::range_error("index out of range");
  return nodes[i];
};



template <unsigned int S, unsigned int D>
inline const double *NN::Layer<S, D>::getSigma() const noexcept {
  return sigma;
};



// =============================== //
// ======= Forward/Backprop ====== //
// =============================== //
template <unsigned int S, unsigned int D>
template <unsigned int N>
inline void NN::Layer<S, D>::forward(NN::Layer<D, N> &layer) {
  nodes[S] = 1.0;
  double output[D];

  activated_after_forward = false;
  activateNodes();

  workers.parallel_for(0, D, [&](size_t begin, size_t end){
    for(unsigned int i = begin; i < end; i++){
      double sum = weights[i * (S + 1) + S];
      for(unsigned int j = 0; j < S; j++)
        sum += activated[j] * weights[i * (S + 1) + j];
      output[i] = sum;
    };
  });
  workers.wait();

  std::memcpy(layer.getNodes(), output, sizeof(double) * D);
  layer[D] = 1;
};



template <unsigned int S, unsigned int D>
inline void NN::Layer<S, D>::backprop_initial(std::initializer_list<double> target) noexcept {
  backprop_initial(std::span<const double>(target.begin(), target.size()));
};



template <unsigned int S, unsigned int D>
inline void NN::Layer<S, D>::backprop_initial(std::span<const double> target) noexcept {
  activateNodes();
  
  if(dynamic_cast<NN::Softmax*>(activation.get()) && dynamic_cast<NN::CrossEntropy*>(loss.get())){
    backprop_initial_softmax_cross_entropy_fuse(target);
    return;
  };
  
  double output[S]{};
  std::span<const double> layer(nodes, S);

  double loss_gradient[S];

  const size_t size = std::min<size_t>(S, target.size());
  for(size_t i = 0; i < size; i++){
    loss_gradient[i] = loss->fun_prime(activated[i], target[i]);
  };

  workers.parallel_for(0, S, [&](size_t begin, size_t end){
    for(size_t i = begin; i < end; i++){
      double sum = 0.0;
      for(size_t j = 0; j < size; j++) sum += loss_gradient[j] * activation->fun_prime(layer, j, i);
      output[i] = sum;
    };
  });
  workers.wait();

  std::memcpy(sigma, output, sizeof(output));
};



template <unsigned int S, unsigned int D>
inline void NN::Layer<S, D>::backprop_initial_softmax_cross_entropy_fuse(std::span<const double> target) noexcept {
  activateNodes();
  
  double output[S]{};

  double exponent[S];
  double max = nodes[0];

  for(size_t i = 1; i < S; i++) if(nodes[i] > max) max = nodes[i];

  double sum = 0.0;
  for(size_t i = 0; i < S; i++){
    exponent[i] = std::exp(nodes[i] - max);
    sum += exponent[i];
  };

  const size_t size = std::min<size_t>(S, target.size());
  for(size_t i = 0; i < size; i++) output[i] = exponent[i] / sum - target[i];

  std::memcpy(sigma, output, sizeof(output));
  activated_after_forward = false;
};



template <unsigned int S, unsigned int D>
template <unsigned int N>
inline void NN::Layer<S, D>::backprop(Layer<D, N>& next_layer) noexcept {
  activateNodes();
  const double* sigma_next = next_layer.getSigma();

  double gradient[S]{};
  double output[S]{};

  workers.parallel_for(0, S, [&](size_t begin, size_t end){
    for(size_t i = begin; i < end; i++){
      double sum = 0.0;
      for(size_t j = 0; j < D; j++) sum += weights[j * (S + 1) + i] * sigma_next[j];
      gradient[i] = sum;
    };
  });
  workers.wait();

  std::span<const double> layer(nodes, S);
  if(dynamic_cast<NN::Linear*>(activation.get())){
    std::memcpy(output, gradient, sizeof(output));
  }
  else if(dynamic_cast<NN::Sigmoid*>(activation.get()) || dynamic_cast<NN::ReLU*>(activation.get())){
    for(size_t i = 0; i < S; i++) output[i] = gradient[i] * activation->fun_prime(layer, i, i);
  }
  else{
    workers.parallel_for(0, S, [&](size_t begin, size_t end){
      for(size_t i = begin; i < end; i++){
        double sum = 0.0;
        for(size_t j = 0; j < S; j++) sum += gradient[j] * activation->fun_prime(layer, j, i);
        output[i] = sum;
      };
    });

    workers.wait();
  };

  std::memcpy(sigma, output, sizeof(output));

  workers.parallel_for(0, D, [&](size_t begin, size_t end){
    for(size_t j = begin; j < end; j++){
      const double factor = learning_rate * sigma_next[j];
      double* row = weights + j * (S + 1);
      for(size_t i = 0; i < S; i++)
        row[i] -= factor * activated[i];
      row[S] -= factor;
    };
  });
  workers.wait();

  activated_after_forward = false;
};



// =========================== //
// ======= Presentation ====== //
// =========================== //
template <unsigned int S, unsigned int D>
inline std::string NN::Layer<S, D>::print() const {
  std::string s = "size: \n";

  s += std::to_string(S) + ", ";
  s += std::to_string(D) + "; ";

  // s += "\nloss, activation \n";
  // s += std::to_string(loss->getType()) + "; ";
  // if(activation)
    // s += std::to_string(activation->getType()) + "; ";
  // else
    // s += "-1; ";


  bool first = true;
  s += "\nnodes: \n[";
  for(double el : nodes){
    if(!first) 
      s += ", ";
      s += std::to_string(el);
      first = false;
  }
  s += "]; ";
  
  first = true;
  unsigned int i = 0;
  s += "\nweights: \n[[";
  for(double el : weights){
    if(!first && i % (S + 1) != 0)
      s += ", ";

    if(i % (S + 1) == 0 && i <= (S + 1) * D && i != 0) 
      s += "], \n[";

    s += std::to_string(el);
    first = false;
    i++;
  };
  s += "]];\n ";
  return s;
};



template <unsigned int S, unsigned int D>
inline std::string NN::Layer<S, D>::serialize() const noexcept {
  std::string data;

  auto write = [&](Field id, const void* source, uint64_t size){
    uint32_t field = static_cast<uint32_t>(id);

    data.append(reinterpret_cast<const char*>(&field), sizeof(field));
    data.append(reinterpret_cast<const char*>(&size), sizeof(size));

    if(size > 0) data.append(reinterpret_cast<const char*>(source), size);
  };

  uint32_t s = S;
  uint32_t d = D;

  write(Field::S_val, &s, sizeof(s));
  write(Field::D_val, &d, sizeof(d));
  write(Field::LEARNING_RATE, &learning_rate, sizeof(learning_rate));

  uint32_t activation_id = 0;

  if(dynamic_cast<NN::Linear*>(activation.get()))
    activation_id = static_cast<uint32_t>(Activation::LINEAR);
  else if(dynamic_cast<NN::Sigmoid*>(activation.get()))
    activation_id = static_cast<uint32_t>(Activation::SIGMOID);
  else if(dynamic_cast<NN::Softmax*>(activation.get()))
    activation_id = static_cast<uint32_t>(Activation::SOFTMAX);
  else if(dynamic_cast<NN::ReLU*>(activation.get()))
    activation_id = static_cast<uint32_t>(Activation::RELU);

  write(Field::ACTIVATION, &activation_id, sizeof(activation_id));

  uint32_t loss_id = 0;

  if(dynamic_cast<NN::MSE*>(loss.get()))
  loss_id = static_cast<uint32_t>(Loss::MSE);
  else if(dynamic_cast<NN::CrossEntropy*>(loss.get()))
    loss_id = static_cast<uint32_t>(Loss::CROSSENTROPY);

  write(Field::LOSS, &loss_id, sizeof(loss_id));

  write(Field::WEIGHTS, weights, sizeof(weights));

  uint32_t end = static_cast<uint32_t>(Field::END);
  data.append(reinterpret_cast<const char*>(&end), sizeof(end));

  return data;
};



template <unsigned int S, unsigned int D>
inline void NN::Layer<S, D>::deserialize(const std::string &data){
  size_t offset = 0;

  auto read = [&](void* destination, size_t size){
    if(offset + size > data.size()) throw std::runtime_error("Invalid serialized layer data");

    std::memcpy(destination, data.data() + offset, size);
    offset += size;
  };

  while(offset < data.size()){
    uint32_t field_id = 0;
    read(&field_id, sizeof(field_id));

    Field field = static_cast<Field>(field_id);

    if(field == Field::END) break;

    uint64_t size = 0;
    read(&size, sizeof(size));

    if(offset + size > data.size()) throw std::runtime_error("Invalid serialized layer field size");

    switch(field){
      case Field::S_val:{
        uint32_t value = 0;
        read(&value, sizeof(value));
        if(value != S)
          throw std::runtime_error("Layer S size mismatch");
        break;
      };

      case Field::D_val:{
        uint32_t value = 0;
        read(&value, sizeof(value));
        if(value != D)
          throw std::runtime_error("Layer D size mismatch");
        break;
      };

      case Field::LEARNING_RATE:{
        if(size != sizeof(learning_rate))
          throw std::runtime_error("Invalid learning rate size");
        read(&learning_rate, sizeof(learning_rate));
        break;
      };

      case Field::ACTIVATION:{
        uint32_t activation_id = 0;
        read(&activation_id, sizeof(activation_id));

        switch(static_cast<Activation>(activation_id)){
          case Activation::LINEAR:
            setActivation<NN::Linear>();
            break;
          case Activation::SIGMOID:
            setActivation<NN::Sigmoid>();
            break;
          case Activation::SOFTMAX:
            setActivation<NN::Softmax>();
            break;
          case Activation::RELU:
            setActivation<NN::ReLU>();
            break;
          default:
            throw std::runtime_error("Unknown activation type");
        };
        break;
      };

      case Field::LOSS:{
        uint32_t loss_id = 0;
        read(&loss_id, sizeof(loss_id));

        switch(static_cast<Loss>(loss_id)){
          case Loss::MSE:
            setLoss<NN::MSE>();
            break;
          case Loss::CROSSENTROPY:
            setLoss<NN::CrossEntropy>();
            break;
          default:
            throw std::runtime_error("Unknown loss type");
        };
        break;
      };

      case Field::WEIGHTS:{
        if(size != sizeof(weights))
          throw std::runtime_error("Invalid weights size");
        read(weights, sizeof(weights));
        break;
      };

      default:{
        offset += size;
        break;
      };
    };
  };
};
