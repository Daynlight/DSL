// Neural Network
// Copyright 2026 Daynlight
// Licensed under the GNU General.
// See LICENSE file for details.



#pragma once

#include <thread>
#include <vector>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <cstddef>
#include <utility>



namespace NN {

class ThreadPool{

// ======================================== //
// ================= Data ================= //
// ======================================== //
private:
  std::vector<std::thread> threads;

  std::queue<std::function<void()>> tasks;

  std::mutex mutex;

  std::condition_variable condition;

  std::condition_variable finished_condition;

  size_t active_tasks = 0;

  bool stop = false;

// ======================================== //
// ============== Functions =============== //
// ======================================== //

// =========================== //
// ======= Constructors ====== //
// =========================== //
public:
  explicit ThreadPool(size_t thread_count = std::max(1u, std::thread::hardware_concurrency()));

  ~ThreadPool() noexcept;

  ThreadPool(const ThreadPool&) = delete;

  ThreadPool& operator=(const ThreadPool&) = delete;

  ThreadPool(ThreadPool&&) = delete;

  ThreadPool& operator=(ThreadPool&&) = delete;

// =========================== //
// ========= Workers ========= //
// =========================== //
private:
  void worker() noexcept;

// =========================== //
// ======== Execution ======== //
// =========================== //
public:
  template<typename F>
  void parallel_for(size_t begin, size_t end, F&& function);

  void wait() noexcept;

  size_t size() const noexcept;

};

};



#include "ThreadPool.hpp"