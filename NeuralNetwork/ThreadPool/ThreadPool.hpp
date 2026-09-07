// Neural Network
// Copyright 2026 Daynlight
// Licensed under the GNU General.
// See LICENSE file for details.



#include "ThreadPool.h"



// =========================== //
// ======= Constructors ====== //
// =========================== //
inline NN::ThreadPool::ThreadPool(size_t thread_count) {
  if(thread_count == 0) thread_count = 1;
  threads.reserve(thread_count);

  try{
    for(size_t i = 0; i < thread_count; i++) threads.emplace_back([this]{ worker(); });
  }
  catch(...){
    {
      std::lock_guard<std::mutex> lock(mutex);
      stop = true;
    };

    condition.notify_all();
    for(std::thread& thread : threads) if(thread.joinable()) thread.join();
    throw;
  };
};



inline NN::ThreadPool::~ThreadPool() noexcept {
  {
    std::lock_guard<std::mutex> lock(mutex);
    stop = true;
  };

  condition.notify_all();
  for(std::thread& thread : threads) if(thread.joinable()) thread.join();
};



// =========================== //
// ========= Workers ========= //
// =========================== //
inline void NN::ThreadPool::worker() noexcept {
  while(true){
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock(mutex);
      condition.wait(lock, [this]{ return stop || !tasks.empty(); });

      if(stop && tasks.empty()) return;
      task = std::move(tasks.front());

      tasks.pop();
      active_tasks++;
    };

    task();
    {
      std::lock_guard<std::mutex> lock(mutex);
      active_tasks--;

      if(tasks.empty() && active_tasks == 0) finished_condition.notify_all();
    };
  };
};



// =========================== //
// ======== Execution ======== //
// =========================== //
template<typename F>
inline void NN::ThreadPool::parallel_for(size_t begin, size_t end, F&& function) {
  if(begin >= end) return;

  const size_t range = end - begin;
  const size_t worker_count = std::min(threads.size(), range);

  if(worker_count <= 1){
    function(begin, end);
    return;
  };

  const size_t base_size = range / worker_count;
  const size_t remainder = range % worker_count;

  size_t current = begin;
  std::function<void(size_t, size_t)> func = std::forward<F>(function);
  {
    std::lock_guard<std::mutex> lock(mutex);
    for(size_t i = 0; i < worker_count; i++){
      const size_t current_size = base_size + (i < remainder ? 1 : 0);
      const size_t task_begin = current;
      const size_t task_end = current + current_size;

      tasks.emplace([func, task_begin, task_end]{
        func(task_begin, task_end);
      });

      current = task_end;
    };
  };

  condition.notify_all();
};



inline void NN::ThreadPool::wait() noexcept {
  std::unique_lock<std::mutex> lock(mutex);
  finished_condition.wait(lock, [this]{ return tasks.empty() && active_tasks == 0; });
};



inline size_t NN::ThreadPool::size() const noexcept {
  return threads.size();
};