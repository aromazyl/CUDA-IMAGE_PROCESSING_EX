/*
 * Storage.h
 * Copyright (C) 2017 zhangyule <zyl2336709@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef STORAGE_H
#define STORAGE_H

#include "state.h"

typedef struct Storage {
  unsigned char *data;
  ptrdiff_t size;
  int refcount;
  char flag;
  Allocator* allocator;
  void * allocatorContext;
  struct Storage* view;
  int device;
} Storage;

Storage* Storage_set(State* state, Storage* self, ptrdiff_t index, float value);
float Storage_get(State* state, const Storage * self, ptrdiff_t index);
Storage* Storage_new(State* state);
Storage* Storage_newWithSize(State* state, ptrdiff_t size);
Storage* Storage_newWithAllocator(State* state, ptridff_t size, DeviceAllocator* allocator, void* allocatorContext);
Storage* Storage_newWithSize1(State* state, float data0);
Storage* Storage_newWithSize1(State* state, float data0, float data1);
Storage* Storage_newWithSize1(State* state, float data0, float data1, float data2);
Storage* Storage_newWithSize1(State* state, float data0, float data1, float data2, float data3);
void Storage_retain(State* state, Storage* self);
void Storage_free(State* state, Storage* self);

#endif /* !STORAGE_H */
