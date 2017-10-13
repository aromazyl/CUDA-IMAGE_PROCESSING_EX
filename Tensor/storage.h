/*
 * Storage.h
 * Copyright (C) 2017 zhangyule <zyl2336709@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef STORAGE_H
#define STORAGE_H

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

Storage* Storage_new()
#endif /* !STORAGE_H */
