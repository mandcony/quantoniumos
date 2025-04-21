// services/SimplifiedBloomFilter.js

export class SimplifiedBloomFilter {
  constructor(size, hashFns) {
    this.size = size;
    this.bitArray = new Array(size).fill(false);
    this.hashFns = hashFns;
  }

  add(item) {
    for (const fn of this.hashFns) {
      const idx = fn(item) % this.size;
      this.bitArray[idx] = true;
    }
  }

  test(item) {
    for (const fn of this.hashFns) {
      const idx = fn(item) % this.size;
      if (!this.bitArray[idx]) return false;
    }
    return true;
  }
}

export function hash1(s) {
  let h = 0; 
  const str = String(s);
  for (let i = 0; i < str.length; i++){
    h = (h << 5) - h + str.charCodeAt(i);
    h |= 0;
  }
  return Math.abs(h);
}

export function hash2(s) {
  let h = 5381;
  const str = String(s);
  for (let i = 0; i < str.length; i++){
    h = (h << 5) + h + str.charCodeAt(i);
  }
  return Math.abs(h);
}
