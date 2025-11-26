/**
 * Complex Number Operations
 * Essential mathematical operations for complex numbers in RFT and quantum algorithms
 */

export interface Complex {
  re: number;
  im: number;
}

/**
 * Add two complex numbers
 */
export function add(a: Complex, b: Complex): Complex {
  return {
    re: a.re + b.re,
    im: a.im + b.im
  };
}

/**
 * Subtract two complex numbers
 */
export function subtract(a: Complex, b: Complex): Complex {
  return {
    re: a.re - b.re,
    im: a.im - b.im
  };
}

/**
 * Multiply two complex numbers
 */
export function multiply(a: Complex, b: Complex): Complex {
  return {
    re: a.re * b.re - a.im * b.im,
    im: a.re * b.im + a.im * b.re
  };
}

/**
 * Divide two complex numbers
 */
export function divide(a: Complex, b: Complex): Complex {
  const denominator = b.re * b.re + b.im * b.im;
  if (denominator === 0) {
    throw new Error('Division by zero');
  }
  return {
    re: (a.re * b.re + a.im * b.im) / denominator,
    im: (a.im * b.re - a.re * b.im) / denominator
  };
}

/**
 * Complex conjugate
 */
export function conj(a: Complex): Complex {
  return {
    re: a.re,
    im: -a.im
  };
}

/**
 * Absolute value (magnitude) of complex number
 */
export function abs(a: Complex): number {
  return Math.sqrt(a.re * a.re + a.im * a.im);
}

/**
 * Square root of complex number
 */
export function sqrt(a: Complex): Complex {
  const r = abs(a);
  if (r === 0) {
    return { re: 0, im: 0 };
  }

  const re = Math.sqrt((r + a.re) / 2);
  const im = Math.sign(a.im) * Math.sqrt((r - a.re) / 2);

  return { re, im };
}

/**
 * Complex exponential: e^(a)
 */
export function exp(a: Complex): Complex {
  const expRe = Math.exp(a.re);
  return {
    re: expRe * Math.cos(a.im),
    im: expRe * Math.sin(a.im)
  };
}

/**
 * Scalar multiplication
 */
export function scale(scalar: number, a: Complex): Complex {
  return {
    re: scalar * a.re,
    im: scalar * a.im
  };
}

/**
 * Phase (argument) of complex number
 */
export function phase(a: Complex): number {
  return Math.atan2(a.im, a.re);
}

/**
 * Create complex number from polar coordinates
 */
export function fromPolar(r: number, theta: number): Complex {
  return {
    re: r * Math.cos(theta),
    im: r * Math.sin(theta)
  };
}

/**
 * Complex number to string
 */
export function toString(a: Complex, precision: number = 3): string {
  const re = a.re.toFixed(precision);
  const im = Math.abs(a.im).toFixed(precision);
  const sign = a.im >= 0 ? '+' : '-';
  return `${re} ${sign} ${im}i`;
}
