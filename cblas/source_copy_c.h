/* blas/source_copy_c.h
 * 
 * Copyright (C) 1996, 1997, 1998, 1999, 2000 Gerard Jungman
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or (at
 * your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

{
  size_t i;
  size_t ix = OFFSET(N, incX);
  size_t iy = OFFSET(N, incY);

  for (i = 0; i < N; i++) {
    REAL(Y, iy) = REAL(X, ix);
    IMAG(Y, iy) = IMAG(X, ix);
    ix += incX;
    iy += incY;
  }
}
