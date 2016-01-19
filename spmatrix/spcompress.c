/* spcompress.c
 * 
 * Copyright (C) 2012-2014 Patrick Alken
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or (at
 * your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */

#include <config.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spmatrix.h>

/*
gsl_spmatrix_compress()
  Create a sparse matrix in compressed format

Inputs: T - sparse matrix in triplet format
        sptype - matrix type to which to compress (either CCS or CRS)

Return: pointer to new matrix (should be freed when finished with it)
*/

/* AT: either compressed column or row storage depending on sptype */
gsl_spmatrix *
gsl_spmatrix_compress(const gsl_spmatrix *T, const size_t sptype)
{
  const size_t *TOuter; /* column (CCS) / row (CRS) indices of triplet matrix */
  size_t *Cp;       /* outer pointers of compressed matrix */
  size_t *w;        /* copy of outer pointers */
  gsl_spmatrix *m;
  size_t n;

  /* AT: Error handling for matrix type */
  if (sptype == GSL_SPMATRIX_TRIPLET)
    {
      GSL_ERROR_NULL("choice of sparse matrix type should not be triplet", GSL_EINVAL);
    }
  else if ((sptype != GSL_SPMATRIX_CCS) && (sptype != GSL_SPMATRIX_CRS))
    {
      GSL_ERROR_NULL("unknown choice of sparse matrix type", GSL_EINVAL);
    }

  /* AT: Allocate for the chosen matrix type */
  m = gsl_spmatrix_alloc_nzmax(T->size1, T->size2, T->nz, sptype);
  
  /* AT: Error handling */
  if (!m)
    GSL_ERROR_NULL("failed to allocate space for spmatrix struct",
		   GSL_ENOMEM);

  /* To which indices of triplet matrix
   * do the outer indices of m correspond to? */
  if (sptype == GSL_SPMATRIX_CCS)
    {
      /* outer indices of T = columns of triplet */
      TOuter = T->p;                 
    }
  else if (sptype == GSL_SPMATRIX_CRS)
    {
      /* outer indices of T = rows of triplet */
      TOuter = T->i;                 
    }

  /* Alias to pointer */
  Cp = m->p;

  /* AT: initialize outer pointers to 0 */
  for (n = 0; n < m->outerSize + 1; ++n)
    Cp[n] = 0;

  /*
   * compute the number of elements in each column:
   * Cp[j] = # non-zero elements in column j
   */
  for (n = 0; n < T->nz; ++n)
    Cp[TOuter[n]]++;

  /* compute column pointers: p[j] = p[j-1] + nnz[j-1] */
  gsl_spmatrix_cumsum(m->outerSize, Cp);

  /* AT: make a copy of the outer pointers */
  w = (size_t *) m->work;
  for (n = 0; n < m->outerSize; ++n)
    w[n] = Cp[n];

  if (sptype == GSL_SPMATRIX_CCS)
    {
      /* transfer data from triplet format to compressed column */
      for (n = 0; n < T->nz; ++n)
	{
	  size_t k = w[TOuter[n]]++;
	  m->i[k] = T->i[n];
	  m->data[k] = T->data[n];
	}
    }
  else if (sptype == GSL_SPMATRIX_CRS)
    {
      /* AT: transfer data from triplet format to compressed row */
      for (n = 0; n < T->nz; ++n)
	{
	  size_t k = w[TOuter[n]]++;
	  m->i[k] = T->p[n];
	  m->data[k] = T->data[n];
	}
    }

  m->nz = T->nz;

  return m;
} /* gsl_spmatrix_compcol() */

/*
gsl_spmatrix_cumsum()

Compute the cumulative sum:

p[j] = Sum_{k=0...j-1} c[k]

0 <= j < n + 1

Alternatively,
p[0] = 0
p[j] = p[j - 1] + c[j - 1]

Inputs: n - length of input array
        c - (input/output) array of size n + 1
            on input, contains the n values c[k]
            on output, contains the n + 1 values p[j]

Return: success or error
*/

void
gsl_spmatrix_cumsum(const size_t n, size_t *c)
{
  size_t sum = 0;
  size_t k;

  for (k = 0; k < n; ++k)
    {
      size_t ck = c[k];
      c[k] = sum;
      sum += ck;
    }

  c[n] = sum;
} /* gsl_spmatrix_cumsum() */
