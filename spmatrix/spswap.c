/* spswap.c
 * 
 * Copyright (C) 2014 Patrick Alken
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
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spmatrix.h>

#include "avl.c"

int
gsl_spmatrix_transpose_memcpy(gsl_spmatrix *dest, const gsl_spmatrix *src)
{
  const size_t M = src->size1;
  const size_t N = src->size2;

  if (M != dest->size2 || N != dest->size1)
    {
      GSL_ERROR("dimensions of dest must be transpose of src matrix",
                GSL_EBADLEN);
    }
  else if (dest->sptype != src->sptype)
    {
      GSL_ERROR("cannot copy matrices of different storage formats",
                GSL_EINVAL);
    }
  else
    {
      int s = GSL_SUCCESS;
      const size_t nz = src->nz;

      if (dest->nzmax < src->nz)
        {
          s = gsl_spmatrix_realloc(src->nz, dest);
	  /* ?AT: Error handler ? */
          if (s)
            return s;
        }

      if (GSL_SPMATRIX_ISTRIPLET(src))
        {
          size_t n;
          void *ptr;

          for (n = 0; n < nz; ++n)
            {
              dest->i[n] = src->p[n];
              dest->p[n] = src->i[n];
              dest->data[n] = src->data[n];

              /* copy binary tree data */
              ptr = avl_insert(dest->tree_data->tree, &dest->data[n]);
              if (ptr != NULL)
                {
                  GSL_ERROR("detected duplicate entry", GSL_EINVAL);
                }
            }
        }
      /* AT: add CRS support */
      else if (GSL_SPMATRIX_ISCCS(src) || GSL_SPMATRIX_ISCRS(src))
        {
          size_t *Ai = src->i;
          size_t *Ap = src->p;
          double *Ad = src->data;
          size_t *ATi = dest->i;
          size_t *ATp = dest->p;
          double *ATd = dest->data;
          size_t *w = (size_t *) dest->work;
          size_t n, p, outerIdx;

          /* initialize to 0 */
          for (outerIdx = 0; outerIdx < dest->outerSize + 1; ++outerIdx)
            ATp[outerIdx] = 0;

	  /* count rows (CCS) / columns (CRS) of src^T = dest
	   * (= columns (CCS) / rows (CRS) of src) */
          for (n = 0; n < nz; ++n)
            ATp[Ai[n]]++;
	  
          /* compute  pointers for A (= column pointers for A^T) */
          gsl_spmatrix_cumsum(dest->outerSize, ATp);

          /* make copy of row pointers */
          for (outerIdx = 0; outerIdx < dest->outerSize; ++outerIdx)
            w[outerIdx] = ATp[outerIdx];

          for (outerIdx = 0; outerIdx < src->outerSize; ++outerIdx)
            {
              for (p = Ap[outerIdx]; p < Ap[outerIdx + 1]; ++p)
                {
                  size_t k = w[Ai[p]]++;
                  ATi[k] = outerIdx;
                  ATd[k] = Ad[p];
                }
            }
        }
      else
        {
          GSL_ERROR("unknown sparse matrix type", GSL_EINVAL);
        }

      dest->nz = nz;

      return s;
    }
} /* gsl_spmatrix_transpose_memcpy() */

/*
gsl_spmatrix_transpose()
  Replace the sparse matrix src by its transpose either by
  swapping its row and column indices if it is in triplet storage,
  or by switching its major if it is in compressed storage.

Inputs: src - (input/output) sparse matrix to transpose.
*/

int
gsl_spmatrix_transpose(gsl_spmatrix *src)
{
  size_t tmp;
  size_t n;

  if (GSL_SPMATRIX_ISTRIPLET(src))
    {
      for (n = 0; n < src->nz; ++n)
        {
	  tmp = src->p[n];
	  src->p[n] = src->i[n];
	  src->i[n] = tmp;
        }
    }
  else if (GSL_SPMATRIX_ISCCS(src))
    {
      src->sptype = GSL_SPMATRIX_CRS;
    }
  else if (GSL_SPMATRIX_ISCRS(src))
    {
      src->sptype = GSL_SPMATRIX_CCS;
    }
  else
    {
      GSL_ERROR("unknown sparse matrix type", GSL_EINVAL);
    }
  
  /* Swap dimensions */
  tmp = src->size1;
  src->size1 = src->size2;
  src->size2 = tmp;
  
  return GSL_SUCCESS;
}

/**
 * \brief Convert a compressed matrix from row (resp. column) major
 * to column (resp. row) major.
 *
 * Convert a compressed matrix from row (resp. column) major
 * to column (resp. row) major.
 * \param[in] src Source matrix to convert.
 * \return        Converted matrix.
 */
int
gsl_spmatrix_switch_major(gsl_spmatrix *dest, const gsl_spmatrix *src)
{
  if (GSL_SPMATRIX_ISTRIPLET(src))
    {
      GSL_ERROR_NULL("sparse matrix type should not be triplet", GSL_EINVAL);
    }
  
  /** Get transpose of source matrix with copy */
  if (gsl_spmatrix_transpose_memcpy(dest, src))
    {
      GSL_ERROR_NULL("error transposing with copy", GSL_EINVAL);
    }

  /** Transpose back but in place, effectively switching type */
  gsl_spmatrix_transpose(dest);
  
  return GSL_SUCCESS;
}
