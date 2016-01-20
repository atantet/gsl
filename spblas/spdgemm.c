/* spdgemm.c
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
#include <gsl/gsl_spmatrix.h>
#include <gsl/gsl_spblas.h>
#include <gsl/gsl_errno.h>

/*
gsl_spblas_dgemm()
  Multiply two sparse matrices

Inputs: alpha - scalar factor
        A     - sparse matrix
        B     - sparse matrix
        C     - (output) C = alpha * A * B

Return: success or error

Notes:
1) based on CSparse routine cs_multiply
*/

int
gsl_spblas_dgemm(const double alpha, const gsl_spmatrix *A1,
                 const gsl_spmatrix *B1, gsl_spmatrix *C)
{
  if (A1->size2 != B1->size1 || A1->size1 != C->size1 || B1->size2 != C->size2)
    {
      GSL_ERROR("matrix dimensions do not match", GSL_EBADLEN);
    }
  else if (A1->sptype != B1->sptype || A1->sptype != C->sptype)
    {
      GSL_ERROR("matrix storage formats do not match", GSL_EINVAL);
    }
  else if (!(GSL_SPMATRIX_ISCCS(A1) || GSL_SPMATRIX_CRS(A1)))
    {
      GSL_ERROR("compressed column format required", GSL_EINVAL);
    }
  else
    {
      gsl_spmatrix *C, *A2, *B2;
      
      if (GSL_SPMATRIX_ISCRS(A1))
	{
	  /* AT: In case of CRS use multiply on CCS by calculating (alpha*B^T*A^T)^T
	   * Transposing in place and then back would be dangerous.
	   * Instead, copy and transpose the matrices without reallocating
	   * and changin only the flags and dimensions.
	   * Care should be taken not to modify the pointers. */
	  B = (gsl_spmatrix *) calloc(1, sizeof(gsl_spmatrix));
	  B->size1 = A1->size2;
	  B->size2 = A1->size1;
	  B->sptype = GSL_SPMATRIX_CCS;
	  B->innerSize = A1->innerSize;
	  B->outerSize = A1->outerSize;
	  B->i = A1->i;
	  B->data = A1->data;
	  B->p = A1->p;
	  B->nzmax = A1->nzmax;
	  B->nz = A1->nz;
	  B->work = A1->work;

	  A = gsl_spmatrix_alloc_nzmax(B1->size2, B1->size1, 0, GSL_SPMATRIX_CCS);
	  A->size1 = B1->size2;
	  A->size2 = B1->size1;
	  A->sptype = GSL_SPMATRIX_CCS;
	  A->innerSize = B1->innerSize;
	  A->outerSize = B1->outerSize;
	  A->i = B1->i;
	  A->data = B1->data;
	  A->p = B1->p;
	  A->nzmax = B1->nzmax;
	  A->nz = B1->nz;
	  A->work = B1->work;
	}
      else
	{
	  A = A1;
	  B = B1;
	}
      int status = GSL_SUCCESS;
      const size_t M = A->size1;
      const size_t N = B->size2;
      size_t *Bi = B->i;
      size_t *Bp = B->p;
      double *Bd = B->data;
      size_t *w = (size_t *) A1->work; /* workspace of length M */
      double *x = (double *) C->work; /* workspace of length M */
      size_t *Cp, *Ci;
      double *Cd;
      size_t j, p;
      size_t nz = 0;

      if (C->nzmax < A->nz + B->nz)
        {
          status = gsl_spmatrix_realloc(A->nz + B->nz, C);
          if (status)
            {
              GSL_ERROR("unable to realloc matrix C", status);
            }
        }

      /* initialize workspace to 0 */
      for (j = 0; j < M; ++j)
        w[j] = 0;

      Cp = C->p;
      Ci = C->i;
      Cd = C->data;

      for (j = 0; j < N; ++j)
        {
          if (nz + M > C->nzmax)
            {
              status = gsl_spmatrix_realloc(2 * C->nzmax + M, C);
              if (status)
                {
                  GSL_ERROR("unable to realloc matrix C", status);
                }

              /* these pointers could have changed due to reallocation */
              Ci = C->i;
              Cd = C->data;
            }

          Cp[j] = nz; /* column j of C starts here */

          for (p = Bp[j]; p < Bp[j + 1]; ++p)
            {
              nz = gsl_spblas_scatter(A, Bi[p], Bd[p], w, x, j + 1, C, nz);
            }

          for (p = Cp[j]; p < nz; ++p)
            Cd[p] = x[Ci[p]];
        }

      Cp[N] = nz;
      C->nz = nz;

      /* scale by alpha */
      gsl_spmatrix_scale(C, alpha);

      /* AT: Calculate the transpose of C in place */
      if (GSL_SPMATRIX_ISCRS(A))
	{
	  if ((status = gsl_spmatrix_transpose(C)) != GSL_SUCCESS)
	    {
	      GSL_ERROR_NULL("could not transpose result of \
matrix multiplication in place", GSL_ENOMEM);
	    }
	}
      
      return status;
    }
} /* gsl_spblas_dgemm() */

/*
gsl_spblas_scatter()

  Keep a running total x -> x + alpha*A(:,j) for adding matrices together in CCS,
which will eventually be stored in C(:,j)

  When a new non-zero element with row index i is found, update C->i with
the row index. C->data is updated only by the calling function after all
matrices have been added via this function.

Inputs: A     - sparse matrix m-by-n
        j     - column index
        alpha - scalar factor
        w     - keeps track which rows of column j have been added to C;
                initialize to 0 prior to first call
        x     - column vector of length m
        mark  -
        C     - output matrix whose jth column will be added to A(:,j)
        nz    - (input/output) number of non-zeros in matrix C

Notes:
1) This function is designed to be called successively when adding multiple
matrices together. Column j of C is stored contiguously as per CCS but not
necessarily in order - ie: the row indices C->i may not be in ascending order.

2) based on CSparse routine cs_scatter
*/

size_t
gsl_spblas_scatter(const gsl_spmatrix *A, const size_t outerIdx,
		   const double alpha, size_t *w, double *x,
		   const size_t mark, gsl_spmatrix *C, size_t nz)
{
  size_t p;
  size_t *Ai = A->i;
  size_t *Ap = A->p;
  double *Ad = A->data;
  size_t *Ci = C->i;

  /* AT: just remain i and j which are not row and col in the CRS case */
  for (p = Ap[outerIdx]; p < Ap[outerIdx + 1]; ++p)
    {
      /* A(innerIdx,outerIdx) (CCS)
       * A(outerIdx,innerIdx) (CRS) is nonzero */
      size_t innerIdx = Ai[p];          

      /* check if inner index innerIdx has been stored 
       * in outer outerIdx yet */
      if (w[innerIdx] < mark)           
        {
	  /* innerIdx is new entry in outer outerIdx */
          w[innerIdx] = mark;
	  
	  /* add innerIdx to pattern of
	   * C(:,outerIdx) (CCS) / C(outerIdx,:) (CRS)*/
	  Ci[nz++] = innerIdx;
	  /* x(innerIdx) = alpha * A(innerIdx,outerIdx) (CCS)
	   * x(innerIdx) = alpha * A(outerIdx,innerIdx) (CRS) */
          x[innerIdx] = alpha * Ad[p];  
        }
      else    
        {
	  /* this (innerIdx,outerIdx) (CCS) / (outerIdx,innerIdx) (CRS)
	   * exists in C from a previous call */
	  /* add alpha*A(innerIdx,outerIdx) to C(innerIdx,outerIdx) (CCS) 
	   *     alpha*A(outerIdx,innerIdx) to C(outerIdx,innerIdx) (CRS) */
          x[innerIdx] += alpha * Ad[p]; 
        }
    }

  return (nz) ;
} /* gsl_spblas_scatter() */
