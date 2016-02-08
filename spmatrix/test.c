/* test.c
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
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_test.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_spmatrix.h>

/*
create_random_sparse()
  Create a random sparse matrix with approximately
M*N*density non-zero entries
Inputs: M       - number of rows
        N       - number of columns
        density - sparse density \in [0,1]
                  0 = no non-zero entries
                  1 = all m*n entries are filled
        r       - random number generator
Return: pointer to sparse matrix in triplet format (must be freed by caller)
Notes:
1) non-zero matrix entries are uniformly distributed in [0,1]
*/

static gsl_spmatrix *
create_random_sparse(const size_t M, const size_t N, const double density,
		     const gsl_rng *r)
{
  size_t nnzwanted = (size_t) floor(M * N * GSL_MIN(density, 1.0));
  gsl_spmatrix *m = gsl_spmatrix_alloc_nzmax(M, N,
					     nnzwanted,
					     GSL_SPMATRIX_TRIPLET);

  while (gsl_spmatrix_nnz(m) < nnzwanted)
    {
      /* generate a random row and column */
      size_t i = gsl_rng_uniform(r) * M;
      size_t j = gsl_rng_uniform(r) * N;

      /* generate random m_{ij} and add it */
      double x = gsl_rng_uniform(r);
      gsl_spmatrix_set(m, i, j, x, 0);
    }

  return m;
} /* create_random_sparse() */

static void
test_getset(const size_t M, const size_t N, const gsl_rng *r)
{
  int status;
  size_t i, j;

  /* test triplet versions of _get and _set */
  {
    size_t k = 0;
    gsl_spmatrix *m = gsl_spmatrix_alloc(M, N);

    status = 0;
    for (i = 0; i < M; ++i)
      {
	for (j = 0; j < N; ++j)
	  {
	    double x = (double) ++k;
	    double y;

	    gsl_spmatrix_set(m, i, j, x, 0);
	    y = gsl_spmatrix_get(m, i, j);
	    if (x != y)
	      status = 1;
	  }
      }

    gsl_test(status, "test_getset: M=%zu N=%zu _get != _set", M, N);

    /* test setting an element to 0 */
    gsl_spmatrix_set(m, 0, 0, 1.0, 0);
    gsl_spmatrix_set(m, 0, 0, 0.0, 0);

    status = gsl_spmatrix_get(m, 0, 0) != 0.0;
    gsl_test(status, "test_getset: M=%zu N=%zu m(0,0) = %f",
	     M, N, gsl_spmatrix_get(m, 0, 0));

    /* test gsl_spmatrix_set_zero() */
    gsl_spmatrix_set(m, 0, 0, 1.0, 0);
    gsl_spmatrix_set_zero(m);
    status = gsl_spmatrix_get(m, 0, 0) != 0.0;
    gsl_test(status, "test_getset: M=%zu N=%zu set_zero m(0,0) = %f",
	     M, N, gsl_spmatrix_get(m, 0, 0));

    /* resassemble matrix to ensure nz is calculated correctly */
    k = 0;
    for (i = 0; i < M; ++i)
      {
	for (j = 0; j < N; ++j)
	  {
	    double x = (double) ++k;
	    gsl_spmatrix_set(m, i, j, x, 0);
	  }
      }

    status = gsl_spmatrix_nnz(m) != M * N;
    gsl_test(status, "test_getset: M=%zu N=%zu set_zero nz = %zu",
	     M, N, gsl_spmatrix_nnz(m));

    gsl_spmatrix_free(m);
  }

  /* test duplicate values are handled correctly */
  {
    size_t min = GSL_MIN(M, N);
    size_t expected_nnz = min;
    size_t nnz;
    size_t k = 0;
    gsl_spmatrix *m = gsl_spmatrix_alloc(M, N);

    status = 0;
    for (i = 0; i < min; ++i)
      {
	for (j = 0; j < 5; ++j)
	  {
	    double x = (double) ++k;
	    double y;

	    gsl_spmatrix_set(m, i, i, x, 0);
	    y = gsl_spmatrix_get(m, i, i);
	    if (x != y)
	      status = 1;
	  }
      }

    gsl_test(status, "test_getset: duplicate test M=%zu N=%zu _get != _set", M, N);

    nnz = gsl_spmatrix_nnz(m);
    status = nnz != expected_nnz;
    gsl_test(status, "test_getset: duplicate test M=%zu N=%zu nnz=%zu, expected=%zu",
	     M, N, nnz, expected_nnz);

    gsl_spmatrix_free(m);
  }

  /* test compressed version of gsl_spmatrix_get() */
  {
    gsl_spmatrix *T = create_random_sparse(M, N, 0.3, r);
    gsl_spmatrix *C = gsl_spmatrix_compress(T, GSL_SPMATRIX_CCS);
    gsl_spmatrix *CR = gsl_spmatrix_compress(T, GSL_SPMATRIX_CRS);

    status = 0;
    for (i = 0; i < M; ++i)
      {
	for (j = 0; j < N; ++j)
	  {
	    double Tij = gsl_spmatrix_get(T, i, j);
	    double Cij = gsl_spmatrix_get(C, i, j);

	    if (Tij != Cij)
	      status = 1;
	  }
      }

    gsl_test(status, "test_getset: M=%zu N=%zu compressed column _get", M, N);

    status = 0;
    for (i = 0; i < M; ++i)
      {
	for (j = 0; j < N; ++j)
	  {
	    double Tij = gsl_spmatrix_get(T, i, j);
	    double Cij = gsl_spmatrix_get(CR, i, j);

	    if (Tij != Cij)
	      status = 1;
	  }
      }

    gsl_test(status, "test_getset: M=%zu N=%zu compressed row _get", M, N);

    gsl_spmatrix_free(T);
    gsl_spmatrix_free(C);
    gsl_spmatrix_free(CR);
  }
} /* test_getset() */

static void
test_memcpy(const size_t M, const size_t N, const gsl_rng *r)
{
  int status;

  {
    gsl_spmatrix *at = create_random_sparse(M, N, 0.2, r);
    gsl_spmatrix *ac = gsl_spmatrix_compress(at, GSL_SPMATRIX_CCS);
    gsl_spmatrix *ar = gsl_spmatrix_compress(at, GSL_SPMATRIX_CRS);
    gsl_spmatrix *bt, *bc, *br;

    bt = gsl_spmatrix_alloc(M, N);
    gsl_spmatrix_memcpy(bt, at);

    status = gsl_spmatrix_equal(at, bt) != 1;
    gsl_test(status, "test_memcpy: _memcpy M=%zu N=%zu triplet format", M, N);

    bc = gsl_spmatrix_alloc_nzmax(M, N, ac->nzmax, GSL_SPMATRIX_CCS);
    gsl_spmatrix_memcpy(bc, ac);

    status = gsl_spmatrix_equal(ac, bc) != 1;
    gsl_test(status, "test_memcpy: _memcpy M=%zu N=%zu compressed column format", M, N);

    br = gsl_spmatrix_alloc_nzmax(M, N, ar->nzmax, GSL_SPMATRIX_CRS);
    gsl_spmatrix_memcpy(br, ar);

    status = gsl_spmatrix_equal(ar, br) != 1;
    gsl_test(status, "test_memcpy: _memcpy M=%zu N=%zu compressed row format", M, N);

    gsl_spmatrix_free(at);
    gsl_spmatrix_free(ac);
    gsl_spmatrix_free(ar);
    gsl_spmatrix_free(bt);
    gsl_spmatrix_free(bc);
    gsl_spmatrix_free(br);
  }

  /* test transpose_memcpy */
  {
    gsl_spmatrix *A = create_random_sparse(M, N, 0.3, r);
    gsl_spmatrix *B = gsl_spmatrix_compress(A, GSL_SPMATRIX_CCS);
    gsl_spmatrix *BR = gsl_spmatrix_compress(A, GSL_SPMATRIX_CRS);
    gsl_spmatrix *AT = gsl_spmatrix_alloc(N, M);
    gsl_spmatrix *BT = gsl_spmatrix_alloc_nzmax(N, M, 1, GSL_SPMATRIX_CCS);
    gsl_spmatrix *BTR = gsl_spmatrix_alloc_nzmax(N, M, 1, GSL_SPMATRIX_CRS);
    size_t i, j;

    gsl_spmatrix_transpose_memcpy(AT, A);
    gsl_spmatrix_transpose_memcpy(BT, B);
    gsl_spmatrix_transpose_memcpy(BTR, BR);

    status = 0;
    for (i = 0; i < M; ++i)
      {
	for (j = 0; j < N; ++j)
	  {
	    double Aij = gsl_spmatrix_get(A, i, j);
	    double ATji = gsl_spmatrix_get(AT, j, i);

	    if (Aij != ATji)
	      status = 1;
	  }
      }

    gsl_test(status, "test_memcpy: _transpose_memcpy M=%zu N=%zu triplet format", M, N);

    status = 0;
    for (i = 0; i < M; ++i)
      {
	for (j = 0; j < N; ++j)
	  {
	    double Aij = gsl_spmatrix_get(A, i, j);
	    double Bij = gsl_spmatrix_get(B, i, j);
	    double BTji = gsl_spmatrix_get(BT, j, i);

	    if ((Bij != BTji) || (Aij != Bij))
	      status = 1;
	  }
      }

    gsl_test(status, "test_memcpy: _transpose_memcpy M=%zu N=%zu column format", M, N);

    status = 0;
    for (i = 0; i < M; ++i)
      {
	for (j = 0; j < N; ++j)
	  {
	    double Aij = gsl_spmatrix_get(A, i, j);
	    double BRij = gsl_spmatrix_get(BR, i, j);
	    double BTRji = gsl_spmatrix_get(BTR, j, i);
	    
	    if ((Aij != BRij) || (BRij != BTRji))
	      status = 1;
	  }
      }
    gsl_test(status, "test_memcpy: _transpose_memcpy M=%zu N=%zu row format", M, N);

    gsl_spmatrix_free(A);
    gsl_spmatrix_free(AT);
    gsl_spmatrix_free(B);
    gsl_spmatrix_free(BT);
    gsl_spmatrix_free(BR);
    gsl_spmatrix_free(BTR);
  }
} /* test_memcpy() */

static void
test_ops(const size_t M, const size_t N, const gsl_rng *r)
{
  size_t i, j;
  int status;

  /* test gsl_spmatrix_add */
  {
    gsl_spmatrix *Ta = create_random_sparse(M, N, 0.2, r);
    gsl_spmatrix *Tb = create_random_sparse(M, N, 0.2, r);
    gsl_spmatrix *a = gsl_spmatrix_compress(Ta, GSL_SPMATRIX_CCS);
    gsl_spmatrix *b = gsl_spmatrix_compress(Tb, GSL_SPMATRIX_CCS);
    gsl_spmatrix *c = gsl_spmatrix_alloc_nzmax(M, N, 1, GSL_SPMATRIX_CCS);
    gsl_spmatrix *ar = gsl_spmatrix_compress(Ta, GSL_SPMATRIX_CRS);
    gsl_spmatrix *br = gsl_spmatrix_compress(Tb, GSL_SPMATRIX_CRS);
    gsl_spmatrix *cr = gsl_spmatrix_alloc_nzmax(M, N, 1, GSL_SPMATRIX_CRS);

    gsl_spmatrix_add(c, a, b);
    gsl_spmatrix_add(cr, ar, br);

    status = 0;
    for (i = 0; i < M; ++i)
      {
	for (j = 0; j < N; ++j)
	  {
	    double aij = gsl_spmatrix_get(a, i, j);
	    double bij = gsl_spmatrix_get(b, i, j);
	    double cij = gsl_spmatrix_get(c, i, j);
	    double crij = gsl_spmatrix_get(cr, i, j);

	    if ((aij + bij != cij) || (aij + bij != crij))
	      status = 1;
	  }
      }

    gsl_test(status, "test_ops: _add M=%zu N=%zu compressed format", M, N);

    gsl_spmatrix_free(Ta);
    gsl_spmatrix_free(Tb);
    gsl_spmatrix_free(a);
    gsl_spmatrix_free(b);
    gsl_spmatrix_free(c);
    gsl_spmatrix_free(ar);
    gsl_spmatrix_free(br);
    gsl_spmatrix_free(cr);
  }
} /* test_ops() */

void
test_prop(const size_t M, const size_t N, const double density,
	  const double d, const gsl_rng *r)
{
  gsl_spmatrix *m, *test, *ccs, *crs;
  size_t n, p, outerIdx;
  int status, any;
 
  m = create_random_sparse(M, N, density, r);
  ccs = gsl_spmatrix_compress(m, GSL_SPMATRIX_CCS);
  crs = gsl_spmatrix_compress(m, GSL_SPMATRIX_CRS);
  
  // For triplet
  /** Test greater than */
  test = gsl_spmatrix_gt_elements(m, d);
  status = 0;
  for (n = 0; n < m->nz; n++)
    {
      if ((m->data[n] > d) != (gsl_spmatrix_get(test, m->i[n], m->p[n])))
	status = 1;
    }
  gsl_test(status, "test_prop: M=%zu N=%zu _gt_elements triplet", M, N);
  gsl_spmatrix_free(test);
  
  /** Test greater or equal than */
  test = gsl_spmatrix_ge_elements(m, d);
  status = 0;
  for (n = 0; n < m->nz; n++)
    {
      if ((m->data[n] >= d) != (gsl_spmatrix_get(test, m->i[n], m->p[n])))
	status = 1;
    }
  gsl_test(status, "test_prop: M=%zu N=%zu _ge_elements triplet", M, N);
  gsl_spmatrix_free(test);
  
  /** Test lower than */
  test = gsl_spmatrix_lt_elements(m, d);
  status = 0;
  for (n = 0; n < m->nz; n++)
    {
      if ((m->data[n] < d) != (gsl_spmatrix_get(test, m->i[n], m->p[n])))
	status = 1;
    }
  gsl_test(status, "test_prop: M=%zu N=%zu _lt_elements triplet", M, N);
  gsl_spmatrix_free(test);
  
  /** Test lower or equal than */
  test = gsl_spmatrix_le_elements(m, d);
  status = 0;
  any = 0;
  for (n = 0; n < m->nz; n++)
    {
      if ((m->data[n] <= d) != (gsl_spmatrix_get(test, m->i[n], m->p[n])))
	status = 1;
      if (m->data[n] <= d)
	any = 1;
    }
  gsl_test(status, "test_prop: M=%zu N=%zu _le_elements triplet", M, N);
  gsl_spmatrix_free(test);
  
  /** Test any */
  status = !(any == (int) gsl_spmatrix_any(test));
  gsl_test(status, "test_prop: M=%zu N=%zu _any triplet", M, N);

  // For CCS
  /** Test greater than */
  test = gsl_spmatrix_gt_elements(ccs, d);
  status = 0;
  for (outerIdx = 0; outerIdx < ccs->outerSize; outerIdx++)
    {
      for (p = ccs->p[outerIdx]; p < ccs->p[outerIdx + 1]; p++)
	{
	  if (((int) gsl_spmatrix_get(test, ccs->i[p], outerIdx)) != (ccs->data[p] > d))
	    status = 1;
	}
    }
  gsl_test(status, "test_prop: M=%zu N=%zu _gt_elements CCS", M, N);
  gsl_spmatrix_free(test);
  
  /** Test greater or equal than */
  test = gsl_spmatrix_ge_elements(ccs, d);
  status = 0;
  for (outerIdx = 0; outerIdx < ccs->outerSize; outerIdx++)
    {
      for (p = ccs->p[outerIdx]; p < ccs->p[outerIdx + 1]; p++)
	{
	  if (((int) gsl_spmatrix_get(test, ccs->i[p], outerIdx)) != (ccs->data[p] >= d))
	    status = 1;
	}
    }
  gsl_test(status, "test_prop: M=%zu N=%zu _ge_elements CCS", M, N);
  gsl_spmatrix_free(test);
  
  /** Test lower than */
  test = gsl_spmatrix_lt_elements(ccs, d);
  status = 0;
  for (outerIdx = 0; outerIdx < ccs->outerSize; outerIdx++)
    {
      for (p = ccs->p[outerIdx]; p < ccs->p[outerIdx + 1]; p++)
	{
	  if (((int) gsl_spmatrix_get(test, ccs->i[p], outerIdx)) != (ccs->data[p] < d))
	    status = 1;
	}
    }
  gsl_test(status, "test_prop: M=%zu N=%zu _lt_elements CCS", M, N);
  gsl_spmatrix_free(test);
  
  /** Test lower or equal than */
  test = gsl_spmatrix_le_elements(ccs, d);
  status = 0;
  any = 0;
  for (outerIdx = 0; outerIdx < ccs->outerSize; outerIdx++)
    {
      for (p = ccs->p[outerIdx]; p < ccs->p[outerIdx + 1]; p++)
	{
	  if (((int) gsl_spmatrix_get(test, ccs->i[p], outerIdx)) != (ccs->data[p] <= d))
	    status = 1;
	  if (ccs->data[p] <= d)
	    any = 1;
	}
    }
  gsl_test(status, "test_prop: M=%zu N=%zu _le_elements CCS", M, N);
  gsl_spmatrix_free(test);
  
  /** Test any */
  status = !(any == (int) gsl_spmatrix_any(test));
  gsl_test(status, "test_prop: M=%zu N=%zu _any CCS", M, N);
  

    // For CRS
  /** Test greater than */
  test = gsl_spmatrix_gt_elements(crs, d);
  status = 0;
  for (outerIdx = 0; outerIdx < crs->outerSize; outerIdx++)
    {
      for (p = crs->p[outerIdx]; p < crs->p[outerIdx + 1]; p++)
	{
	  if (((int) gsl_spmatrix_get(test, outerIdx, crs->i[p])) != (crs->data[p] > d))
	    status = 1;
	}
    }
  gsl_test(status, "test_prop: M=%zu N=%zu _gt_elements CRS", M, N);
  gsl_spmatrix_free(test);
  
  /** Test greater or equal than */
  test = gsl_spmatrix_ge_elements(crs, d);
  status = 0;
  for (outerIdx = 0; outerIdx < crs->outerSize; outerIdx++)
    {
      for (p = crs->p[outerIdx]; p < crs->p[outerIdx + 1]; p++)
	{
	  if (((int) gsl_spmatrix_get(test, outerIdx, crs->i[p])) != (crs->data[p] >= d))
	    status = 1;
	}
    }
  gsl_test(status, "test_prop: M=%zu N=%zu _ge_elements CRS", M, N);
  gsl_spmatrix_free(test);
  
  /** Test lower than */
  test = gsl_spmatrix_lt_elements(crs, d);
  status = 0;
  for (outerIdx = 0; outerIdx < crs->outerSize; outerIdx++)
    {
      for (p = crs->p[outerIdx]; p < crs->p[outerIdx + 1]; p++)
	{
	  if (((int) gsl_spmatrix_get(test, outerIdx, crs->i[p])) != (crs->data[p] < d))
	    status = 1;
	}
    }
  gsl_test(status, "test_prop: M=%zu N=%zu _lt_elements CRS", M, N);
  gsl_spmatrix_free(test);
  
  /** Test lower or equal than */
  test = gsl_spmatrix_le_elements(crs, d);
  status = 0;
  any = 0;
  for (outerIdx = 0; outerIdx < crs->outerSize; outerIdx++)
    {
      for (p = crs->p[outerIdx]; p < crs->p[outerIdx + 1]; p++)
	{
	  if (((int) gsl_spmatrix_get(test, outerIdx, crs->i[p])) != (crs->data[p] <= d))
	    status = 1;
	  if (crs->data[p] <= d)
	    any = 1;
	}
    }
  gsl_test(status, "test_prop: M=%zu N=%zu _le_elements CRS", M, N);
  gsl_spmatrix_free(test);
  
  /** Test any */
  status = !(any == (int) gsl_spmatrix_any(test));
  gsl_test(status, "test_prop: M=%zu N=%zu _any CRS", M, N);
  

  gsl_spmatrix_free(m);
  gsl_spmatrix_free(ccs);
  gsl_spmatrix_free(crs);
  
  return;
}

void
test_manip(const size_t M, const size_t N, const double density,
	   const gsl_rng *r)
{
  int status;
  gsl_spmatrix *tri, *ccs, *crs, *test;
  gsl_matrix *dense, *denseDivRows, *denseDivCols;
  double sum, sumDense;
  gsl_vector *v;
  gsl_vector *denseRowSum, *denseColSum;
  size_t i, j;

  tri = create_random_sparse(M, N, density, r);
  dense = gsl_matrix_alloc(M, N);
  gsl_spmatrix_sp2d(dense, tri);

  /** Get row sum and col sum aswell as divided matrices for dense */
  denseDivRows = gsl_matrix_calloc(M, N);
  denseDivCols = gsl_matrix_calloc(M, N);  
  denseRowSum = gsl_vector_calloc(M);
  denseColSum = gsl_vector_calloc(N);
  sumDense = 0.;
  for (i = 0; i < M; i++)
    {
      for (j = 0; j < N; j++)
	{
	  denseRowSum->data[i * denseRowSum->stride] += gsl_matrix_get(dense, i, j);
	  denseColSum->data[j * denseColSum->stride] += gsl_matrix_get(dense, i, j);
	  sumDense += gsl_matrix_get(dense, i, j);
	}
    }
  for (i = 0; i < M; i++)
    {
      for (j = 0; j < N; j++)
	{
	  if (gsl_pow_2(denseRowSum->data[i * denseRowSum->stride]) > 1.e-12)
	    {
	      gsl_matrix_set(denseDivRows, i, j, gsl_matrix_get(dense, i, j)
			     / denseRowSum->data[i * denseRowSum->stride]);
	    }
	  else
	    {
	      gsl_matrix_set(denseDivRows, i, j, gsl_matrix_get(dense, i, j));
	    }

	  if (gsl_pow_2(denseColSum->data[j * denseColSum->stride]) > 1.e-12)
	    {
	      gsl_matrix_set(denseDivCols, i, j, gsl_matrix_get(dense, i, j)
			     / denseColSum->data[j * denseColSum->stride]);
	    }
	  else
	    {
	      gsl_matrix_set(denseDivCols, i, j, gsl_matrix_get(dense, i, j));
	    }
	}
    }
							   
  
  // Compress
  ccs = gsl_spmatrix_compress(tri, GSL_SPMATRIX_CCS);
  crs = gsl_spmatrix_compress(tri, GSL_SPMATRIX_CRS);

  
  /** TOTAL SUM */
  /** Triplet */
  sum = gsl_spmatrix_get_sum(tri);
  status = !(sum == sumDense);
  gsl_test(status, "test_manip: M=%zu N=%zu _get != _get_sum triplet", M, N);
  
  /** CCS */
  sum = gsl_spmatrix_get_sum(ccs);
  status = !(sum == sumDense);
  gsl_test(status, "test_manip: M=%zu N=%zu _get != _get_sum CCS", M, N);
  
  /** CRS */
  sum = gsl_spmatrix_get_sum(crs);
  status = !(sum == sumDense);
  gsl_test(status, "test_manip: M=%zu N=%zu _get != _get_sum CRS", M, N);


  /** COLUMN SUM AND DIVIDE */
  /** Triplet */
  /* Sum */
  v = gsl_vector_alloc(M);
  gsl_spmatrix_get_rowsum(v, tri);
  status = 0;
  for (i = 0; i < M; i++)
    if (v->data[i * v->stride] != denseRowSum->data[i * denseRowSum->stride])
      status = 1;
  gsl_test(status, "test_manip: M=%zu N=%zu _get != _get_rowsum triplet", M, N);
  /* Div */
  test = gsl_spmatrix_alloc_nzmax(crs->size1, crs->size2, 0, GSL_SPMATRIX_TRIPLET);
  gsl_spmatrix_memcpy(test, tri);
  gsl_spmatrix_div_rows(test, v);
  status = 0;
  for (i = 0; i < M; i++)
    {
      for (j = 0; j < N; j++)
	{
	  if (gsl_matrix_get(denseDivRows, i, j) != gsl_spmatrix_get(test, i, j))
	    status = 1;
	}
    }
  gsl_test(status, "test_manip: M=%zu N=%zu _get != _div_rows triplet", M, N);
  gsl_vector_free(v);
  gsl_spmatrix_free(test);

  /** CCS */
  /* Sum */
  v = gsl_vector_alloc(M);
  gsl_spmatrix_get_rowsum(v, ccs);
  status = 0;
  for (i = 0; i < M; i++)
    if (v->data[i * v->stride] != denseRowSum->data[i * denseRowSum->stride])
      status = 1;
  gsl_test(status, "test_manip: M=%zu N=%zu _get != _get_rowsum CCS", M, N);
  /* Div */
  test = gsl_spmatrix_alloc_nzmax(ccs->size1, ccs->size2, 0, GSL_SPMATRIX_CCS);
  gsl_spmatrix_memcpy(test, ccs);
  gsl_spmatrix_div_rows(test, v);
  status = 0;
  for (i = 0; i < M; i++)
    {
      for (j = 0; j < N; j++)
	{
	  if (gsl_matrix_get(denseDivRows, i, j) != gsl_spmatrix_get(test, i, j))
	    status = 1;
	}
    }
  gsl_test(status, "test_manip: M=%zu N=%zu _get != _div_rows CCS", M, N);
  gsl_vector_free(v);
  gsl_spmatrix_free(test);
  
  /* CRS */
  /* Sum */
  v = gsl_vector_alloc(M);
  gsl_spmatrix_get_rowsum(v, crs);
  status = 0;
  for (i = 0; i < M; i++)
    if (v->data[i * v->stride] != denseRowSum->data[i * denseRowSum->stride])
      status = 1;
  gsl_test(status, "test_manip: M=%zu N=%zu _get != _get_rowsum CRS", M, N);
  /* Div */
  test = gsl_spmatrix_alloc_nzmax(crs->size1, crs->size2, 0, GSL_SPMATRIX_CRS);
  gsl_spmatrix_memcpy(test, crs);
  gsl_spmatrix_div_rows(test, v);
  status = 0;
  for (i = 0; i < M; i++)
    {
      for (j = 0; j < N; j++)
	{
	  if (gsl_matrix_get(denseDivRows, i, j) != gsl_spmatrix_get(test, i, j))
	    status = 1;
	}
    }
  gsl_test(status, "test_manip: M=%zu N=%zu _get != _div_rows CRS", M, N);
  gsl_vector_free(v);
  gsl_spmatrix_free(test);


  /** COLUMN SUM AND DIVIDE */
  /** Triplet */
  /* Sum */
  v = gsl_vector_alloc(N);
  gsl_spmatrix_get_colsum(v, tri);
  status = 0;
  for (j = 0; j < N; j++)
    if (v->data[j * v->stride] != denseColSum->data[j * denseColSum->stride])
      status = 1;
  gsl_test(status, "test_manip: M=%zu N=%zu _get != _get_colsum triplet", M, N);
  /* Div */
  test = gsl_spmatrix_alloc_nzmax(tri->size1, tri->size2, 0, GSL_SPMATRIX_TRIPLET);
  gsl_spmatrix_memcpy(test, tri);
  gsl_spmatrix_div_cols(test, v);
  status = 0;
  for (i = 0; i < M; i++)
    {
      for (j = 0; j < N; j++)
	{
	  if (gsl_fcmp(gsl_matrix_get(denseDivCols, i, j), gsl_spmatrix_get(test, i, j), 1.e-12))
	    {
	      fprintf(stdout, "mismatch: (%zu, %zu) %lf != %lf\n", i, j, gsl_matrix_get(denseDivCols, i, j),
		      gsl_spmatrix_get(test, i, j));
	      status = 1;
	    }
	}
    }
  gsl_test(status, "test_manip: M=%zu N=%zu _get != _div_cols triplet", M, N);
  gsl_vector_free(v);
  gsl_spmatrix_free(test);

  /** CCS */
  /** Sum */
  v = gsl_vector_alloc(N);
  gsl_spmatrix_get_colsum(v, ccs);
  status = 0;
  for (j = 0; j < N; j++)
    if (v->data[j * v->stride] != denseColSum->data[j * denseColSum->stride])
      status = 1;
  gsl_test(status, "test_manip: M=%zu N=%zu _get != _get_colsum CCS", M, N);
  /** Div */
  test = gsl_spmatrix_alloc_nzmax(ccs->size1, ccs->size2, 0, GSL_SPMATRIX_CCS);
  gsl_spmatrix_memcpy(test, ccs);
  gsl_spmatrix_div_cols(test, v);
  status = 0;
  for (i = 0; i < M; i++)
    {
      for (j = 0; j < N; j++)
	{
	  if (gsl_matrix_get(denseDivCols, i, j) != gsl_spmatrix_get(test, i, j))
	    status = 1;
	}
    }
  gsl_test(status, "test_manip: M=%zu N=%zu _get != _div_cols CCS", M, N);
  gsl_vector_free(v);
  gsl_spmatrix_free(test);
  
  /** CRS */
  /* Sum */
  v = gsl_vector_alloc(N);
  gsl_spmatrix_get_colsum(v, crs);
  status = 0;
  for (j = 0; j < N; j++)
    if (v->data[j * v->stride] != denseColSum->data[j * denseColSum->stride])
      status = 1;
  gsl_test(status, "test_manip: M=%zu N=%zu _get != _get_colsum CRS", M, N);
  /* Div */
  test = gsl_spmatrix_alloc_nzmax(crs->size1, crs->size2, 0, GSL_SPMATRIX_CRS);
  gsl_spmatrix_memcpy(test, crs);
  gsl_spmatrix_div_cols(test, v);
  status = 0;
  for (i = 0; i < M; i++)
    {
      for (j = 0; j < N; j++)
	{
	  if (gsl_matrix_get(denseDivCols, i, j) != gsl_spmatrix_get(test, i, j))
	    status = 1;
	}
    }
  gsl_test(status, "test_manip: M=%zu N=%zu _get != _div_cols CRS", M, N);
  gsl_vector_free(v);
  gsl_spmatrix_free(test);


  /** Free */
  gsl_spmatrix_free(tri);
  gsl_spmatrix_free(ccs);
  gsl_spmatrix_free(crs);
  gsl_matrix_free(dense);
  gsl_matrix_free(denseDivRows);
  gsl_matrix_free(denseDivCols);
  gsl_vector_free(denseRowSum);
  gsl_vector_free(denseColSum);
  
  return;
}

void
test_io(FILE *inStream, FILE *outStream)
{
  gsl_spmatrix *m;
  int status;

  m = gsl_spmatrix_fscanf(inStream, 1);

  status = gsl_spmatrix_fprintf(outStream, m, "%.3lf");

  gsl_spmatrix_free(m);
  
  return;
}

void
test_compress(const size_t M, const size_t N, const double density,
	      const gsl_rng *r)
{
  int status;
  size_t i, j;
  gsl_spmatrix *m, *ccs, *crs, *ccstr;

  m = create_random_sparse(M, N, density, r);

  // Compress column sum duplicates
  ccs = gsl_spmatrix_compress(m, GSL_SPMATRIX_CCS); 

  // Compress row sum duplicates
  crs = gsl_spmatrix_compress(m, GSL_SPMATRIX_CRS);
  status = 0;
  for (i = 0; i < ccs->size1; i++)
    for (j = 0; j < ccs->size2; j++)
      if (gsl_spmatrix_get(crs, i, j) != gsl_spmatrix_get(ccs, i, j))
	status = 1;
  gsl_test(status, "test_compress: _compress at M=%zu, N=%zu", M, N);

  return;
  // Transpose in place by changing major
  gsl_spmatrix_transpose(crs);
  status = 0;
  for (i = 0; i < crs->size1; i++)
    for (j = 0; j < crs->size2; j++)
      if (gsl_spmatrix_get(crs, i, j) != gsl_spmatrix_get(ccs, j, i))
	status = 1;
  gsl_test(status, "test_compress: transpose inplace at M=%zu, N=%zu", M, N);
  gsl_spmatrix_transpose(crs);


  // Convert by transpose copy
  gsl_spmatrix_switch_major(crs, ccs);
  status = 0;
  for (i = 0; i < ccs->size1; i++)
    for (j = 0; j < ccs->size2; j++)
      if (gsl_spmatrix_get(crs, i, j) != gsl_spmatrix_get(ccs, i, j))
	status = 1;
  gsl_test(status, "test_compress: _switch_major at M=%zu, N=%zu", M, N);

  gsl_spmatrix_free(m);
  gsl_spmatrix_free(ccs);
  gsl_spmatrix_free(crs);
  gsl_spmatrix_free(ccstr);

  return;
}

int
main()
{
  gsl_rng *r = gsl_rng_alloc(gsl_rng_default);
  
  test_memcpy(10, 10, r);
  test_memcpy(10, 15, r);
  test_memcpy(53, 213, r);
  test_memcpy(920, 2, r);
  test_memcpy(2, 920, r);

  test_getset(20, 20, r);
  test_getset(30, 20, r);
  test_getset(15, 210, r);

  test_ops(20, 20, r);
  test_ops(50, 20, r);
  test_ops(20, 50, r);
  test_ops(76, 43, r);

  /** Test compresion (spcompress.c) with duplicates and transposition (spswap.c) */
  test_compress(15, 5, 0.2, r);

  /** Test manipulation (spmanip.c)*/
  test_manip(5, 8, 0.2, r);
  
  /** Test tests (spprop.c) */
  test_prop(5, 5, 0.5, 0.5, r);

  /** Test Input/Output */
  //fprintf(stdout, "Testing Input/Output:\nInput matrix (summing duplicate):\n");
  //test_io(stdin, stdout);

  gsl_rng_free(r);

  exit (gsl_test_summary());
} /* main() */
