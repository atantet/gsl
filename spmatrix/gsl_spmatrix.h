/* gsl_spmatrix.h
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

#ifndef __GSL_SPMATRIX_H__
#define __GSL_SPMATRIX_H__

#include <stdlib.h>

#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

#undef __BEGIN_DECLS
#undef __END_DECLS
#ifdef __cplusplus
# define __BEGIN_DECLS extern "C" {
# define __END_DECLS }
#else
# define __BEGIN_DECLS /* empty */
# define __END_DECLS /* empty */
#endif

__BEGIN_DECLS

/*
 * Binary tree data structure for storing sparse matrix elements
 * in triplet format. This is used for efficiently detecting
 * duplicates and element retrieval via gsl_spmatrix_get
 */
typedef struct
{
  void *tree;       /* tree structure */
  void *node_array; /* preallocated array of tree nodes */
  size_t n;         /* number of tree nodes in use (<= nzmax) */
} gsl_spmatrix_tree;

/*
 * Triplet format:
 *
 * If data[n] = A_{ij}, then:
 *   i = A->i[n]
 *   j = A->p[n]
 *
 * Compressed column format:
 *
 * If data[n] = A_{ij}, then:
 *   i = A->i[n]
 *   A->p[j] <= n < A->p[j+1]
 * so that column j is stored in
 * [ data[p[j]], data[p[j] + 1], ..., data[p[j+1] - 1] ]
 */

typedef struct
{
  size_t size1;     /* number of rows */
  size_t size2;     /* number of columns */
  size_t innerSize; /* AT: size of inner indices: number of rows (CCS) / cols (CRS) */
  size_t outerSize; /* AT: size of outer indices: number of cols (CCS) / rows (CRS) */

  size_t *i;     /* row (triplet,CCS) / column (CRS) indices of size nzmax */
  double *data;  /* matrix elements of size nzmax */

  /*
   * p contains the column indices (triplet) or column pointers (compcol)
   *
   * triplet:   p[n] = column number of element data[n]
   * comp. col: p[j] = index in data of first non-zero element in column j
   * comp. row: p[i] = index in data of first non-zero element in row i
   */
  size_t *p;

  size_t nzmax;  /* maximum number of matrix elements */
  size_t nz;     /* number of non-zero values in matrix */

  gsl_spmatrix_tree *tree_data; /* binary tree for sorting triplet data */

  /*
   * workspace of size MAX(size1,size2)*MAX(sizeof(double),sizeof(size_t))
   * used in various routines
   */
  void *work;

  size_t sptype; /* sparse storage type */
} gsl_spmatrix;

#define GSL_SPMATRIX_TRIPLET      (0)
#define GSL_SPMATRIX_CCS          (1)
#define GSL_SPMATRIX_CRS          (2) /* AT: Add CRS type */

#define GSL_SPMATRIX_ISTRIPLET(m) ((m)->sptype == GSL_SPMATRIX_TRIPLET)
#define GSL_SPMATRIX_ISCCS(m)     ((m)->sptype == GSL_SPMATRIX_CCS)
#define GSL_SPMATRIX_ISCRS(m)     ((m)->sptype == GSL_SPMATRIX_CRS)

/*
 * Prototypes
 */

gsl_spmatrix *gsl_spmatrix_alloc(const size_t n1, const size_t n2);
gsl_spmatrix *gsl_spmatrix_alloc_nzmax(const size_t n1, const size_t n2,
                                       const size_t nzmax, const size_t flags);
void gsl_spmatrix_free(gsl_spmatrix *m);
int gsl_spmatrix_realloc(const size_t nzmax, gsl_spmatrix *m);
int gsl_spmatrix_set_zero(gsl_spmatrix *m);
size_t gsl_spmatrix_nnz(const gsl_spmatrix *m);
int gsl_spmatrix_compare_idx(const size_t ia, const size_t ja,
                             const size_t ib, const size_t jb);

/* spcopy.c */
int gsl_spmatrix_memcpy(gsl_spmatrix *dest, const gsl_spmatrix *src);

/* spgetset.c */
double gsl_spmatrix_get(const gsl_spmatrix *m, const size_t i,
                        const size_t j);
/* AT: possibility to sum duplicates */
int gsl_spmatrix_set(gsl_spmatrix *m, const size_t i, const size_t j,
                     const double x, const int sum_duplicate);

/* spcompress.c */
/* AT: compress either in CCS or CRS */
 gsl_spmatrix *gsl_spmatrix_compress(const gsl_spmatrix *T, const size_t sptype);
void gsl_spmatrix_cumsum(const size_t n, size_t *c);

/* spoper.c */
int gsl_spmatrix_scale(gsl_spmatrix *m, const double x);
int gsl_spmatrix_minmax(const gsl_spmatrix *m, double *min_out,
                        double *max_out);
int gsl_spmatrix_add(gsl_spmatrix *c, const gsl_spmatrix *a,
                     const gsl_spmatrix *b);
int gsl_spmatrix_d2sp(gsl_spmatrix *S, const gsl_matrix *A);
int gsl_spmatrix_sp2d(gsl_matrix *A, const gsl_spmatrix *S);

/* spprop.c */
int gsl_spmatrix_equal(const gsl_spmatrix *a, const gsl_spmatrix *b);
/* AT: tests on matrix elements */
gsl_spmatrix *gsl_spmatrix_gt_elements(const gsl_spmatrix *m, const double d);
gsl_spmatrix *gsl_spmatrix_ge_elements(const gsl_spmatrix *m, const double d);
gsl_spmatrix *gsl_spmatrix_lt_elements(const gsl_spmatrix *m, const double d);
gsl_spmatrix *gsl_spmatrix_le_elements(const gsl_spmatrix *m, const double d);
int gsl_spmatrix_any(const gsl_spmatrix *m);

/* spswap.c */
int gsl_spmatrix_transpose_memcpy(gsl_spmatrix *dest, const gsl_spmatrix *src);
/* AT: transpose in place by changing type and type conversion */
int gsl_spmatrix_transpose(gsl_spmatrix *src);
gsl_spmatrix *gsl_spmatrix_switch_major(gsl_spmatrix *src);

/* spmanip.c */
/* AT: useful functions to study Markov chain transition matrices */
gsl_vector *gsl_spmatrix_get_rowsum(const gsl_spmatrix *m);
gsl_vector *gsl_spmatrix_get_colsum(const gsl_spmatrix *m);
double gsl_spmatrix_get_sum(const gsl_spmatrix *m);
int gsl_spmatrix_div_rows(gsl_spmatrix *m, const gsl_vector *v);
int gsl_spmatrix_div_cols(gsl_spmatrix *m, const gsl_vector *v);

/* spio.c */
/* AT: input/output functions */
int gsl_spmatrix_fprintf(FILE *stream, const gsl_spmatrix *m, const char *format);
 gsl_spmatrix *gsl_spmatrix_fscanf(FILE *stream, const int sum_duplicate);


__END_DECLS

#endif /* __GSL_SPMATRIX_H__ */
