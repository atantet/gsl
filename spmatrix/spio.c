#include <stdlib.h>
#include <stdio.h>

#include <gsl/gsl_errno.h>

#include "gsl_spmatrix.h"


/** \file spio.c
 * \brief GSL sparse matrix input/output functions.
 *
 * GSL sparse matrix input/output functions.
 */

/**
 * \brief Write sparse matrix as triplet list to stream
 *
 * Write sparse matrix to stream as triplet list "i_k, j_k, d_k"
 * with a one line header "size1, size2, nz".
 * \param[in] stream Opened stream for writing.
 * \param[in] m      Sparse matrix to write.
 * \param[in] format Format specifier.
 */
int
gsl_spmatrix_fprintf(FILE *stream, const gsl_spmatrix *m, const char *format)
{
  size_t n, outerIdx, p;
  char formatStr[256];

  /** Make element format string out of data format specifier */
  sprintf(formatStr, "%s%s%s", "%zu\t%zu\t", format, "\n");

  /** Print one-line header */
  if (fprintf(stream, "%zu\t%zu\t%zu\n", m->size1, m->size2, m->nz) <= 0)
    GSL_ERROR("Not all arguments printed for header", GSL_EFAILED);

  /** Print elements */
  if (GSL_SPMATRIX_ISTRIPLET(m))
    {
      for (n = 0; n < m->nz; n++)
	{
	  fprintf(stream, formatStr, m->i[n], m->p[n], m->data[n]);
	}
    }
  else if (GSL_SPMATRIX_ISCCS(m))
    {
      for (outerIdx = 0; outerIdx < m->outerSize; outerIdx++)
	{
	  for (p = m->p[outerIdx]; p < m->p[outerIdx + 1]; ++p)
	    {
	      fprintf(stream, formatStr, m->i[p], outerIdx, m->data[p]);
	    }
	}
    }
  else if  (GSL_SPMATRIX_ISCRS(m))
    {
      for (outerIdx = 0; outerIdx < m->outerSize; outerIdx++)
	{
	  for (p = m->p[outerIdx]; p < m->p[outerIdx + 1]; ++p)
	    {
	      fprintf(stream, formatStr, outerIdx, m->i[p], m->data[p]);
	    }
	}
    }
  else
    {
      GSL_ERROR_NULL("unknown sparse matrix type", GSL_EINVAL);
    }

  if (ferror(stream))
    {
      GSL_ERROR("Not all arguments printed for elements", GSL_EFAILED);
    }

  return GSL_SUCCESS;
}

/**
 * \brief Read sparse matrix as triplet list from stream
 *
 * Read sparse matrix from stream as triplet list "i_k, j_k, d_k"
 * with a one line header "size1, size2, nz".
 * The matrix should not be allocated, since its dimensions
 * and number of non-zero elements is read from the header.
 * \param[in]  stream Opened stream for reading.
 * \return     Sparse matrix scanned.
 */
gsl_spmatrix *
gsl_spmatrix_fscanf(FILE *stream, const int sum_duplicate)
{
  size_t n;
  size_t M, N, nz;
  size_t i, j;
  double x;
  gsl_spmatrix *m;

  /** Read one-line header and allocate */
  if (fscanf(stream, "%zu %zu %zu", &M, &N, &nz) < 3)
    {
      GSL_ERROR_NULL("Not all arguments scanned for header", GSL_EFAILED);
    }
  m = gsl_spmatrix_alloc_nzmax(M, N, nz, GSL_SPMATRIX_TRIPLET);
  
  /** Add all elements */
  for (n = 0; n < nz; ++n)
    {
      if (fscanf(stream, "%zu %zu %lf", &i, &j, &x) < 3)
	{
	  GSL_ERROR_NULL("Not all arguments scanned for element", GSL_EFAILED);
	}
      gsl_spmatrix_set(m, i, j, x, sum_duplicate);
    }
  
  return m;
}
