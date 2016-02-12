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
 * \brief Write sparse matrix in MatrixMarket coordinate format.
 *
 * Write sparse matrix in MatrixMarket coordinate format.
 * The first line is a header with the matrix type.
 * The first line after a % delimited comment section
 * gives the numer of rows, columns and non-zero elements
 * of the matrix: "size1 size2 nz".
 * The following nz lines give the matrix elements 
 * with row, column and value: "i j mij"
 * \param[in] stream Opened stream for writing.
 * \param[in] m      Sparse matrix to write.
 * \param[in] format Format specifier.
 */
int
gsl_spmatrix_fprintf(FILE *stream, const gsl_spmatrix *m, const char *format)
{
  size_t n, outerIdx, p;
  char formatStr[256];
  const char headerMM[] = "%%MatrixMarket";
  const char object[] = "matrix";
  const char matrixFormat[] = "coordinate";
  /**const char matrixFormat[] = "array";*/
  const char dataType[] = "real";
  /**const char dataType[] = "integer";*/
  /**const char dataType[] = "complex";*/
  /**const char dataType[] = "pattern";*/
  const char structure[] = "general";
  /**const char structure[] = "symmetric";*/
  /**const char structure[] = "skew-symmetric";*/
  /**const char structure[] = "Hermitian";*/

  /** Make element format string out of data format specifier */
  sprintf(formatStr, "%s%s%s", "%zu\t%zu\t", format, "\n");

  /** Print one-line header */
  if (fprintf(stream, "%s %s %s %s %s\n", headerMM, object, matrixFormat, dataType, structure) <= 0)
    {
      GSL_ERROR("Not all arguments printed for header", GSL_EFAILED);
    }

  /** Print dimensions header */
  if (fprintf(stream, "%zu\t%zu\t%zu\n", m->size1, m->size2, m->nz) <= 0)
    {
      GSL_ERROR("Not all arguments printed for matrix dimension line", GSL_EFAILED);
    }

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
 * \brief Read sparse matrix in MatrixMarket coordinate format.
 *
 * Read sparse matrix in MatrixMarket coordinate format.
 * The first line is a header with the matrix type.
 * The first line after a % delimited comment section
 * gives the numer of rows, columns and non-zero elements
 * of the matrix: "size1 size2 nz".
 * The following nz lines give the matrix elements 
 * with row, column and value: "i j mij"
 * \param[in] stream Opened stream for reading.
 * \param[in] m      Sparse matrix to read.
 * \param[in] format Format specifier.
 */
gsl_spmatrix *
gsl_spmatrix_fscanf(FILE *stream, const int sum_duplicate)
{
  size_t n;
  size_t M, N, nz;
  size_t i, j;
  double x;
  gsl_spmatrix *m;
  char buffer[256];

  /** Skip comments */
  fgets(buffer, 256, stream);
  while (buffer[0] == '%')
    fgets(buffer, 256, stream);

  /** Read one-line header and allocate */
  if (sscanf(buffer, "%zu %zu %zu", &M, &N, &nz) < 3)
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
