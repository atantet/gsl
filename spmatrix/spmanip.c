#include <stdlib.h>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_vector.h>

#include "gsl_spmatrix.h"


/** \file spmanip.c 
 * \brief GSL sparse matrix manipulation.
 *
 * GSL sparse matrix manipulation.
 */


/** \brief Get the sum of the elements of a compressed matrix over each row.
 * 
 * Get the sum of the elements of a compressed matrix over each row.
 * \param[in] m Compressed matrix over which to sum.
 * \return      Vector of the sum of the rows.
 */
gsl_vector *
gsl_spmatrix_get_rowsum(const gsl_spmatrix *m)
{
  size_t n, outerIdx, p;
  gsl_vector *sum = gsl_vector_calloc(m->size1);;
    
  if (GSL_SPMATRIX_ISTRIPLET(m))
    {
      for (n = 0; n < m->nz; n++)
	{
	  sum->data[m->i[n] * sum->stride] += m->data[n];
	}
    }
  else if (GSL_SPMATRIX_ISCCS(m))
    {
      for (outerIdx = 0; outerIdx < m->outerSize; outerIdx++)
	{
	  for (p = m->p[outerIdx]; p < m->p[outerIdx + 1]; ++p)
	    {
	      sum->data[m->i[p] * sum->stride] += m->data[p];
	    }
	}
    }
  else if (GSL_SPMATRIX_ISCRS(m))
    {
      for (outerIdx = 0; outerIdx < m->outerSize; outerIdx++)
	{
	  for (p = m->p[outerIdx]; p < m->p[outerIdx + 1]; ++p)
	    {
	      sum->data[outerIdx * sum->stride] += m->data[p];
	    }
	}
    }
  else
    {
      GSL_ERROR_NULL("unknown sparse matrix type", GSL_EINVAL);
    }


  return sum;
}

/** \brief Get the sum of the elements of a compressed matrix over each column.
 * 
 * Get the sum of the elements of a compressed matrix over each column.
 * \param[in] m Compressed matrix over which to sum.
 * \return      Vector of the sum of the columns.
 */
gsl_vector *
gsl_spmatrix_get_colsum(const gsl_spmatrix *m)
{
  size_t outerIdx, p, n;
  gsl_vector *sum = gsl_vector_calloc(m->size2);
    
  if (GSL_SPMATRIX_ISTRIPLET(m))
    {
      for (n = 0; n < m->nz; n++)
	{
	  sum->data[m->p[n] * sum->stride] += m->data[n];
	}
    }
  else if (GSL_SPMATRIX_ISCCS(m))
    {
      for (outerIdx = 0; outerIdx < m->outerSize; outerIdx++)
	{
	  for (p = m->p[outerIdx]; p < m->p[outerIdx + 1]; ++p)
	    {
	      sum->data[outerIdx * sum->stride] += m->data[p];
	    }
	}
    }
  else if (GSL_SPMATRIX_ISCRS(m))
    {
      for (outerIdx = 0; outerIdx < m->outerSize; outerIdx++)
	{
	  for (p = m->p[outerIdx]; p < m->p[outerIdx + 1]; ++p)
	    {
	      sum->data[m->i[p] * sum->stride] += m->data[p];
	    }
	}
    }
  else
    {
      GSL_ERROR_NULL("unknown sparse matrix type", GSL_EINVAL);
    }
  
  return sum;
}

/** 
 * \brief Get the sum of the elements of a sparse matrix.
 *
 * Get the sum of the elements of a sparse matrix.
 * \param[in] m    Sparse matrix over which to sum.
 * \return         Sum over all the elements of the sparse matrix.
 */
double
gsl_spmatrix_get_sum(const gsl_spmatrix *m)
{
  double sum = 0.;
  size_t n;

  for (n = 0; n < m->nz; n++)
    {
      sum += m->data[n];
    }

  return sum;
}

/** 
 * \brief Divide each row of a compressed matrix by a vector.
 *
 * Divide each row of a compressed matrix by the corresponding vector element.
 * \param[in] m    compressed matrix to divide.
 * \param[in] v    vector to divide the rows of the sparse matrix.
 * \return         Exit status.
 */
int
gsl_spmatrix_div_rows(gsl_spmatrix *m, const gsl_vector *v)
{
  size_t outerIdx, p, n;

  if (GSL_SPMATRIX_ISTRIPLET(m))
    {
      for (n = 0; n < m->nz; n++)
	{
	  if (gsl_pow_2(v->data[m->i[n] * v->stride]) > 1.e-12)
	    {
	      m->data[n] /= v->data[m->i[n] * v->stride];
	    }
	}
    }
  else if (GSL_SPMATRIX_ISCCS(m))
    {
      for (outerIdx = 0; outerIdx < m->outerSize; outerIdx++)
	{
	  for (p = m->p[outerIdx]; p < m->p[outerIdx + 1]; ++p)
	    {
	      if (gsl_pow_2(v->data[m->i[p] * v->stride]) > 1.e-12)
		{
		  m->data[p] /= v->data[m->i[p] * v->stride];
		}
	    }
	}
    }
  else if (GSL_SPMATRIX_ISCRS(m))
    {
      for (outerIdx = 0; outerIdx < m->outerSize; outerIdx++)
	{
	  for (p = m->p[outerIdx]; p < m->p[outerIdx + 1]; ++p)
	    {
	      if (gsl_pow_2(v->data[outerIdx * v->stride]) > 1.e-12)
		{
		  m->data[p] /= v->data[outerIdx * v->stride];
		}
	    }
	}
    }
  else
    {
      GSL_ERROR("unknown sparse matrix type", GSL_EINVAL);
    }

  return GSL_SUCCESS;
}

/** 
 * \brief Divide each column of a compressed matrix by a vector.
 *
 * Divide each column of a compressed matrix by the corresponding vector element.
 * \param[in] m    compressed matrix to divide.
 * \param[in] v    vector to divide the columns of the sparse matrix.
 * \return         Exit status.
 */
int
gsl_spmatrix_div_cols(gsl_spmatrix *m, const gsl_vector *v)
{
  size_t outerIdx, p, n;

  if (GSL_SPMATRIX_ISTRIPLET(m))
    {
      for (n = 0; n < m->nz; n++)
	{
	  if (gsl_pow_2(v->data[m->p[n] * v->stride]) > 1.e-12)
	    {
	      m->data[n] = m->data[n] / v->data[m->p[n] * v->stride];
	    }
	}
    }
  else if (GSL_SPMATRIX_ISCCS(m))
    {
      for (outerIdx = 0; outerIdx < m->outerSize; outerIdx++)
	{
	  for (p = m->p[outerIdx]; p < m->p[outerIdx + 1]; ++p)
	    {
	      if (gsl_pow_2(v->data[outerIdx * v->stride]) > 1.e-12)
		{
		  m->data[p] /= v->data[outerIdx * v->stride];
		}
	    }
	}
    }
  else if (GSL_SPMATRIX_ISCRS(m))
    {
      for (outerIdx = 0; outerIdx < m->outerSize; outerIdx++)
	{
	  for (p = m->p[outerIdx]; p < m->p[outerIdx + 1]; ++p)
	    {
	      if (gsl_pow_2(v->data[m->i[p] * v->stride]) > 1.e-12)
		{
		  m->data[p] /= v->data[m->i[p] * v->stride];
		}
	    }
	}
    }
  else
    {
      GSL_ERROR("unknown sparse matrix type", GSL_EINVAL);
    }

  return GSL_SUCCESS;
}
